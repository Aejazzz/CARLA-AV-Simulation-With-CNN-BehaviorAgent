#!/usr/bin/env python
"""CARLA Behavioral Cloning - Data Collector.

Drive manually in CARLA. Three RGB cameras (left/center/right) plus a CSV log
of the human's control inputs are saved to disk while recording is enabled.
The resulting dataset is consumed by ``bc_train.py``.

This script mirrors the threading model of ``human_control.py`` so the driving
feel is identical:
  * Asynchronous mode by default - the server runs at its own rate, the main
    loop never blocks waiting for sensor data.
  * Each camera writes its latest frame from its own callback (background
    thread); the main loop just reads whatever frame is freshest.
  * The display refreshes at 60 fps for smooth driving feedback.
  * Recording is decoupled and runs on its own 20 Hz timer; image encoding and
    disk writes happen in a background thread, so pressing R never affects
    the simulation responsiveness.

Run CARLA first (``CarlaUE4.exe``), then in a separate shell:

    python bc_data_collector.py --output dataset/run1

Manual keyboard driving is limited to ``--max-speed`` km/h (default 40); above
that, throttle is cut and brake is applied. CARLA autopilot (``P``) is not capped.

Controls
--------
    W / UP        : throttle
    S / DOWN      : brake
    A / LEFT      : steer left
    D / RIGHT     : steer right
    Q             : toggle reverse
    SPACE         : hand-brake
    P             : toggle autopilot  (you can record autopilot demos too)
    R             : toggle dataset recording  (red REC indicator when on)
    BACKSPACE     : respawn vehicle at a new spawn point
    C / SHIFT+C   : next / previous weather preset
    ESC           : quit

Output layout
-------------
    <output>/center/00000123.jpg
    <output>/left/00000123.jpg
    <output>/right/00000123.jpg
    <output>/labels.csv      (frame, steer, throttle, brake, reverse,
                              hand_brake, speed_kmh)
"""

from __future__ import print_function

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import argparse
import csv
import datetime
import logging
import math
import queue
import random
import re
import threading
import time
import weakref

try:
    import numpy as np
except ImportError:
    raise RuntimeError('numpy is required (pip install numpy)')

try:
    from PIL import Image as PILImage
except ImportError:
    raise RuntimeError('Pillow is required (pip install Pillow)')

try:
    import pygame
    from pygame.locals import (
        KMOD_CTRL, KMOD_SHIFT,
        K_BACKSPACE, K_DOWN, K_ESCAPE, K_LEFT, K_RIGHT, K_SPACE,
        K_UP, K_a, K_c, K_d, K_p, K_q, K_r, K_s, K_w,
    )
except ImportError:
    raise RuntimeError('pygame is required (pip install pygame)')


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Capture resolution per training camera. Small for fast I/O; bc_train.py
# resizes to 200x66 (PilotNet) before training.
CAM_W = 320
CAM_H = 180
CAM_FOV = 90

# Mounting positions relative to vehicle (x = forward, y = right, z = up).
CAMERA_TRANSFORMS = {
    'center': carla.Transform(carla.Location(x=1.5, y=0.0, z=1.7)),
    'left':   carla.Transform(carla.Location(x=1.5, y=-0.5, z=1.7)),
    'right':  carla.Transform(carla.Location(x=1.5, y=0.5, z=1.7)),
}

# Manual driving: soften acceleration and peak speed (~30% lower than full W).
MANUAL_THROTTLE_CAP = 0.7
MANUAL_THROTTLE_RAMP = 0.1 * MANUAL_THROTTLE_CAP

# Ground speed ceiling for manual keyboard driving (CARLA autopilot not clamped).
DEFAULT_TOP_SPEED_KMH = 40.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_speed_kmh(vehicle):
    v = vehicle.get_velocity()
    return 3.6 * math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def carla_image_to_rgb_array(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    return arr[:, :, :3][:, :, ::-1].copy()  # BGRA -> RGB


def spawn_vehicle(world, filter_str):
    bp_lib = world.get_blueprint_library()
    candidates = bp_lib.filter(filter_str)
    if not candidates:
        raise RuntimeError('No blueprints match filter %r' % filter_str)
    bp = random.choice(candidates)
    bp.set_attribute('role_name', 'bc_hero')
    if bp.has_attribute('color'):
        bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError('Map has no spawn points.')
    random.shuffle(spawn_points)
    for sp in spawn_points:
        actor = world.try_spawn_actor(bp, sp)
        if actor is not None:
            return actor
    raise RuntimeError('Could not spawn vehicle at any spawn point.')


def destroy_actors(actors):
    for a in actors:
        if a is None:
            continue
        try:
            if hasattr(a, 'stop'):
                a.stop()
        except Exception:
            pass
        try:
            a.destroy()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Latest-frame holders (thread-safe). Cameras write here from their callback;
# the main loop and the recording timer read from here. NEVER blocks.
# ---------------------------------------------------------------------------

class LatestFrame(object):
    # __weakref__ required so sensor callbacks can use weakref.ref(self)
    __slots__ = ('_lock', 'frame', 'rgb', 'surface', '__weakref__')

    def __init__(self):
        self._lock = threading.Lock()
        self.frame = None
        self.rgb = None       # RGB ndarray (H, W, 3) uint8 - for ML / disk
        self.surface = None   # pygame Surface - for display only

    def set_rgb(self, frame, rgb):
        with self._lock:
            self.frame = frame
            self.rgb = rgb

    def set_surface(self, frame, surface):
        with self._lock:
            self.frame = frame
            self.surface = surface

    def snapshot_rgb(self):
        with self._lock:
            return self.frame, self.rgb

    def snapshot_surface(self):
        with self._lock:
            return self.surface


def _make_training_callback(holder):
    """Returns a CARLA sensor.listen() callback that updates a LatestFrame
    with an RGB numpy array. Runs on a CARLA worker thread."""
    weak = weakref.ref(holder)
    def cb(image):
        h = weak()
        if h is None:
            return
        rgb = carla_image_to_rgb_array(image)
        h.set_rgb(image.frame, rgb)
    return cb


def _make_display_callback(holder):
    """CARLA callback for the chase camera. Builds a pygame Surface so the
    main loop just blits without doing any per-frame conversion work."""
    weak = weakref.ref(holder)
    def cb(image):
        h = weak()
        if h is None:
            return
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        rgb = arr[:, :, :3][:, :, ::-1]  # BGRA -> RGB
        # pygame surfarray expects (W, H, 3); .swapaxes is a view, not a copy.
        surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        h.set_surface(image.frame, surf)
    return cb


def setup_training_cameras(world, vehicle):
    """Spawn the 3 training cameras. Each writes its latest frame into a
    LatestFrame holder via its own callback (background thread)."""
    bp = world.get_blueprint_library().find('sensor.camera.rgb')
    bp.set_attribute('image_size_x', str(CAM_W))
    bp.set_attribute('image_size_y', str(CAM_H))
    bp.set_attribute('fov', str(CAM_FOV))
    sensors = {}
    holders = {}
    for name, tf in CAMERA_TRANSFORMS.items():
        sensor = world.spawn_actor(bp, tf, attach_to=vehicle)
        holder = LatestFrame()
        sensor.listen(_make_training_callback(holder))
        sensors[name] = sensor
        holders[name] = holder
    return sensors, holders


def setup_display_camera(world, vehicle, width, height, gamma=2.2):
    """Spawn the standard CARLA chase camera (TPP). Transform / attachment
    match human_control.py's CameraManager._camera_transforms[0]."""
    bound_x = 0.5 + vehicle.bounding_box.extent.x
    bound_y = 0.5 + vehicle.bounding_box.extent.y
    bound_z = 0.5 + vehicle.bounding_box.extent.z
    transform = carla.Transform(
        carla.Location(x=-2.0 * bound_x, y=+0.0 * bound_y, z=2.0 * bound_z),
        carla.Rotation(pitch=8.0))
    attachment = carla.AttachmentType.SpringArmGhost

    bp = world.get_blueprint_library().find('sensor.camera.rgb')
    bp.set_attribute('image_size_x', str(width))
    bp.set_attribute('image_size_y', str(height))
    if bp.has_attribute('gamma'):
        bp.set_attribute('gamma', str(gamma))
    sensor = world.spawn_actor(bp, transform, attach_to=vehicle,
                               attachment_type=attachment)
    holder = LatestFrame()
    sensor.listen(_make_display_callback(holder))
    return sensor, holder


# ---------------------------------------------------------------------------
# Asynchronous dataset writer.
#
# JPEG encoding + disk write run in their own thread, so the simulation is
# fully isolated from disk speed. The main loop (or, more precisely, the
# recording timer in the main loop) only enqueues already-decoded RGB arrays.
# ---------------------------------------------------------------------------

CSV_HEADER = ['frame', 'steer', 'throttle', 'brake',
              'reverse', 'hand_brake', 'speed_kmh']


class DatasetWriter(object):
    def __init__(self, output_dir, jpeg_quality=85, queue_size=400):
        self.output_dir = output_dir
        for sub in CAMERA_TRANSFORMS:
            os.makedirs(os.path.join(output_dir, sub), exist_ok=True)
        self.csv_path = os.path.join(output_dir, 'labels.csv')
        is_new = not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0
        self._csv_file = open(self.csv_path, 'a', newline='')
        self._csv = csv.writer(self._csv_file)
        if is_new:
            self._csv.writerow(CSV_HEADER)
        self.enabled = False
        self.count = 0
        self.dropped = 0
        self.jpeg_quality = jpeg_quality

        self._image_queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._writer_thread = threading.Thread(
            target=self._writer_loop, name='bc-image-writer', daemon=True)
        self._writer_thread.start()

    def _writer_loop(self):
        while True:
            try:
                item = self._image_queue.get(timeout=0.5)
            except queue.Empty:
                if self._stop.is_set():
                    break
                continue
            if item is None:
                self._image_queue.task_done()
                break
            path, rgb = item
            try:
                PILImage.fromarray(rgb).save(
                    path, format='JPEG', quality=self.jpeg_quality, optimize=False)
            except Exception as e:
                logging.warning('Image save failed for %s: %s', path, e)
            self._image_queue.task_done()

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

    def write_rgb(self, frame, rgbs, control, speed_kmh):
        """Enqueue images for async save and write the matching CSV row."""
        for name, rgb in rgbs.items():
            path = os.path.join(self.output_dir, name, '{:08d}.jpg'.format(frame))
            try:
                self._image_queue.put_nowait((path, rgb))
            except queue.Full:
                self.dropped += 1
        self._csv.writerow([
            frame,
            float(control.steer),
            float(control.throttle),
            float(control.brake),
            int(bool(control.reverse)),
            int(bool(control.hand_brake)),
            float(speed_kmh),
        ])
        self.count += 1
        if self.count % 50 == 0:
            self._csv_file.flush()

    def close(self):
        try:
            try:
                self._image_queue.put(None, timeout=1.0)
            except queue.Full:
                pass
            self._stop.set()
            self._writer_thread.join(timeout=15.0)
            self._csv_file.flush()
            self._csv_file.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Keyboard control (vehicles only) - same feel as human_control.py
# ---------------------------------------------------------------------------

class KeyboardControl(object):
    def __init__(self, vehicle, start_in_autopilot, max_speed_kmh):
        self.vehicle = vehicle
        self.control = carla.VehicleControl()
        self.autopilot = start_in_autopilot
        self.max_speed_kmh = float(max_speed_kmh)
        vehicle.set_autopilot(self.autopilot)
        self._steer_cache = 0.0

    def parse(self, milliseconds):
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True, None
            if event.type == pygame.KEYUP:
                if event.key == K_ESCAPE or (event.key == K_q and pygame.key.get_mods() & KMOD_CTRL):
                    return True, None
                if event.key == K_p:
                    self.autopilot = not self.autopilot
                    self.vehicle.set_autopilot(self.autopilot)
                    action = 'TOGGLE_AUTOPILOT'
                elif event.key == K_q:
                    self.control.gear = 1 if self.control.reverse else -1
                    self.control.reverse = self.control.gear < 0
                elif event.key == K_r:
                    action = 'TOGGLE_RECORD'
                elif event.key == K_BACKSPACE:
                    action = 'RESPAWN'
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    action = 'WEATHER_PREV'
                elif event.key == K_c:
                    action = 'WEATHER_NEXT'

        keys = pygame.key.get_pressed()
        if keys[K_UP] or keys[K_w]:
            self.control.throttle = min(
                self.control.throttle + MANUAL_THROTTLE_RAMP, MANUAL_THROTTLE_CAP)
        else:
            self.control.throttle = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self.control.brake = min(self.control.brake + 0.2, 1.0)
        else:
            self.control.brake = 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache = 0.0 if self._steer_cache > 0 else self._steer_cache - steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache = 0.0 if self._steer_cache < 0 else self._steer_cache + steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = max(-0.7, min(0.7, self._steer_cache))
        self.control.steer = float(self._steer_cache)
        self.control.hand_brake = bool(keys[K_SPACE])

        if not self.autopilot:
            speed = get_speed_kmh(self.vehicle)
            if speed > self.max_speed_kmh:
                self.control.throttle = 0.0
                over = speed - self.max_speed_kmh
                self.control.brake = max(
                    self.control.brake, min(1.0, 0.3 + 0.03 * over))
            self.vehicle.apply_control(self.control)
        return False, action

    def current_control(self):
        if self.autopilot:
            return self.vehicle.get_control()
        return self.control


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------

def render_hud(surface, font, info_lines, recording, autopilot):
    overlay = pygame.Surface((surface.get_width(), 22 * (len(info_lines) + 1) + 10))
    overlay.set_alpha(160)
    overlay.fill((0, 0, 0))
    surface.blit(overlay, (0, 0))
    y = 6
    for line in info_lines:
        surface.blit(font.render(line, True, (255, 255, 255)), (10, y))
        y += 22
    if recording:
        pygame.draw.circle(surface, (220, 30, 30), (surface.get_width() - 24, 24), 9)
        surface.blit(font.render('REC', True, (255, 255, 255)),
                     (surface.get_width() - 70, 16))
    if autopilot:
        surface.blit(font.render('AUTOPILOT', True, (40, 220, 120)),
                     (surface.get_width() - 130, 40))


# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------

def game_loop(args):
    pygame.init()
    pygame.font.init()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    sim_world = client.get_world()

    # Run in async mode just like human_control.py default. Don't touch the
    # server settings, so the world feels exactly the same.

    weather_presets = find_weather_presets()
    weather_idx = 0

    vehicle = None
    train_sensors = {}
    train_holders = {}
    display_sensor = None
    display_holder = None
    writer = None

    record_dt = max(0.01, args.record_dt)
    last_record_time = 0.0

    try:
        vehicle = spawn_vehicle(sim_world, args.filter)
        train_sensors, train_holders = setup_training_cameras(sim_world, vehicle)
        display_sensor, display_holder = setup_display_camera(
            sim_world, vehicle, args.width, args.height)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption('CARLA - BC Data Collector')
        font = pygame.font.SysFont('couriernew', 16, bold=True)

        writer = DatasetWriter(args.output)
        controller = KeyboardControl(vehicle, args.autopilot, args.max_speed_kmh)
        clock = pygame.time.Clock()

        print('[BC] Ready. Drive with WASD. Press R to start recording.')
        print('[BC] Output:', os.path.abspath(args.output))
        print('[BC] Recording rate: {:.0f} Hz'.format(1.0 / record_dt))

        while True:
            # Pure pygame loop - never blocks on the simulator.
            ms = clock.tick_busy_loop(60)

            quit_flag, action = controller.parse(ms)
            if quit_flag:
                break

            if action == 'TOGGLE_RECORD':
                state = writer.toggle()
                last_record_time = 0.0
                print('[BC] Recording:', 'ON' if state else 'OFF',
                      '(total saved: {})'.format(writer.count))
            elif action == 'RESPAWN':
                destroy_actors(list(train_sensors.values()))
                destroy_actors([display_sensor])
                destroy_actors([vehicle])
                vehicle = spawn_vehicle(sim_world, args.filter)
                train_sensors, train_holders = setup_training_cameras(sim_world, vehicle)
                display_sensor, display_holder = setup_display_camera(
                    sim_world, vehicle, args.width, args.height)
                controller = KeyboardControl(
                    vehicle, controller.autopilot, args.max_speed_kmh)
                continue
            elif action in ('WEATHER_NEXT', 'WEATHER_PREV'):
                step = -1 if action == 'WEATHER_PREV' else 1
                weather_idx = (weather_idx + step) % len(weather_presets)
                preset = weather_presets[weather_idx]
                sim_world.set_weather(preset[0])
                print('[BC] Weather:', preset[1])
            elif action == 'TOGGLE_AUTOPILOT':
                print('[BC] Autopilot:', 'ON' if controller.autopilot else 'OFF')

            # ----- Recording timer (decoupled from rendering) -----
            now = time.time()
            if writer.enabled and (now - last_record_time) >= record_dt:
                center_frame, center_rgb = train_holders['center'].snapshot_rgb()
                _, left_rgb = train_holders['left'].snapshot_rgb()
                _, right_rgb = train_holders['right'].snapshot_rgb()
                if center_rgb is not None and left_rgb is not None and right_rgb is not None:
                    last_record_time = now
                    rgbs = {'center': center_rgb, 'left': left_rgb, 'right': right_rgb}
                    writer.write_rgb(
                        center_frame,
                        rgbs,
                        controller.current_control(),
                        get_speed_kmh(vehicle))

            # ----- Display (60 fps, just blit the latest chase frame) -----
            disp_surface = display_holder.snapshot_surface()
            if disp_surface is not None:
                display.blit(disp_surface, (0, 0))
            else:
                display.fill((0, 0, 0))

            ctrl = controller.current_control()
            speed = get_speed_kmh(vehicle)
            info = [
                'Server FPS: {:>4.0f}'.format(clock.get_fps()),
                'Speed:      {:>6.1f} km/h (cap {:.0f})'.format(
                    speed, controller.max_speed_kmh),
                'Steer:      {:>+6.3f}'.format(ctrl.steer),
                'Throttle:   {:>6.2f}'.format(ctrl.throttle),
                'Brake:      {:>6.2f}'.format(ctrl.brake),
                'Reverse:    {}'.format('Y' if ctrl.reverse else 'N'),
                'Saved:      {:>8d} samples'.format(writer.count),
                'Dropped:    {:>8d} frames'.format(writer.dropped),
            ]
            render_hud(display, font, info, writer.enabled, controller.autopilot)
            pygame.display.flip()

    finally:
        if writer is not None:
            writer.close()
        destroy_actors(list(train_sensors.values()))
        destroy_actors([display_sensor])
        destroy_actors([vehicle])
        pygame.quit()
        print('[BC] Done. Samples saved this session:',
              writer.count if writer else 0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='CARLA Behavioral Cloning Data Collector')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--res', default='1280x720', help='display resolution WxH')
    parser.add_argument('--filter', default='vehicle.tesla.model3',
                        help='vehicle blueprint filter')
    parser.add_argument('--output',
                        default='dataset/run_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                        help='output directory for images and labels.csv')
    parser.add_argument('--record-dt', type=float, default=0.05,
                        help='dataset sample period in seconds (0.05 = 20 Hz)')
    parser.add_argument('-a', '--autopilot', action='store_true',
                        help='start with CARLA autopilot enabled')
    parser.add_argument('--max-speed', type=float, default=DEFAULT_TOP_SPEED_KMH,
                        dest='max_speed_kmh',
                        help='manual driving: ground speed ceiling in km/h (default: %(default)s)')
    args = parser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    logging.info('Connecting to CARLA at %s:%s', args.host, args.port)
    print(__doc__)

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user.')


if __name__ == '__main__':
    main()
