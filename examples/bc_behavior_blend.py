#!/usr/bin/env python
"""Blend CARLA Planning Agent + Behavioral-Cloning CNN (new standalone client).

Combines navigation from ``agents.navigation.BehaviorAgent`` (same agent family as
``automatic_control.py``: traffic lights, speed limits, path to a goal) with
steer/throttle/brake hints from ``bc_train.py`` / PilotNet running on the
narrow front RGB camera used during data collection.

* **`--fusion` modes**
  - ``cnn_steering`` (recommended): **CNN steering only**; throttle, brake,
    gear, reverse, hand brake come from ``BehaviorAgent`` (lights, TrafficManager,
    path following).
  - ``cnn_steering_safe``: same, but ``brake = max(agent, CNN)``.
  - ``blend``: mix **steering only** — ``w`` interpolates CNN vs agent **lateral**;
    **throttle/brake always from agent** so traffic rules (lights, stops, limits)
    stay intact. ``w=1`` → CNN steer + agent pedals; ``w=0`` → pure agent.
  - ``blend_full``: experimental — ``w`` mixes CNN and agent on **steer, throttle,
    and brake`` (CNN dilutes stopping at lights; not recommended for traffic).

Existing repo files are **not** modified; this script only imports shared BC
modules ``bc_dataset`` / ``bc_model`` and CARLA ``agents``.

Run CARLA first, then (from ``PythonAPI/examples``)::

    python bc_behavior_blend.py --model models/bc.pt --fusion cnn_steering

Keys
----
    F         : cycle fusion modes (see --fusion)
    [ / ]     : blend weight (``blend`` and ``blend_full`` only)
    M         : toggle manual keyboard driving (WASD)
    R         : respawn ego + replan to destination
    L         : pick a new random destination (when --loop)
    ESC       : quit
"""

from __future__ import print_function

import argparse
import glob
import logging
import math
import os
import random
import queue
import sys
import time

import numpy as np
import torch

import pygame
from pygame.locals import (
    KMOD_CTRL, K_ESCAPE, K_DOWN, K_LEFT, K_RIGHT, K_UP,
    K_a, K_d, K_LEFTBRACKET, K_m, K_q, K_r, K_s, K_w,
    K_RIGHTBRACKET, K_l, K_f,
)

FUSION_MODES_ORDER = (
    'cnn_steering', 'cnn_steering_safe', 'blend', 'blend_full')

# ---------------------------------------------------------------------------
# CARLA egg + PythonAPI ``carla/`` (for ``agents.navigation``)
# ---------------------------------------------------------------------------
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla  # noqa: E402

from agents.navigation.behavior_agent import BehaviorAgent  # noqa: E402

from bc_dataset import preprocess_numpy_rgb  # noqa: E402
from bc_model import build_model  # noqa: E402

# PilotNet training camera (must match ``bc_data_collector`` / ``automatic_controll``).
CAM_W = 320
CAM_H = 180
CAM_FOV = 90
CAM_TRANSFORM = carla.Transform(carla.Location(x=1.5, y=0.0, z=1.7))


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model = build_model().to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def carla_image_to_rgb_array(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    return arr[:, :, :3][:, :, ::-1].copy()


def get_speed_kmh(vehicle):
    v = vehicle.get_velocity()
    return 3.6 * math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def spawn_vehicle(world, filter_str):
    bp_lib = world.get_blueprint_library()
    candidates = bp_lib.filter(filter_str)
    if not candidates:
        raise RuntimeError('No blueprints match filter %r' % filter_str)
    bp = random.choice(candidates)
    bp.set_attribute('role_name', 'bc_blend')
    if bp.has_attribute('color'):
        bp.set_attribute(
            'color', random.choice(bp.get_attribute('color').recommended_values))
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    for sp in spawn_points:
        v = world.try_spawn_actor(bp, sp)
        if v is not None:
            return v
    raise RuntimeError('Could not spawn vehicle.')


def setup_model_camera(world, vehicle):
    bp = world.get_blueprint_library().find('sensor.camera.rgb')
    bp.set_attribute('image_size_x', str(CAM_W))
    bp.set_attribute('image_size_y', str(CAM_H))
    bp.set_attribute('fov', str(CAM_FOV))
    sensor = world.spawn_actor(bp, CAM_TRANSFORM, attach_to=vehicle)
    q = queue.Queue()
    sensor.listen(q.put)
    return sensor, q


def setup_display_camera(world, vehicle, width, height, gamma=2.2):
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
    q = queue.Queue()
    sensor.listen(q.put)
    return sensor, q


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


def _vehicle_control_like_agent(agent_ctl, steer, throttle, brake):
    out = carla.VehicleControl(
        steer=max(-1.0, min(1.0, steer)),
        throttle=max(0.0, min(1.0, throttle)),
        brake=max(0.0, min(1.0, brake)),
        hand_brake=bool(agent_ctl.hand_brake),
        reverse=bool(agent_ctl.reverse),
        manual_gear_shift=False,
    )
    if hasattr(agent_ctl, 'gear'):
        try:
            out.gear = agent_ctl.gear
        except Exception:
            pass
    return out


def blend_vehicle_control_full(agent_ctl, steer_m, thr_m, brk_m, w_model):
    """Convex mix on steer, throttle, brake (``w_model`` on CNN — dilutes stops)."""
    u = 1.0 - w_model
    steer = w_model * steer_m + u * agent_ctl.steer
    throttle = w_model * thr_m + u * agent_ctl.throttle
    brake = w_model * brk_m + u * agent_ctl.brake
    return _vehicle_control_like_agent(agent_ctl, steer, throttle, brake)


def blend_vehicle_control_steer_only(agent_ctl, steer_m, w_model):
    """Interpolate CNN vs agent steering; pedals 100% BehaviorAgent (traffic-safe)."""
    u = 1.0 - w_model
    steer = w_model * steer_m + u * float(agent_ctl.steer)
    throttle = float(agent_ctl.throttle)
    brake = float(agent_ctl.brake)
    return _vehicle_control_like_agent(agent_ctl, steer, throttle, brake)


def fuse_vehicle_control(agent_ctl, steer_m, thr_m, brk_m, fusion, blend_w):
    """Compose final ``VehicleControl`` from BehaviorAgent + CNN.

    fusion:
      ``blend`` -- mix **steer** only; throttle/brake always agent.
      ``blend_full`` -- mix all three axes (CNN can break traffic compliance).
      ``cnn_steering`` / ``cnn_steering_safe`` -- see module docstring.
    """
    if fusion == 'blend':
        return blend_vehicle_control_steer_only(agent_ctl, steer_m, blend_w)
    if fusion == 'blend_full':
        return blend_vehicle_control_full(agent_ctl, steer_m, thr_m, brk_m, blend_w)

    throttle = float(agent_ctl.throttle)
    steer = steer_m

    if fusion == 'cnn_steering':
        brake = float(agent_ctl.brake)
    elif fusion == 'cnn_steering_safe':
        brake = max(float(agent_ctl.brake), float(brk_m))
    else:
        raise ValueError('Unknown fusion: %s' % fusion)

    return _vehicle_control_like_agent(agent_ctl, steer, throttle, brake)


def render_hud(surface, font, info_lines, right_label, right_color):
    overlay = pygame.Surface((surface.get_width(), 22 * (len(info_lines) + 1) + 10))
    overlay.set_alpha(160)
    overlay.fill((0, 0, 0))
    surface.blit(overlay, (0, 0))
    y = 6
    for line in info_lines:
        surface.blit(font.render(line, True, (255, 255, 255)), (10, y))
        y += 22
    surface.blit(font.render(right_label, True, right_color),
                 (surface.get_width() - 280, 16))


class ManualOverride(object):
    def __init__(self):
        self.control = carla.VehicleControl()
        self._steer_cache = 0.0

    def update(self, milliseconds):
        keys = pygame.key.get_pressed()
        self.control.throttle = 0.7 if (keys[K_w] or keys[K_UP]) else 0.0
        self.control.brake = 1.0 if (keys[K_s] or keys[K_DOWN]) else 0.0
        steer_increment = 7e-4 * milliseconds
        if keys[K_a] or keys[K_LEFT]:
            self._steer_cache = (0.0 if self._steer_cache > 0
                                  else self._steer_cache - steer_increment)
        elif keys[K_d] or keys[K_RIGHT]:
            self._steer_cache = (0.0 if self._steer_cache < 0
                                  else self._steer_cache + steer_increment)
        else:
            self._steer_cache = 0.0
        self._steer_cache = max(-0.7, min(0.7, self._steer_cache))
        self.control.steer = float(self._steer_cache)
        return self.control


def pick_destination(world_map, spawn_points, preferred_index):
    if not spawn_points:
        raise RuntimeError('No spawn points in map.')
    idx = preferred_index % len(spawn_points)
    return spawn_points[idx].location


def make_behavior_agent(vehicle, behavior):
    return BehaviorAgent(vehicle, behavior=behavior)


def game_loop(args):
    pygame.init()
    pygame.font.init()

    device = torch.device(
        'cuda' if args.device == 'cuda' else (
            'cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu'))

    logging.info('[BC+Agent] Loading model %s on %s', args.model, device)
    model = load_model(args.model, device)

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    sim_world = client.get_world()
    traffic_manager = client.get_trafficmanager()

    original_settings = sim_world.get_settings()
    settings = sim_world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = args.delta
    sim_world.apply_settings(settings)
    traffic_manager.set_synchronous_mode(True)

    ego = None
    model_sensor = None
    display_sensor = None
    agent = None

    try:
        carla_map = sim_world.get_map()
        spawn_points = carla_map.get_spawn_points()
        ego = spawn_vehicle(sim_world, args.filter)
        model_sensor, model_queue = setup_model_camera(sim_world, ego)
        display_sensor, display_queue = setup_display_camera(
            sim_world, ego, args.width, args.height)

        destination = pick_destination(carla_map, spawn_points, args.destination_index)
        agent = make_behavior_agent(ego, args.behavior)
        agent.set_destination(destination)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption('CARLA - BC + BehaviorAgent blend')
        font = pygame.font.SysFont('couriernew', 16, bold=True)

        manual = ManualOverride()
        manual_mode = False
        blend_w = args.blend_weight
        fusion_mode = args.fusion

        sim_world.tick()
        clock = pygame.time.Clock()
        target_fps = max(1, int(round(1.0 / args.delta)))

        last_model = {'steer': 0.0, 'throttle': 0.0, 'brake': 0.0}
        last_agent = {'steer': 0.0, 'throttle': 0.0, 'brake': 0.0}
        infer_ms_ema = 0.0

        logging.info(
            '[BC+Agent] dest=%s  fusion=%s  blend_w=%.2f  keys: F fusion  [ / ] blend  M manual',
            args.destination_index, fusion_mode, blend_w)

        while True:
            sim_world.tick()
            ms = clock.tick_busy_loop(target_fps)

            quit_flag = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_flag = True
                elif event.type == pygame.KEYUP:
                    if event.key == K_ESCAPE or (
                            event.key == K_q and pygame.key.get_mods() & KMOD_CTRL):
                        quit_flag = True
                    elif event.key == K_m:
                        manual_mode = not manual_mode
                        logging.info('[BC+Agent] Manual mode: %s', manual_mode)
                    elif event.key == K_f:
                        i = FUSION_MODES_ORDER.index(fusion_mode)
                        fusion_mode = FUSION_MODES_ORDER[(i + 1) % len(FUSION_MODES_ORDER)]
                        logging.info('[BC+Agent] Fusion mode: %s', fusion_mode)
                    elif event.key == K_LEFTBRACKET:
                        if fusion_mode not in ('blend', 'blend_full'):
                            logging.info('[BC+Agent] [ / ] only for fusion blend / blend_full')
                        else:
                            blend_w = max(0.0, round(blend_w - 0.1, 2))
                            logging.info('[BC+Agent] CNN blend weight: %.2f', blend_w)
                    elif event.key == K_RIGHTBRACKET:
                        if fusion_mode not in ('blend', 'blend_full'):
                            logging.info('[BC+Agent] [ / ] only for fusion blend / blend_full')
                        else:
                            blend_w = min(1.0, round(blend_w + 0.1, 2))
                            logging.info('[BC+Agent] CNN blend weight: %.2f', blend_w)
                    elif event.key == K_r:
                        destroy_actors([model_sensor, display_sensor, ego])
                        ego = spawn_vehicle(sim_world, args.filter)
                        model_sensor, model_queue = setup_model_camera(sim_world, ego)
                        display_sensor, display_queue = setup_display_camera(
                            sim_world, ego, args.width, args.height)
                        destination = pick_destination(
                            carla_map, spawn_points, args.destination_index)
                        agent = make_behavior_agent(ego, args.behavior)
                        agent.set_destination(destination)
                        manual = ManualOverride()
                        manual_mode = False
                        sim_world.tick()
                        continue
                    elif event.key == K_l and args.loop:
                        destination = random.choice(spawn_points).location
                        agent.set_destination(destination)
                        logging.info('[BC+Agent] New random destination')

            if quit_flag:
                break

            try:
                image_model = model_queue.get(timeout=2.0)
                image_display = display_queue.get(timeout=2.0)
            except queue.Empty:
                logging.warning('Camera timeout, skipping tick')
                continue

            rgb = carla_image_to_rgb_array(image_model)

            agent_control = agent.run_step()
            agent_control.manual_gear_shift = False
            last_agent = {
                'steer': float(agent_control.steer),
                'throttle': float(agent_control.throttle),
                'brake': float(agent_control.brake),
            }

            t0 = time.time()
            with torch.no_grad():
                arr = preprocess_numpy_rgb(rgb)
                x = torch.from_numpy(arr).unsqueeze(0).to(device)
                out = model(x)
            infer_ms = (time.time() - t0) * 1000.0
            infer_ms_ema = 0.9 * infer_ms_ema + 0.1 * infer_ms

            steer_m = float(out['steer'].item())
            thr_m = float(out['throttle'].item())
            brk_m = float(out['brake'].item())
            if brk_m < 0.1:
                brk_m = 0.0
            last_model = {'steer': steer_m, 'throttle': thr_m, 'brake': brk_m}

            if manual_mode:
                ctl = manual.update(ms)
                ego.apply_control(ctl)
            else:
                fused = fuse_vehicle_control(
                    agent_control, steer_m, thr_m, brk_m, fusion_mode, blend_w)
                speed = get_speed_kmh(ego)
                if speed > args.max_speed:
                    fused.throttle = 0.0
                    fused.brake = max(fused.brake, 0.3)
                ego.apply_control(fused)

            if agent.done():
                if args.loop:
                    agent.set_destination(random.choice(spawn_points).location)
                    logging.info('[BC+Agent] Goal reached; new random destination')
                else:
                    logging.info('[BC+Agent] Goal reached. Quit (--loop keeps driving).')
                    break

            rgb_view = carla_image_to_rgb_array(image_display)
            small = pygame.surfarray.make_surface(rgb_view.swapaxes(0, 1))
            if rgb_view.shape[1] != args.width or rgb_view.shape[0] != args.height:
                frame = pygame.transform.smoothscale(small, (args.width, args.height))
            else:
                frame = small
            display.blit(frame, (0, 0))

            speed = get_speed_kmh(ego)
            mode_label = 'MANUAL' if manual_mode else fusion_mode.upper().replace('_', ' ')
            mode_color = (220, 180, 40) if manual_mode else (80, 200, 255)
            if fusion_mode in ('blend', 'blend_full'):
                sfx = 'steer' if fusion_mode == 'blend' else 'full'
                fuse_line = 'Fusion:   {}  w={:.2f} [ / ]'.format(sfx, blend_w)
            else:
                fuse_line = 'Fusion:   {}   ( F )'.format(fusion_mode)
            info = [
                'Mode:      {}'.format(mode_label),
                fuse_line,
                'Speed:     {:>6.1f} km/h  (cap {})'.format(speed, args.max_speed),
                'CNN  s/t/b {:+6.3f} / {:4.2f} / {:4.2f}'.format(
                    last_model['steer'], last_model['throttle'], last_model['brake']),
                'Agnt s/t/b {:+6.3f} / {:4.2f} / {:4.2f}'.format(
                    last_agent['steer'], last_agent['throttle'], last_agent['brake']),
                'Infer:     {:>6.1f} ms'.format(infer_ms_ema),
            ]
            render_hud(display, font, info, mode_label, mode_color)
            pygame.display.flip()

    finally:
        destroy_actors([model_sensor, display_sensor, ego])
        try:
            sim_world.apply_settings(original_settings)
            traffic_manager.set_synchronous_mode(False)
        except Exception:
            pass
        pygame.quit()


def main():
    p = argparse.ArgumentParser(
        description='Blend BehaviorAgent (automatic_control stack) with BC PilotNet')
    p.add_argument('--model', required=True, help='checkpoint from bc_train.py')
    p.add_argument(
        '--fusion', type=str, default='cnn_steering',
        choices=('blend', 'blend_full', 'cnn_steering', 'cnn_steering_safe'),
        help='blend = mix CNN/agent STEER only + agent throttle/brake (traffic-safe); '
             'blend_full = mix all axes (breaks lights/stops); '
             'cnn_steering / cnn_steering_safe = see docs.')
    p.add_argument('--blend-weight', type=float, default=0.5,
                   help='CNN weight [0,1] for --fusion blend and blend_full.')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=2000)
    p.add_argument('--res', default='1280x720')
    p.add_argument('--filter', default='vehicle.tesla.model3')
    p.add_argument('--delta', type=float, default=0.05)
    p.add_argument('--max-speed', type=float, default=40.0,
                   help='post-blend speed cap (km/h)')
    p.add_argument('--destination-index', type=int, default=60,
                   help='spawn index for goal location (mod map spawns)')
    p.add_argument('--behavior', choices=['cautious', 'normal', 'aggressive'],
                   default='normal')
    p.add_argument('--loop', action='store_true',
                   help='after reaching goal, pick another random destination')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])

    args = p.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    args.blend_weight = max(0.0, min(1.0, args.blend_weight))

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    print(__doc__)

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user.')


if __name__ == '__main__':
    main()
