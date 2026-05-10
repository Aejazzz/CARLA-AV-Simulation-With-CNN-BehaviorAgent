#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example of automatic vehicle control from client side."""

from __future__ import print_function

import argparse
import collections
import datetime
import glob
import logging
import math
import os
import numpy.random as random
import re
import sys
import weakref
import time  # added for safe spawn retry delays
import numpy as np

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ---------------------------
# ADVANCED PERFORMANCE DASHBOARD
# ---------------------------
import psutil
import threading
import matplotlib.pyplot as plt
from collections import deque
import math

# Configuration: how many data points to keep on the plots
DATABUFFER = 400     # number of recent samples to retain (increase if you want longer history)
PLOT_REFRESH = 0.05  # seconds between plot updates (tweak if needed)

class PerformanceMonitorAdvanced:
    """
    Collects runtime metrics and exposes a dashboard thread that renders
    4 live plots + textual agent/route state. Designed to be non-blocking.
    """
    def __init__(self, maxlen=DATABUFFER):
        self.maxlen = maxlen
        self.lock = threading.Lock()

        # time axis (sim seconds, or tick counts)
        self.time = deque(maxlen=maxlen)

        # vehicle metrics
        self.speed = deque(maxlen=maxlen)      # km/h
        self.steer = deque(maxlen=maxlen)      # -1..1
        self.throttle = deque(maxlen=maxlen)   # 0..1
        self.brake = deque(maxlen=maxlen)      # 0..1

        # collisions
        self.collision = deque(maxlen=maxlen)  # intensity

        # misc system metrics
        self.cpu = deque(maxlen=maxlen)
        self.ram = deque(maxlen=maxlen)

        # agent & route summary state (strings / numbers)
        self.agent_mode = "N/A"
        self.agent_target_speed = 0.0
        self.distance_remaining = 0.0
        self.agent_extra = ""

        # running flag
        self.running = True

        # start lightweight system monitor thread
        self._sys_thread = threading.Thread(target=self._system_monitor_thread, daemon=True)
        self._sys_thread.start()

    def _system_monitor_thread(self):
        """Record CPU/RAM at low frequency (independent of CARLA tick)."""
        while self.running:
            with self.lock:
                self.cpu.append(psutil.cpu_percent(interval=None))
                self.ram.append(psutil.virtual_memory().percent)
            # sleep a little — dashboard draws more frequently
            time.sleep(0.5)

    def update(self, sim_time, player, world, agent, destination_location):
        """
        Called every simulation tick from the main loop.
        - sim_time: world simulation time or tick count (float)
        - player: ego actor
        - world: carla.World
        - agent: BehaviorAgent (or other agent). We'll query best-effort attributes.
        - destination_location: carla.Location of the target
        """
        control = player.get_control()
        vel = player.get_velocity()
        speed_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

        # collision intensity: best-effort from CollisionSensor history
        col_intensity = 0.0
        try:
            history = world.collision_sensor.history
            if history:
                # take the most recent recorded intensity (last entry)
                col_intensity = history[-1][1] if isinstance(history[-1], tuple) else float(history[-1])
        except Exception:
            col_intensity = 0.0

        # distance to destination (euclidean on XY plane)
        try:
            dx = player.get_location().x - destination_location.x
            dy = player.get_location().y - destination_location.y
            dz = player.get_location().z - destination_location.z
            distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        except Exception:
            distance = 0.0

        # glean some agent info but don't assume internals exist
        mode = getattr(agent, 'behavior', getattr(agent, '_behavior', None))
        target_speed = getattr(agent, 'target_speed', getattr(agent, '_target_speed', None))
        # fallback: if args.behavior was used to create the agent, pass it externally by setting agent._behavior_mode if available

        # Compose a small agent-extra string: e.g., if BehaviorAgent has a planner
        extra = ""
        # Best-effort checks for common BehaviorAgent internals (safe getattr)
        try:
            # Some BehaviorAgent variants expose a state or debug string
            state = getattr(agent, 'get_local_planner', None)
            if callable(state):
                # don't call heavy internals — we prefer attributes if present
                pass
            # If BehaviorAgent exposes a debug mode or a decision flag, show it
            if hasattr(agent, 'hero'):
                extra = "hero=True"
        except Exception:
            extra = ""

        # store values thread-safely
        with self.lock:
            self.time.append(sim_time)
            self.speed.append(speed_kmh)
            self.steer.append(control.steer if control is not None else 0.0)
            self.throttle.append(control.throttle if control is not None else 0.0)
            self.brake.append(control.brake if control is not None else 0.0)
            self.collision.append(col_intensity)
            self.distance_remaining = distance
            self.agent_mode = str(mode) if mode is not None else "N/A"
            try:
                self.agent_target_speed = float(target_speed) if target_speed is not None else 0.0
            except Exception:
                self.agent_target_speed = 0.0
            self.agent_extra = str(extra)

    def stop(self):
        self.running = False


def dashboard_thread_fn(perf: PerformanceMonitorAdvanced):
    """
    Matplotlib thread that draws 4 live subplots + textual overlays.
    This runs in a daemon thread so it won't block process exit.
    """
    plt.ion()
    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(3, 3)

    ax_speed = fig.add_subplot(gs[0, 0:2])   # wide speed plot
    ax_steer = fig.add_subplot(gs[0, 2])     # steer vertical
    ax_throttle = fig.add_subplot(gs[1, 0:2]) # throttle & brake
    ax_collision = fig.add_subplot(gs[1, 2]) # collision intensities
    ax_agent = fig.add_subplot(gs[2, :])    # agent and route textual info

    # Styling helper
    for ax in [ax_speed, ax_steer, ax_throttle, ax_collision]:
        ax.grid(True)

    fig.suptitle("CARLA — Real-time Analytics Dashboard (BehaviorAgent)", fontsize=14)

    while perf.running:
        with perf.lock:
            t = list(perf.time)
            speed = list(perf.speed)
            steer = list(perf.steer)
            throttle = list(perf.throttle)
            brake = list(perf.brake)
            collision = list(perf.collision)
            cpu = list(perf.cpu)
            ram = list(perf.ram)
            # agent/route summary
            mode = perf.agent_mode
            target_speed = perf.agent_target_speed
            dist = perf.distance_remaining
            extra = perf.agent_extra

        # Skip plotting until we have some data
        if len(t) < 2:
            time.sleep(PLOT_REFRESH)
            continue

        # Speed plot
        ax_speed.cla()
        ax_speed.plot(t, speed)
        ax_speed.set_ylabel("Speed (km/h)")
        ax_speed.set_title("Speed vs Time")
        ax_speed.grid(True)

        # Steering plot (small)
        ax_steer.cla()
        ax_steer.plot(t, steer)
        ax_steer.set_ylim(-1.1, 1.1)
        ax_steer.set_title("Steer (-1..1)")
        ax_steer.grid(True)

        # Throttle & Brake
        ax_throttle.cla()
        ax_throttle.plot(t, throttle, label="Throttle")
        ax_throttle.plot(t, brake, label="Brake")
        ax_throttle.set_ylim(-0.05, 1.05)
        ax_throttle.set_title("Throttle & Brake")
        ax_throttle.legend(loc='upper right')
        ax_throttle.grid(True)

        # Collision intensity (spike-like)
        ax_collision.cla()
        ax_collision.plot(t, collision)
        ax_collision.set_title("Collision Intensity")
        ax_collision.grid(True)

        # Agent / system text box
        ax_agent.cla()
        ax_agent.axis('off')
        # Compose left and right columns
        left = [
            f"Agent Mode: {mode}",
            f"Agent Target Speed: {target_speed:.2f} km/h",
            f"Distance to Goal: {dist:.2f} m",
            "",
            f"CPU (recent avg): { (sum(cpu)/len(cpu)) if cpu else 0.0 :.1f} %",
            f"RAM (recent avg): { (sum(ram)/len(ram)) if ram else 0.0 :.1f} %",
        ]
        right = [
            f"Recent speed: {speed[-1]:.2f} km/h",
            f"Recent steer: {steer[-1]:.3f}",
            f"Recent throttle: {throttle[-1]:.3f}",
            f"Recent brake: {brake[-1]:.3f}",
            f"Last collision: {collision[-1]:.3f}",
            "",
            f"Extra: {extra}"
        ]
        # Pretty print as two columns
        pad = 4
        for i, line in enumerate(left):
            ax_agent.text(0.01, 0.9 - i*0.12, line, fontsize=10, transform=ax_agent.transAxes, family='monospace')
        for i, line in enumerate(right):
            ax_agent.text(0.55, 0.9 - i*0.12, line, fontsize=10, transform=ax_agent.transAxes, family='monospace')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        try:
            plt.pause(PLOT_REFRESH)
        except Exception:
            # If the window is closed by the user, stop gracefully
            perf.stop()
            break

    # cleanup (close figure)
    try:
        plt.close(fig)
    except Exception:
        pass



# ============================================================================== 
# -- User-configurable fixed start/end indices ---------------------------------
# ==============================================================================

# Change these indices to the spawn point indices you want.
# If the chosen index is blocked the safe-spawn will try nearby spawn points.
START_INDEX = 10
END_INDEX = 80

# ============================================================================== 
# -- Find CARLA module ---------------------------------------------------------
# ============================================================================== 
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ============================================================================== 
# -- Add PythonAPI for release mode --------------------------------------------
# ============================================================================== 
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent  # pylint: disable=import-error

# ============================================================================== 
# -- Global functions ----------------------------------------------------------
# ============================================================================== 

def find_weather_presets():
    """Method to find weather presets"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2, 3]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

# -------------------------------------------------------------------------
# Safe spawn helper (tries preferred spawn and nearby points, non-destructive)
# -------------------------------------------------------------------------
def _distance(loc1, loc2):
    return math.sqrt((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2 + (loc1.z - loc2.z)**2)

def safe_spawn_actor(world, blueprint, preferred_spawn_point,
                     max_retries=12, retry_delay=0.5, max_candidate_distance=150.0,
                     allow_destroy=False, search_radius=3.0):
    """
    Try to spawn an actor at preferred_spawn_point.
    If blocked, try nearest spawn points (sorted by distance).
    Retries up to max_retries (each retry will iterate candidate spawn points).
    If allow_destroy=True then actors within search_radius will be destroyed (USE WITH CAUTION).
    Returns the spawned actor or None if spawn failed.
    """
    # Quick attempt at preferred location first
    actor = world.try_spawn_actor(blueprint, preferred_spawn_point)
    if actor:
        return actor

    # Precompute spawn point list and sort by distance from preferred
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        return None

    spawn_points_sorted = sorted(spawn_points,
                                 key=lambda sp: _distance(sp.location, preferred_spawn_point.location))

    attempts = 0
    while attempts < max_retries:
        for sp in spawn_points_sorted:
            # Skip candidates that are too far
            if _distance(sp.location, preferred_spawn_point.location) > max_candidate_distance:
                break

            # Try to spawn directly
            actor = world.try_spawn_actor(blueprint, sp)
            if actor:
                return actor

            # If blocked and allow_destroy, try to clear small blocking actors
            if allow_destroy:
                for v in world.get_actors().filter('vehicle.*'):
                    try:
                        if _distance(v.get_location(), sp.location) < search_radius:
                            v.destroy()
                    except Exception:
                        pass
                actor = world.try_spawn_actor(blueprint, sp)
                if actor:
                    return actor

        # nothing worked this iteration, wait and retry
        time.sleep(retry_delay)
        attempts += 1

    # final attempt: fallback to random spawn order across all spawn points
    for sp in spawn_points_sorted:
        actor = world.try_spawn_actor(blueprint, sp)
        if actor:
            return actor

    return None

# ============================================================================== 
# -- World --------------------------------------------------------------- 
# ============================================================================== 

class World(object):
    """ Class representing the surrounding environment """

    def __init__(self, carla_world, hud, args):
        """Constructor method"""
        self._args = args
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self.restart(args)
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

    def restart(self, args):
        """Restart the world"""
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager is not None else 0

        # Get a random blueprint.
        blueprint_list = get_actor_blueprints(self.world, self._actor_filter, self._actor_generation)
        if not blueprint_list:
            raise ValueError("Couldn't find any blueprints with the specified filters")
        blueprint = random.choice(blueprint_list)
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # Spawn the player.
        if self.player is not None:
            # If restarting and there was a previous player, we attempt to respawn above previous location
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            # Use safe spawn for respawn case as well (preferred point is above previous spot)
            self.player = safe_spawn_actor(self.world, blueprint, spawn_point,
                                           max_retries=8, retry_delay=0.5,
                                           allow_destroy=False)
        else:
            # Normal first-time spawn: use fixed START_INDEX (user-configurable at top)
            spawn_points = self.map.get_spawn_points()
            if not spawn_points:
                raise RuntimeError("No spawn points available in the map.")

            preferred_index = START_INDEX % len(spawn_points)
            preferred_spawn_point = spawn_points[preferred_index]

            # Try safe spawn at preferred location (non-destructive)
            self.player = safe_spawn_actor(self.world, blueprint, preferred_spawn_point,
                                           max_retries=12, retry_delay=0.5,
                                           max_candidate_distance=150.0,
                                           allow_destroy=False, search_radius=3.0)

        # If safe spawn failed, fallback to trying any spawn point (graceful)
        if self.player is None:
            logging.warning("Safe spawn failed for preferred point; attempting fallback random spawn loop")
            spawn_points = self.map.get_spawn_points()
            for sp in spawn_points:
                self.player = self.world.try_spawn_actor(blueprint, sp)
                if self.player:
                    break

        if self.player is None:
            # Final failure — raise a clear exception.
            raise RuntimeError("Failed to spawn ego vehicle. All spawn attempts returned None.")

        self.modify_vehicle_physics(self.player)
        if self._args.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_id
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        """Get next weather setting"""
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def modify_vehicle_physics(self, actor):
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        """Method for every tick"""
        self.hud.tick(self, clock)

    def render(self, display):
        """Render world"""
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        """Destroy sensors"""
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        """Destroys all actors"""
        actors = [
            self.camera_manager.sensor if self.camera_manager else None,
            self.collision_sensor.sensor if self.collision_sensor else None,
            self.lane_invasion_sensor.sensor if self.lane_invasion_sensor else None,
            self.gnss_sensor.sensor if self.gnss_sensor else None,
            self.player]
        for actor in actors:
            if actor is not None:
                try:
                    actor.destroy()
                except Exception:
                    pass

# ============================================================================== 
# -- KeyboardControl ----------------------------------------------------------- 
# ============================================================================== 

class KeyboardControl(object):
    def __init__(self, world):
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True

    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

# ============================================================================== 
# -- HUD ----------------------------------------------------------------------- 
# ============================================================================== 

class HUD(object):
    """Class for HUD text"""

    def __init__(self, width, height):
        """Constructor method"""
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        """Gets informations from the world at every tick"""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        """HUD method for every tick"""
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        transform = world.player.get_transform()
        vel = world.player.get_velocity()
        control = world.player.get_control()
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')

        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (transform.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % transform.location.z,
            '']
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -1.0, 1.0),
                ('Brake:', control.brake, 0.0, 1.0),
                ('Reverse:', control.reverse),
                ('Hand brake:', control.hand_brake),
                ('Manual:', control.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('Speed:', control.speed, 0.0, 5.556),
                ('Jump:', control.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

        def dist(l):
            return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z)**2)
        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != world.player.id]

        for dist, vehicle in sorted(vehicles):
            if dist > 200.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append('% 4dm %s' % (dist, vehicle_type))

    def toggle_info(self):
        """Toggle info on or off"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """Notification text"""
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """Error text"""
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        """Render for HUD class"""
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        fig = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

# ============================================================================== 
# -- FadingText ----------------------------------------------------------------
# ============================================================================== 

class FadingText(object):
    """ Class for fading text """

    def __init__(self, font, dim, pos):
        """Constructor method"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """Set fading text"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        """Fading text method for every tick"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        """Render fading text method"""
        display.blit(self.surface, self.pos)

# ============================================================================== 
# -- HelpText ------------------------------------------------------------------
# ============================================================================== 

class HelpText(object):
    """ Helper class for text render"""

    def __init__(self, font, width, height):
        """Constructor method"""
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for i, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, i * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """Toggle on or off the render help"""
        self._render = not self._render

    def render(self, display):
        """Render help text method"""
        if self._render:
            display.blit(self.surface, self.pos)

# ============================================================================== 
# -- CollisionSensor -----------------------------------------------------------
# ============================================================================== 

class CollisionSensor(object):
    """ Class for collision sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

# ============================================================================== 
# -- LaneInvasionSensor --------------------------------------------------------
# ============================================================================== 

class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        """On invasion method"""
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ============================================================================== 
# -- GnssSensor -------------------------------------------------------- 
# ============================================================================== 

class GnssSensor(object):
    """ Class for GNSS sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS method"""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

# ============================================================================== 
# -- CameraManager ------------------------------------------------------------- 
# ============================================================================== 

class CameraManager(object):
    """ Class for camera management"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0)), attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z)), attachment.Rigid),
            (carla.Transform(carla.Location(x=+1.9*bound_x, y=+1.0*bound_y, z=1.2*bound_z)), attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=-2.8*bound_x, y=+0.0*bound_y, z=4.6*bound_z), carla.Rotation(pitch=6.0)), attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=-1.0, y=-1.0*bound_y, z=0.4*bound_z)), attachment.Rigid)]

        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            blp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                blp.set_attribute('image_size_x', str(hud.dim[0]))
                blp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                blp.set_attribute('range', '50')
            item.append(blp)
        self.index = None

    def toggle_camera(self):
        """Activate a camera"""
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """Set a sensor"""
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (
            force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])

            # We need to pass the lambda a weak reference to
            # self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=assignment-from-no-return
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

# ============================================================================== 
# -- Game Loop --------------------------------------------------------- 
# ============================================================================== 

def game_loop(args):
    """
    Main loop of the simulation. It handles updating all the HUD information,
    ticking the agent and, if needed, the world.
    """

    pygame.init()
    pygame.font.init()
    world = None

    try:
        if args.seed:
            random.seed(args.seed)

        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)

        traffic_manager = client.get_trafficmanager()
        sim_world = client.get_world()

        if args.sync:
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)

            traffic_manager.set_synchronous_mode(True)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world)
        if args.agent == "Basic":
            agent = BasicAgent(world.player, 30)
            agent.follow_speed_limits(True)
        elif args.agent == "Constant":
            agent = ConstantVelocityAgent(world.player, 30)
            ground_loc = world.world.ground_projection(world.player.get_location(), 5)
            if ground_loc:
                world.player.set_location(ground_loc.location + carla.Location(z=0.01))
            agent.follow_speed_limits(True)
        elif args.agent == "Behavior":
            agent = BehaviorAgent(world.player, behavior=args.behavior)

        # Set the agent destination using fixed END_INDEX (user-configurable at top)
        spawn_points = world.map.get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available in the map for destination.")

        destination_index = END_INDEX % len(spawn_points)
        destination = spawn_points[destination_index].location

        # If you want robust destination selection (avoid unreachable/no-road points),
        # you can try to find a nearby spawn that's reachable. For now, we use the chosen index.
        agent.set_destination(destination)

        # --- start advanced performance dashboard (BehaviorAgent tuned) ---
        perf = PerformanceMonitorAdvanced(maxlen=DATABUFFER)
        threading.Thread(target=dashboard_thread_fn, args=(perf,), daemon=True).start()


        clock = pygame.time.Clock()

        while True:
            clock.tick()
            if args.sync:
                world.world.tick()
            else:
                world.world.wait_for_tick()
            if controller.parse_events():
                return

            world.tick(clock)
            # update performance monitor (use sim time for x axis)
            perf.update(world.hud.simulation_time, world.player, world, agent, destination)


            world.render(display)
            pygame.display.flip()

            if agent.done():
                if args.loop:
                    agent.set_destination(random.choice(spawn_points).location)
                    world.hud.notification("Target reached", seconds=4.0)
                    print("The target has been reached, searching for another target")
                else:
                    print("The target has been reached, stopping the simulation")
                    break

            control = agent.run_step()
            control.manual_gear_shift = False
            world.player.apply_control(control)

    finally:

        try:
            perf.stop()
        except Exception:
            pass

        if world is not None:
            settings = world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

            world.destroy()

        pygame.quit()

# ============================================================================== 
# -- main() -------------------------------------------------------------- 
# ============================================================================== 

def main():
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        "-a", "--agent", type=str,
        choices=["Behavior", "Basic", "Constant"],
        help="select which agent to run",
        default="Behavior")
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal) ',
        default='normal')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

if __name__ == '__main__':
    main()


