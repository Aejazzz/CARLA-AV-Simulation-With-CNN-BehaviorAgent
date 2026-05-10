#!/usr/bin/env python
"""CARLA behavioral cloning — Autopilot built on ``automatic_control.py``.

This client reuses the same **World**, **HUD**, **CameraManager**, collision/lane
sensors, and **BehaviorAgent** loop as ``automatic_control.py``. A narrow
front RGB camera (matching ``bc_data_collector`` / PilotNet training) is added.

**PilotNet is used only when the driving situation looks "easy"** (default), so
the stack normally behaves like ``automatic_control``: full BehaviorAgent for
steering, traffic lights, and obstacles. CNN throttle/brake are fused only in
that regime, with the same safety filter as before (agent brake always wins,
throttle capped by the agent).

``--cnn-mode``:
  * ``when_needed`` (default) — CNN off during hard steering or heavy agent braking
  * ``never`` — pure BehaviorAgent (same as automatic control, model still loaded)
  * ``always`` — always run CNN longitudinal (previous autopilot-style blend)

Run from ``PythonAPI/examples`` with CARLA running::

    python automatic_controll.py --model models/bc.pt

Optional session metrics (distance, collisions, goals, time-to-first-collision)::

    python automatic_controll.py --model models/bc.pt --eval-report eval/session01.json

Other flags match ``automatic_control.py`` where applicable (``--res``, ``--sync``,
``--filter``, ``--loop``, ``--behavior``, etc.).

The **matplotlib + psutil dashboard** mirrors ``automatic_control`` and starts **by default** (``--no-dashboard`` to disable). The pygame view uses the same **HUD**, **CameraManager**, and sensor readouts as ``automatic_control``, plus appended **PilotNet** lines.

PilotNet outputs continuous controls — there is **no softmax confidence**. Show **blend agreement**: a 0–100% heuristic comparing CNN vs BehaviorAgent pedals when fused (not calibrated crash risk).
"""

from __future__ import print_function

import argparse
import glob
import json
import logging
import math
import os
import queue
import sys
import threading
import time

import numpy as np
import torch

import pygame

# Same-directory CARLA API + agents (as automatic_control.py)
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
except Exception:
    pass

import carla  # noqa: E402

# Reuse the full automatic_control client (World, HUD, agents, dashboard, …)
import automatic_control as ac  # noqa: E402

from bc_dataset import preprocess_numpy_rgb
from bc_model import build_model

# PilotNet camera — must match bc_data_collector / bc_train geometry.
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


def setup_model_camera(world, vehicle):
    bp = world.get_blueprint_library().find('sensor.camera.rgb')
    bp.set_attribute('image_size_x', str(CAM_W))
    bp.set_attribute('image_size_y', str(CAM_H))
    bp.set_attribute('fov', str(CAM_FOV))
    sensor = world.spawn_actor(bp, CAM_TRANSFORM, attach_to=vehicle)
    q = queue.Queue()
    sensor.listen(q.put)
    return sensor, q


def _vehicle_control_from_base(agent_ctl, steer, throttle, brake):
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


def fuse_behavior_agent_cnn_longitudinal(agent_ctl, thr_m, brk_m):
    steer = float(agent_ctl.steer)
    thr_m = max(0.0, min(1.0, float(thr_m)))
    brk_m = float(brk_m)
    if brk_m < 0.1:
        brk_m = 0.0
    brk_m = max(0.0, min(1.0, brk_m))
    ag_t = float(agent_ctl.throttle)
    ag_b = float(agent_ctl.brake)
    throttle = min(thr_m, ag_t)
    brake = max(brk_m, ag_b)
    return _vehicle_control_from_base(agent_ctl, steer, throttle, brake)


def should_use_cnn(args, agent_ctl, speed_kmh):
    """CNN longitudinal only in 'calm' regimes; otherwise full BehaviorAgent."""
    if args.cnn_mode == 'never':
        return False
    if args.cnn_mode == 'always':
        return True
    ag_br = float(agent_ctl.brake)
    ag_st = abs(float(agent_ctl.steer))
    if ag_br > args.cnn_max_agent_brake:
        return False
    if ag_st > args.cnn_max_agent_steer:
        return False
    if speed_kmh < args.cnn_min_speed_kmh:
        return False
    return True


def destroy_sensor(sensor):
    if sensor is None:
        return
    try:
        sensor.stop()
    except Exception:
        pass
    try:
        sensor.destroy()
    except Exception:
        pass


class AutopilotHUD(ac.HUD):
    """Same left-panel HUD as ``automatic_control``, plus PilotNet overlay lines."""

    def __init__(self, width, height):
        super(AutopilotHUD, self).__init__(width, height)
        self._bc_overlay_lines = []

    def set_bc_overlay(self, lines):
        self._bc_overlay_lines = list(lines) if lines else []

    def tick(self, world, clock):
        super(AutopilotHUD, self).tick(world, clock)
        if self._show_info and self._bc_overlay_lines:
            self._info_text += ['', 'PilotNet assist'] + self._bc_overlay_lines


def _blend_agreement_heuristic(agent_ctl, thr_cnn, brk_cnn):
    """Rough 0–1 score when CNN pedals align with BehaviorAgent pedals."""
    ag_t = float(agent_ctl.throttle)
    ag_b = float(agent_ctl.brake)
    d = abs(float(thr_cnn) - ag_t) + abs(float(brk_cnn) - ag_b)
    return max(0.0, min(1.0, 1.0 - 0.5 * d))


def _write_eval_report(path, payload):
    try:
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        logging.info('Wrote evaluation report: %s', path)
    except Exception as exc:
        logging.warning('Could not write eval report: %s', exc)


def _build_eval_payload(args, destination_index, world, metrics):
    """metrics: outcome, sim_start, sim_end, distance_m, goals, ttc_s, n_collisions."""
    map_name = None
    try:
        map_name = world.map.name
    except Exception:
        pass
    dur = None
    if metrics['sim_start'] is not None and metrics['sim_end'] is not None:
        dur = float(metrics['sim_end'] - metrics['sim_start'])

    cap = args.max_speed if getattr(args, 'max_speed', None) is not None and args.max_speed > 0 else None
    out = {
        'model': args.model,
        'cnn_mode': args.cnn_mode,
        'max_speed_cap_kmh': cap,
        'behavior': args.behavior,
        'destination_index': destination_index,
        'map_name': map_name,
        'outcome': metrics['outcome'],
        'simulation': {
            'time_start_s': metrics['sim_start'],
            'time_end_s': metrics['sim_end'],
            'duration_s': dur,
            'distance_driven_m': round(metrics['distance_m'], 3),
            'goals_reached': int(metrics['goals']),
            'collision_events': int(metrics['n_collisions']),
            'time_to_first_collision_s': metrics['ttc_s'],
        },
    }
    return out


def game_loop(args):
    pygame.init()
    pygame.font.init()

    world = None
    model_sensor = None
    perf = None
    traffic_manager = None

    # Optional closed-loop metrics (written on exit if --eval-report set)
    eval_start_sim = None
    eval_prev_sim = None
    eval_distance_m = 0.0
    eval_goals = 0
    eval_prev_collision_len = 0
    eval_first_collision_sim_time = None
    eval_session_outcome = 'active'
    sim_end_for_eval = None
    report_destination_index = args.destination_index

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    logging.info('Loading PilotNet from %s on %s', args.model, device)
    model = load_model(args.model, device)

    try:
        if args.seed is not None:
            ac.random.seed(args.seed)

        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)
        traffic_manager = client.get_trafficmanager()
        sim_world = client.get_world()

        if args.sync:
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = args.delta
            sim_world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        ac.START_INDEX = args.start_index

        hud = AutopilotHUD(args.width, args.height)
        world = ac.World(client.get_world(), hud, args)

        model_sensor, model_queue = setup_model_camera(world.world, world.player)

        agent = ac.BehaviorAgent(world.player, behavior=args.behavior)

        spawn_points = world.map.get_spawn_points()
        if not spawn_points:
            raise RuntimeError('No spawn points available for destination.')

        report_destination_index = args.destination_index % len(spawn_points)
        destination_index = report_destination_index
        destination = spawn_points[destination_index].location
        agent.set_destination(destination)

        perf = None
        if args.dashboard:
            try:
                perf = ac.PerformanceMonitorAdvanced(maxlen=ac.DATABUFFER)
                threading.Thread(
                    target=ac.dashboard_thread_fn, args=(perf,), daemon=True).start()
            except Exception as exc:
                logging.warning('Performance dashboard disabled: %s', exc)
                perf = None

        controller = ac.KeyboardControl(world)
        clock = pygame.time.Clock()
        infer_ms_ema = 0.0
        last_cnn_steer = 0.0
        last_cnn_thr = 0.0
        last_cnn_brk = 0.0

        logging.info(
            'BehaviorAgent + PilotNet (cnn-mode=%s). Dest index %d.',
            args.cnn_mode, destination_index)

        while True:
            clock.tick()
            if args.sync:
                world.world.tick()
            else:
                world.world.wait_for_tick()

            if controller.parse_events():
                eval_session_outcome = 'user_quit'
                try:
                    sim_end_for_eval = world.hud.simulation_time
                except Exception:
                    sim_end_for_eval = None
                return

            # Match automatic_control.py game_loop ordering exactly:
            #   world.world.tick → parse_events → world.tick(HUD) → perf → render → flip
            #   → if agent.done(): ... break
            #   → run_step → apply_control
            #
            # Control is applied once per frame, last, same as BehaviorAgent baseline.

            world.tick(clock)

            sim_t = world.hud.simulation_time
            if eval_start_sim is None:
                eval_start_sim = sim_t
            if eval_prev_sim is not None:
                dt_sim = max(0.0, sim_t - eval_prev_sim)
                spd_ms = get_speed_kmh(world.player) / 3.6
                eval_distance_m += spd_ms * dt_sim
            eval_prev_sim = sim_t

            coll = world.collision_sensor
            ch = coll.history if coll is not None else []
            n_coll = len(ch)
            if n_coll > eval_prev_collision_len and eval_first_collision_sim_time is None:
                eval_first_collision_sim_time = sim_t
            eval_prev_collision_len = n_coll

            if perf is not None:
                perf.update(sim_t, world.player, world, agent, destination)

            world.render(display)
            pygame.display.flip()

            if agent.done():
                eval_goals += 1
                if args.loop:
                    agent.set_destination(ac.random.choice(spawn_points).location)
                    hud.notification('Target reached', seconds=4.0)
                    print(
                        'The target has been reached, searching for another target')
                else:
                    eval_session_outcome = 'goal_reached'
                    try:
                        sim_end_for_eval = world.hud.simulation_time
                    except Exception:
                        sim_end_for_eval = None
                    print(
                        'The target has been reached, stopping the simulation')
                    break

            agent_control = agent.run_step()
            agent_control.manual_gear_shift = False

            speed = get_speed_kmh(world.player)
            use_cnn = should_use_cnn(args, agent_control, speed)
            ran_infer = False
            thr_disp = last_cnn_thr
            brk_disp = last_cnn_brk
            steer_disp = last_cnn_steer

            if use_cnn:
                rgb_img = None
                while True:
                    try:
                        rgb_img = model_queue.get_nowait()
                    except queue.Empty:
                        break
                if rgb_img is not None:
                    rgb = carla_image_to_rgb_array(rgb_img)
                    t0 = time.time()
                    with torch.no_grad():
                        arr = preprocess_numpy_rgb(rgb)
                        x = torch.from_numpy(arr).unsqueeze(0).to(device)
                        out = model(x)
                    infer_ms = (time.time() - t0) * 1000.0
                    infer_ms_ema = 0.9 * infer_ms_ema + 0.1 * infer_ms
                    thr_disp = float(out['throttle'].item())
                    brk_disp = float(out['brake'].item())
                    steer_disp = float(out['steer'].item())
                    last_cnn_steer = steer_disp
                    last_cnn_thr = thr_disp
                    last_cnn_brk = brk_disp
                    ran_infer = True
                    brk_infer = brk_disp
                    if brk_infer < 0.1:
                        brk_infer = 0.0
                    control = fuse_behavior_agent_cnn_longitudinal(
                        agent_control, thr_disp, brk_infer)
                else:
                    control = agent_control
            else:
                control = agent_control

            if args.max_speed > 0 and speed > args.max_speed:
                control.throttle = 0.0
                control.brake = max(control.brake, 0.3)

            agree_pct = None
            if ran_infer:
                thr_for_agree = thr_disp
                brk_for_agree = brk_disp if brk_disp >= 0.1 else 0.0
                agree_pct = 100.0 * _blend_agreement_heuristic(
                    agent_control, thr_for_agree, brk_for_agree)

            # Shown on the next world.tick(hud) (same pattern as stock client).
            hud.set_bc_overlay([
                'cnn-mode: %s' % args.cnn_mode,
                'CNN gates: %s' % ('active' if use_cnn else 'off'),
                'CNN infer this tick: %s' % ('yes' if ran_infer else 'no'),
                'Infer EMA ms: %.1f' % infer_ms_ema,
                'CNN steer (raw): %+6.4f — not applied' % steer_disp,
                'CNN thr / brake: %.3f / %.3f' % (
                    thr_disp, brk_disp if brk_disp >= 0.1 else 0.0),
                'Agent thr / brake: %.3f / %.3f' % (
                    agent_control.throttle, agent_control.brake),
                'Applied thr / brake: %.3f / %.3f' % (
                    control.throttle, control.brake),
                'Speed cap km/h: %s' % (
                    ('%.0f (script)' % args.max_speed)
                    if args.max_speed > 0
                    else 'off (BehaviorAgent limits)'),
                'Blend agreement: %s' % (
                    '%5.1f %% (heuristic)' % agree_pct
                    if agree_pct is not None
                    else '- (no CNN fused this tick)'),
            ])

            world.player.apply_control(control)

    finally:
        try:
            if perf is not None:
                perf.stop()
        except Exception:
            pass

        if (
            getattr(args, 'eval_report', None)
            and world is not None
            and eval_start_sim is not None
        ):
            if eval_session_outcome == 'active':
                eval_session_outcome = 'interrupted'
            end_t = sim_end_for_eval
            if end_t is None:
                try:
                    end_t = world.hud.simulation_time
                except Exception:
                    end_t = eval_prev_sim
            ttc = None
            if eval_first_collision_sim_time is not None and eval_start_sim is not None:
                ttc = float(eval_first_collision_sim_time - eval_start_sim)
            metrics = {
                'outcome': eval_session_outcome,
                'sim_start': eval_start_sim,
                'sim_end': end_t,
                'distance_m': eval_distance_m,
                'goals': eval_goals,
                'n_collisions': eval_prev_collision_len,
                'ttc_s': ttc,
            }
            _write_eval_report(
                args.eval_report,
                _build_eval_payload(
                    args, report_destination_index, world, metrics))

        destroy_sensor(model_sensor)

        if world is not None:
            try:
                settings = world.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                world.world.apply_settings(settings)
            except Exception:
                pass

        if traffic_manager is not None:
            try:
                traffic_manager.set_synchronous_mode(False)
            except Exception:
                pass

        if world is not None:
            world.destroy()

        pygame.quit()


def main():
    p = argparse.ArgumentParser(
        description='Autopilot: automatic_control.py stack + optional PilotNet '
                    '(CNN longitudinal only when safe / requested).')

    p.add_argument('--model', required=True,
                   help='PilotNet checkpoint (.pt) from bc_train.py')

    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('-p', '--port', default=2000, type=int)

    p.add_argument(
        '--res', metavar='WIDTHxHEIGHT', default='1280x720',
        help='pygame / HUD resolution (default: 1280x720)')

    p.add_argument(
        '--sync', action='store_true', default=True,
        help='Synchronous mode (default: on)')
    p.add_argument(
        '--no-sync', dest='sync', action='store_false',
        help='Asynchronous server stepping')

    p.add_argument(
        '--delta', type=float, default=0.05,
        help='fixed_delta_seconds when sync (default: 0.05)')

    p.add_argument(
        '--filter', metavar='PATTERN', default='vehicle.tesla.model3',
        help='vehicle blueprint filter (default: Tesla Model3 for BC match)')
    p.add_argument(
        '--generation', metavar='G', default='2',
        help='vehicle generation: 1, 2, or All (default: 2)')

    p.add_argument(
        '-l', '--loop', action='store_true', dest='loop', default=False,
        help='pick a new random destination after each goal')
    p.add_argument(
        '--behavior', choices=['cautious', 'normal', 'aggressive'],
        default='normal',
        help='BehaviorAgent preset (same as automatic_control.py -b; default: normal)')

    p.add_argument(
        '--start-index', type=int, default=None,
        help='override automatic_control.START_INDEX for ego spawn')
    p.add_argument(
        '--destination-index', type=int, default=80,
        help='spawn index used for first goal location (mod len(spawns))')

    p.add_argument(
        '--max-speed', type=float, default=-1.0,
        help='Extra post-control speed ceiling (km/h). Default -1 = off — only '
             'BehaviorAgent + map speed limits (like automatic_control). '
             'Set e.g. 30 to force a low cap.')

    p.add_argument(
        '--cnn-mode', choices=['when_needed', 'never', 'always'],
        default='when_needed',
        help='when CNN longitudinal is applied (default: when_needed)')
    p.add_argument(
        '--cnn-max-agent-steer', type=float, default=0.18,
        help='above this |agent steer|, skip CNN (when_needed)')
    p.add_argument(
        '--cnn-max-agent-brake', type=float, default=0.12,
        help='above this agent brake, skip CNN (when_needed)')
    p.add_argument(
        '--cnn-min-speed-kmh', type=float, default=1.5,
        help='below this speed, skip CNN (when_needed)')

    p.add_argument(
        '--no-dashboard', dest='dashboard', action='store_false', default=True,
        help='disable matplotlib + psutil dashboard (same as automatic_control; on by default)')

    p.add_argument('-s', '--seed', default=None, type=int)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument(
        '--eval-report', metavar='PATH',
        help='optional JSON path: session distance, collisions, goals, time-to-first-collision')

    args = p.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    if args.start_index is None:
        args.start_index = ac.START_INDEX

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    print(__doc__)

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user.')


if __name__ == '__main__':
    main()
