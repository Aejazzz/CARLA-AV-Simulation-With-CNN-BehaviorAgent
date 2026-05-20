#!/usr/bin/env python
"""Batch Town10 vs Town01 closed-loop eval for automatic_controll.py.

Runs four conditions (map x cnn_mode), writes JSON under eval/, and prints a
summary table. Requires CARLA on 127.0.0.1:2000 (starts CarlaUE4.exe if missing).

Usage (from PythonAPI/examples, Python 3.7):

    py -3.7 run_eval_town_comparison.py
    py -3.7 run_eval_town_comparison.py --trials 3 --eval-duration-s 300
"""

from __future__ import print_function

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

EXAMPLES = os.path.dirname(os.path.abspath(__file__))
UTIL = os.path.normpath(os.path.join(EXAMPLES, '..', 'util'))
CARLA_EXE = os.path.normpath(
    os.path.join(EXAMPLES, '..', '..', 'CarlaUE4.exe'))

MAPS = [
    ('Town10HD_Opt', 'town10'),
    ('Town01_Opt', 'town01'),
]
CNN_MODES = [
    ('when_needed', 'cnn'),
    ('never', 'baseline'),
]


def wait_for_carla(host, port, timeout_s=180):
    import glob

    sys.path.insert(0, EXAMPLES)
    egg = glob.glob(os.path.join(EXAMPLES, '../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    if egg not in sys.path:
        sys.path.append(egg)
    import carla

    client = carla.Client(host, port)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            client.set_timeout(5.0)
            client.get_world()
            return True
        except RuntimeError:
            time.sleep(2.0)
    return False


def start_carla_if_needed():
    if wait_for_carla('127.0.0.1', 2000, timeout_s=3):
        print('CARLA already listening on :2000')
        return
    if not os.path.isfile(CARLA_EXE):
        raise RuntimeError('CarlaUE4.exe not found: %s' % CARLA_EXE)
    print('Starting CARLA:', CARLA_EXE)
    subprocess.Popen(
        [CARLA_EXE],
        cwd=os.path.dirname(CARLA_EXE),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not wait_for_carla('127.0.0.1', 2000, timeout_s=300):
        raise RuntimeError('CARLA did not become ready within 300 s')


def load_map(map_name, host, port):
    cmd = [
        sys.executable, os.path.join(UTIL, 'config.py'),
        '--host', host, '-p', str(port), '-m', map_name,
        '--no-rendering',
    ]
    print('Loading map:', map_name)
    subprocess.check_call(cmd, cwd=UTIL)


def run_session(args, map_slug, cnn_mode, trial, seed):
    out_dir = os.path.join(EXAMPLES, 'eval')
    os.makedirs(out_dir, exist_ok=True)
    report = os.path.join(
        out_dir,
        '%s_%s_trial%02d_seed%d.json' % (map_slug, cnn_mode, trial, seed))

    cmd = [
        sys.executable, os.path.join(EXAMPLES, 'automatic_controll.py'),
        '--model', args.model,
        '--host', args.host,
        '-p', str(args.port),
        '--cnn-mode', cnn_mode,
        '--seed', str(seed),
        '--destination-index', str(args.destination_index),
        '--no-dashboard',
        '--eval-report', report,
    ]
    if args.eval_duration_s is not None and args.eval_duration_s > 0:
        cmd.extend(['--eval-duration-s', str(args.eval_duration_s)])

    print('\n--- Session:', ' '.join(cmd))
    wall_cap = args.wall_timeout_s
    proc = subprocess.run(
        cmd, cwd=EXAMPLES, timeout=wall_cap if wall_cap > 0 else None)
    if proc.returncode != 0:
        print('WARNING: session exited with code', proc.returncode)
    if not os.path.isfile(report):
        raise RuntimeError('Eval report not written: %s' % report)
    with open(report, 'r') as f:
        return json.load(f)


def session_success(payload):
    sim = payload.get('simulation', {})
    goals = int(sim.get('goals_reached', 0))
    collisions = int(sim.get('collision_events', 0))
    return goals >= 1 and collisions == 0


def aggregate(rows):
    """rows: list of (map, cnn_mode, payload)."""
    from collections import defaultdict

    groups = defaultdict(list)
    for map_slug, cnn_mode, p in rows:
        key = (map_slug, cnn_mode)
        groups[key].append(p)

    table = []
    for (map_slug, cnn_mode), items in sorted(groups.items()):
        n = len(items)
        succ = sum(1 for p in items if session_success(p))
        coll_free = sum(
            1 for p in items
            if int(p.get('simulation', {}).get('collision_events', 0)) == 0)
        dists = [
            float(p['simulation']['distance_driven_m'])
            for p in items if p.get('simulation')]
        ttcs = []
        for p in items:
            ttc = p.get('simulation', {}).get('time_to_first_collision_s')
            if ttc is not None:
                ttcs.append(float(ttc))
        table.append({
            'map_slug': map_slug,
            'cnn_mode': cnn_mode,
            'sessions': n,
            'goal_success_rate': round(succ / n, 4) if n else None,
            'collision_free_rate': round(coll_free / n, 4) if n else None,
            'mean_distance_m': round(sum(dists) / len(dists), 2) if dists else None,
            'mean_ttc_first_collision_s': (
                round(sum(ttcs) / len(ttcs), 2) if ttcs else None),
            'reports': [
                p.get('_report_path') for p in items
            ],
        })
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('-p', '--port', default=2000, type=int)
    p.add_argument(
        '--model', default=os.path.join('models', 'bc.pt'),
        help='checkpoint relative to examples/')
    p.add_argument('--trials', type=int, default=3,
                   help='sessions per (map, cnn_mode) cell')
    p.add_argument('--seed-base', type=int, default=42)
    p.add_argument('--destination-index', type=int, default=80)
    p.add_argument(
        '--eval-duration-s', type=float, default=300.0,
        help='max sim seconds per session (0 = until goal only)')
    p.add_argument(
        '--wall-timeout-s', type=float, default=600.0,
        help='wall-clock subprocess cap per session')
    p.add_argument('--skip-carla-start', action='store_true')
    args = p.parse_args()

    if not args.skip_carla_start:
        start_carla_if_needed()

    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = os.path.join(EXAMPLES, model_path)
    if not os.path.isfile(model_path):
        raise RuntimeError('Model not found: %s' % model_path)
    args.model = model_path

    all_rows = []
    for map_name, map_slug in MAPS:
        load_map(map_name, args.host, args.port)
        time.sleep(3.0)
        for cnn_mode, mode_slug in CNN_MODES:
            for trial in range(args.trials):
                seed = args.seed_base + trial
                payload = run_session(
                    args, map_slug, cnn_mode, trial + 1, seed)
                report_path = os.path.join(
                    EXAMPLES, 'eval',
                    '%s_%s_trial%02d_seed%d.json' % (
                        map_slug, cnn_mode, trial + 1, seed))
                payload['_report_path'] = report_path
                all_rows.append((map_slug, cnn_mode, payload))
                sim = payload.get('simulation', {})
                print(
                    '  outcome=%s goals=%s collisions=%s distance=%.1f m' % (
                        payload.get('outcome'),
                        sim.get('goals_reached'),
                        sim.get('collision_events'),
                        sim.get('distance_driven_m', 0.0)))

    summary = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'success_definition': 'goals_reached >= 1 and collision_events == 0',
        'eval_duration_s_cap': args.eval_duration_s,
        'trials_per_cell': args.trials,
        'table': aggregate(all_rows),
        'sessions': [
            {
                'map_slug': m,
                'cnn_mode': c,
                'report': p.get('_report_path'),
                'outcome': p.get('outcome'),
                'map_name': p.get('map_name'),
                'simulation': p.get('simulation'),
            }
            for m, c, p in all_rows
        ],
    }

    out_summary = os.path.join(EXAMPLES, 'eval', 'town_comparison_summary.json')
    with open(out_summary, 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print('\nWrote', out_summary)
    print('\n| Map | CNN mode | N | Goal success | Collision-free | Mean dist (m) | Mean TTC (s) |')
    print('|-----|----------|---|--------------|----------------|---------------|--------------|')
    for row in summary['table']:
        print(
            '| %s | %s | %d | %.1f%% | %.1f%% | %s | %s |' % (
                row['map_slug'],
                row['cnn_mode'],
                row['sessions'],
                100.0 * (row['goal_success_rate'] or 0),
                100.0 * (row['collision_free_rate'] or 0),
                row['mean_distance_m'],
                row['mean_ttc_first_collision_s'],
            ))


if __name__ == '__main__':
    main()
