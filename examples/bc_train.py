"""Train a PilotNet behavioral-cloning model on data from bc_data_collector.py.

Usage:
    python bc_train.py --data dataset --epochs 30 --batch-size 64 --out models/bc.pt

The dataset path may point either to a single run directory (with a
``labels.csv``) or to a parent folder containing several run subdirectories.

Outputs:
    --out path  : best-validation-loss checkpoint (state_dict)
    --out path with suffix '.last.pt' : last epoch checkpoint
    Also writes ``{stem}_training_history.csv``, ``{stem}_training_summary.json``,
    and ``{stem}_training_curves.png`` beside the checkpoint (unless --no-plots).
"""

from __future__ import print_function

import argparse
import csv
import datetime
import json
import math
import os
import time

import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bc_dataset import CarlaBCDataset, discover_runs, load_all_rows
from bc_model import build_model


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class WeightedBCLoss(nn.Module):
    """MSE on (steer, throttle, brake) with configurable weights.

    Steering matters most (it's the hardest output and biggest safety signal);
    brake is rarely non-zero, so we weight it down to avoid the model just
    predicting brake=0 to win the loss.
    """
    def __init__(self, w_steer=1.0, w_throttle=0.5, w_brake=0.5):
        super(WeightedBCLoss, self).__init__()
        self.w_steer = w_steer
        self.w_throttle = w_throttle
        self.w_brake = w_brake
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        l_steer = self.mse(pred['steer'], target['steer'])
        l_thr = self.mse(pred['throttle'], target['throttle'])
        l_brk = self.mse(pred['brake'], target['brake'])
        total = (self.w_steer * l_steer
                 + self.w_throttle * l_thr
                 + self.w_brake * l_brk)
        return total, {'steer': l_steer.item(),
                       'throttle': l_thr.item(),
                       'brake': l_brk.item()}


# ---------------------------------------------------------------------------
# Train / eval one epoch
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    totals = {'loss': 0.0, 'steer': 0.0, 'throttle': 0.0, 'brake': 0.0, 'n': 0}
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            x = batch['image'].to(device, non_blocking=True)
            target = {
                'steer': batch['steer'].to(device, non_blocking=True),
                'throttle': batch['throttle'].to(device, non_blocking=True),
                'brake': batch['brake'].to(device, non_blocking=True),
            }
            pred = model(x)
            loss, parts = criterion(pred, target)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            bs = x.size(0)
            totals['loss'] += loss.item() * bs
            totals['steer'] += parts['steer'] * bs
            totals['throttle'] += parts['throttle'] * bs
            totals['brake'] += parts['brake'] * bs
            totals['n'] += bs
    n = max(totals['n'], 1)
    return {k: (v / n if k != 'n' else v) for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Logging & plots
# ---------------------------------------------------------------------------

def _write_history_csv(path, rows):
    if not rows:
        return
    fieldnames = [
        'epoch', 'lr',
        'train_loss', 'val_loss',
        'train_steer', 'val_steer',
        'train_throttle', 'val_throttle',
        'train_brake', 'val_brake',
        'epoch_time_s',
    ]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _save_summary_json(path, payload):
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _try_save_plots(history_rows, out_png, no_plots):
    if no_plots or not history_rows:
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:
        print('[BC] Skipping plots (matplotlib unavailable): {}'.format(exc))
        return

    epochs = [r['epoch'] for r in history_rows]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(epochs, [r['train_loss'] for r in history_rows], label='train')
    ax.plot(epochs, [r['val_loss'] for r in history_rows], label='val')
    ax.set_title('Total loss')
    ax.set_xlabel('epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(epochs, [r['lr'] for r in history_rows], color='gray', ls='--', label='lr')
    ax2.set_ylabel('learning rate')

    axes[0, 1].plot(epochs, [r['train_steer'] for r in history_rows], label='train')
    axes[0, 1].plot(epochs, [r['val_steer'] for r in history_rows], label='val')
    axes[0, 1].set_title('Steer MSE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, [r['train_throttle'] for r in history_rows], label='train')
    axes[1, 0].plot(epochs, [r['val_throttle'] for r in history_rows], label='val')
    axes[1, 0].set_title('Throttle MSE')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, [r['train_brake'] for r in history_rows], label='train')
    axes[1, 1].plot(epochs, [r['val_brake'] for r in history_rows], label='val')
    axes[1, 1].set_title('Brake MSE')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle('CARLA behavioral cloning — PilotNet training')
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print('[BC] Saved plot:', out_png)


def save_run_artifacts(history_rows, args, device, n_params,
                       train_rows_n, val_rows_n,
                       len_train_ds, len_val_ds,
                       best_val, best_epoch, runs,
                       no_plots, last_path):
    """Write CSV, JSON summary, and optional PNG next to checkpoint."""
    out_dir = os.path.dirname(os.path.abspath(args.out)) or '.'
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.out))[0]

    base = os.path.join(out_dir, '{}_training'.format(stem))
    csv_path = base + '_history.csv'
    json_path = base + '_summary.json'
    png_path = base + '_curves.png'

    _write_history_csv(csv_path, history_rows)

    summary = {
        'args': {
            'data': args.data,
            'out': args.out,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'val_split': args.val_split,
            'seed': args.seed,
            'device': str(device),
            'no_augment': bool(args.no_augment),
            'no_side_cameras': bool(args.no_side_cameras),
        },
        'dataset': {
            'runs': runs,
            'train_label_rows': train_rows_n,
            'val_label_rows': val_rows_n,
            'train_samples': len_train_ds,
            'val_samples': len_val_ds,
        },
        'model': {
            'parameters': int(n_params),
        },
        'result': {
            'best_val_loss': float(best_val),
            'best_epoch': int(best_epoch) if best_epoch is not None else None,
            'finished_at_utc': datetime.datetime.utcnow().isoformat() + 'Z',
        },
        'artifacts': {
            'checkpoint_best': os.path.abspath(args.out),
            'checkpoint_last': os.path.abspath(last_path),
            'history_csv': os.path.abspath(csv_path),
            'summary_json': os.path.abspath(json_path),
            'curves_png': os.path.abspath(png_path) if (
                not no_plots and history_rows) else None,
        },
    }
    _save_summary_json(json_path, summary)
    _try_save_plots(history_rows, png_path, no_plots)

    print('[BC] Wrote:', csv_path)
    print('[BC] Wrote:', json_path)
    if not no_plots:
        if os.path.isfile(png_path):
            pass
        else:
            print('[BC] (no PNG — matplotlib missing or empty history)')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='Train PilotNet on CARLA BC data')
    p.add_argument('--data', required=True, help='dataset root or single run')
    p.add_argument('--out', default='models/bc.pt', help='checkpoint output path')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--val-split', type=float, default=0.15)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--side-offset', type=float, default=0.20,
                   help='steer correction added to left/right cameras')
    p.add_argument('--drop-zero-steer-prob', type=float, default=0.5,
                   help='probability of dropping near-zero-steer frames during training')
    p.add_argument('--no-augment', action='store_true')
    p.add_argument('--no-side-cameras', action='store_true',
                   help='train only on the center camera (no NVIDIA recovery trick)')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument(
        '--no-plots', action='store_true',
        help='skip writing loss curves PNG (still writes CSV + JSON)')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Dataset: split at the timestamp (row) level BEFORE expansion, so a
    # given physical frame is never both in train and val.
    runs = discover_runs(args.data)
    print('[BC] Found {} run(s):'.format(len(runs)))
    for r in runs:
        print('     ', r)

    all_rows = load_all_rows(runs)
    if len(all_rows) < 10:
        raise RuntimeError('Need at least 10 labelled frames; got %d.' % len(all_rows))
    rng = random.Random(args.seed)
    rng.shuffle(all_rows)
    val_n = max(1, int(len(all_rows) * args.val_split))
    val_rows = all_rows[:val_n]
    train_rows = all_rows[val_n:]

    train_dataset = CarlaBCDataset(
        rows=train_rows,
        use_side_cameras=not args.no_side_cameras,
        side_offset=args.side_offset,
        augment=not args.no_augment,
        drop_zero_steer_prob=args.drop_zero_steer_prob,
    )
    val_dataset = CarlaBCDataset(
        rows=val_rows,
        use_side_cameras=False,   # honest center-camera evaluation
        augment=False,
        drop_zero_steer_prob=0.0,
    )

    print('[BC] train rows {:,} -> {:,} samples (3-cam expansion + augment)'.format(
        len(train_rows), len(train_dataset)))
    print('[BC] val   rows {:,} -> {:,} samples (center-cam, no augment)'.format(
        len(val_rows), len(val_dataset)))

    pin = (device.type == 'cuda')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)

    # Model + opt
    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print('[BC] PilotNet parameters: {:,}'.format(n_params))
    print('[BC] Device:', device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    criterion = WeightedBCLoss()

    out_dir = os.path.dirname(args.out) or '.'
    os.makedirs(out_dir, exist_ok=True)
    last_path = args.out + '.last.pt' if not args.out.endswith('.pt') else \
        args.out.replace('.pt', '.last.pt')

    best_val = math.inf
    best_epoch = None
    history_rows = []
    print('[BC] Training for {} epochs ...'.format(args.epochs))
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        dt = time.time() - t0
        lr_cur = float(optimizer.param_groups[0]['lr'])
        history_rows.append({
            'epoch': epoch,
            'lr': lr_cur,
            'train_loss': tr['loss'],
            'val_loss': vl['loss'],
            'train_steer': tr['steer'],
            'val_steer': vl['steer'],
            'train_throttle': tr['throttle'],
            'val_throttle': vl['throttle'],
            'train_brake': tr['brake'],
            'val_brake': vl['brake'],
            'epoch_time_s': round(dt, 3),
        })
        print(('[BC] Epoch {ep:3d}/{tot}  '
               'train {trl:.4f} (s {trs:.4f} t {trt:.4f} b {trb:.4f})  '
               'val {vll:.4f} (s {vls:.4f} t {vlt:.4f} b {vlb:.4f})  '
               'lr {lr:.2e}  {dt:.1f}s').format(
            ep=epoch, tot=args.epochs,
            trl=tr['loss'], trs=tr['steer'], trt=tr['throttle'], trb=tr['brake'],
            vll=vl['loss'], vls=vl['steer'], vlt=vl['throttle'], vlb=vl['brake'],
            lr=lr_cur, dt=dt))

        # Save last
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'val_loss': vl['loss'],
        }, last_path)

        # Save best-on-val
        if vl['loss'] < best_val:
            best_val = vl['loss']
            best_epoch = epoch
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': vl['loss'],
            }, args.out)
            print('[BC]   * new best val loss {:.4f} -> saved to {}'.format(best_val, args.out))

    print('[BC] Done. Best val loss: {:.4f}'.format(best_val))

    save_run_artifacts(
        history_rows, args, device, n_params,
        len(train_rows), len(val_rows),
        len(train_dataset), len(val_dataset),
        best_val, best_epoch, runs,
        args.no_plots,
        last_path,
    )


if __name__ == '__main__':
    main()
