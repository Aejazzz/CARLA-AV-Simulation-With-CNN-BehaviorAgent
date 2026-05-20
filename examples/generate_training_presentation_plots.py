#!/usr/bin/env python
"""Generate presentation-quality PNG plots from bc_training_history.csv.

Output directory (default): project_docs/presentation/

Usage (from PythonAPI/examples):
    python generate_training_presentation_plots.py
"""

from __future__ import print_function

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

EXAMPLES = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(EXAMPLES, 'models', 'bc_training_history.csv')
DEFAULT_JSON = os.path.join(EXAMPLES, 'models', 'bc_training_summary.json')
DEFAULT_OUT = os.path.join(EXAMPLES, 'project_docs', 'presentation')

# Presentation palette (colorblind-friendly)
C_TRAIN = '#2563eb'
C_VAL = '#ea580c'
C_LR = '#64748b'
C_STEER = '#059669'
C_THROTTLE = '#7c3aed'
C_BRAKE = '#dc2626'
BG = '#ffffff'


def load_rows(csv_path):
    with open(csv_path, newline='') as f:
        return list(csv.DictReader(f))


def load_summary(json_path):
    if not os.path.isfile(json_path):
        return {}
    with open(json_path) as f:
        return json.load(f)


def _style_axes(ax, title, xlabel='Epoch', ylabel=None):
    ax.set_title(title, fontsize=16, fontweight='bold', pad=12)
    ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.6)
    ax.set_axisbelow(True)


def fig_total_loss(rows, summary, out_dir):
    epochs = [int(r['epoch']) for r in rows]
    train = [float(r['train_loss']) for r in rows]
    val = [float(r['val_loss']) for r in rows]
    lr = [float(r['lr']) for r in rows]

    fig = plt.figure(figsize=(12.8, 7.2), facecolor=BG)
    gs = GridSpec(1, 2, width_ratios=[2.2, 1], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    ax.plot(epochs, train, color=C_TRAIN, lw=2.5, marker='o', ms=4,
            label='Train loss')
    ax.plot(epochs, val, color=C_VAL, lw=2.5, marker='o', ms=4,
            label='Val loss')
    best_ep = int(summary.get('result', {}).get('best_epoch', epochs[-1]))
    best_vl = float(summary.get('result', {}).get('best_val_loss', val[-1]))
    ax.scatter([best_ep], [best_vl], s=120, color=C_VAL, zorder=5,
               edgecolors='white', linewidths=2)
    ax.annotate(
        'Best val {:.4f}\n(epoch {})'.format(best_vl, best_ep),
        xy=(best_ep, best_vl), xytext=(best_ep - 8, best_vl + 0.012),
        fontsize=11, arrowprops=dict(arrowstyle='->', color='#334155'))

    _style_axes(ax, 'Total Loss — PilotNet Behavioral Cloning',
                ylabel='MSE (steer + throttle + brake)')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.set_xlim(1, max(epochs))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

    ax_lr = ax.twinx()
    ax_lr.plot(epochs, lr, color=C_LR, ls='--', lw=1.8, alpha=0.85,
               label='Learning rate')
    ax_lr.set_ylabel('Learning rate', fontsize=11, color=C_LR)
    ax_lr.tick_params(axis='y', labelcolor=C_LR, labelsize=10)

    # Summary panel
    ax2.axis('off')
    ds = summary.get('dataset', {})
    res = summary.get('result', {})
    args = summary.get('args', {})
    lines = [
        ('Training summary', 'title'),
        ('', 'gap'),
        ('Dataset', 'h'),
        ('Run: {}'.format(', '.join(ds.get('runs', ['dataset/run1']))), 'b'),
        ('Train samples: {:,}'.format(ds.get('train_samples', '—')), 'b'),
        ('Val samples: {:,}'.format(ds.get('val_samples', '—')), 'b'),
        ('', 'gap'),
        ('Hyperparameters', 'h'),
        ('Epochs: {}'.format(args.get('epochs', 30)), 'b'),
        ('Batch size: {}'.format(args.get('batch_size', 64)), 'b'),
        ('LR: {}'.format(args.get('lr', 1e-4)), 'b'),
        ('Device: {}'.format(args.get('device', 'cpu')), 'b'),
        ('Seed: {}'.format(args.get('seed', 42)), 'b'),
        ('', 'gap'),
        ('Best result', 'h'),
        ('Val loss: {:.4f}'.format(res.get('best_val_loss', best_vl)), 'b'),
        ('Best epoch: {}'.format(res.get('best_epoch', best_ep)), 'b'),
        ('Improvement: {:.1f}%'.format(
            100.0 * (1.0 - val[-1] / val[0]) if val[0] else 0), 'b'),
    ]
    y = 0.96
    for text, kind in lines:
        if kind == 'gap':
            y -= 0.03
            continue
        if kind == 'title':
            ax2.text(0.0, y, text, fontsize=18, fontweight='bold',
                     transform=ax2.transAxes, va='top')
            y -= 0.09
        elif kind == 'h':
            ax2.text(0.0, y, text, fontsize=13, fontweight='bold',
                     transform=ax2.transAxes, va='top', color='#334155')
            y -= 0.07
        else:
            ax2.text(0.04, y, text, fontsize=12,
                     transform=ax2.transAxes, va='top', color='#475569')
            y -= 0.055

    fig.text(0.02, 0.02,
             'Source: bc_training_history.csv · CARLA 0.9.15 · dataset/run1',
             fontsize=9, color='#94a3b8')
    path = os.path.join(out_dir, '01_total_loss_summary.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    return path


def fig_per_axis(rows, out_dir):
    epochs = [int(r['epoch']) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2), facecolor=BG)
    specs = [
        ('Steer MSE', 'train_steer', 'val_steer', C_STEER),
        ('Throttle MSE', 'train_throttle', 'val_throttle', C_THROTTLE),
        ('Brake MSE', 'train_brake', 'val_brake', C_BRAKE),
    ]
    for ax, (title, tk, vk, accent) in zip(axes, specs):
        tr = [float(r[tk]) for r in rows]
        vl = [float(r[vk]) for r in rows]
        ax.plot(epochs, tr, color=C_TRAIN, lw=2, marker='o', ms=3, label='Train')
        ax.plot(epochs, vl, color=C_VAL, lw=2, marker='o', ms=3, label='Val')
        ax.fill_between(epochs, vl, alpha=0.08, color=accent)
        _style_axes(ax, title, ylabel='MSE')
        ax.legend(fontsize=10, loc='upper right')
        ax.set_xlim(1, max(epochs))

    fig.suptitle('Per-Axis Validation Loss (30 Epochs)',
                 fontsize=16, fontweight='bold', y=1.02)
    fig.text(0.02, -0.02,
             'Throttle head remains highest-error on validation split.',
             fontsize=10, color='#64748b')
    path = os.path.join(out_dir, '02_per_axis_loss.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    return path


def fig_val_comparison_bar(rows, out_dir):
    """Bar chart: epoch 1 vs epoch 30 validation components."""
    e1, e30 = rows[0], rows[-1]
    labels = ['Steer', 'Throttle', 'Brake', 'Total']
    ep1_vals = [
        float(e1['val_steer']), float(e1['val_throttle']),
        float(e1['val_brake']), float(e1['val_loss']),
    ]
    ep30_vals = [
        float(e30['val_steer']), float(e30['val_throttle']),
        float(e30['val_brake']), float(e30['val_loss']),
    ]
    x = range(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    b1 = ax.bar([i - w / 2 for i in x], ep1_vals, w, label='Epoch 1',
                color='#94a3b8', edgecolor='white')
    b2 = ax.bar([i + w / 2 for i in x], ep30_vals, w, label='Epoch 30',
                color=C_VAL, edgecolor='white')
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=12)
    _style_axes(ax, 'Validation Loss: Epoch 1 vs Epoch 30',
                xlabel='Output head', ylabel='MSE')
    ax.legend(fontsize=11)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                    '{:.4f}'.format(h), ha='center', va='bottom', fontsize=9)
    path = os.path.join(out_dir, '03_val_epoch1_vs_30.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    return path


def fig_improvement(rows, out_dir):
    """Relative improvement per metric from epoch 1 to 30."""
    e1, e30 = rows[0], rows[-1]
    metrics = [
        ('Total val loss', float(e1['val_loss']), float(e30['val_loss'])),
        ('Steer', float(e1['val_steer']), float(e30['val_steer'])),
        ('Throttle', float(e1['val_throttle']), float(e30['val_throttle'])),
        ('Brake', float(e1['val_brake']), float(e30['val_brake'])),
    ]
    names = [m[0] for m in metrics]
    pct = [100.0 * (1.0 - m[2] / m[1]) if m[1] > 0 else 0 for m in metrics]
    colors = [C_VAL if p > 0 else '#94a3b8' for p in pct]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    y_pos = range(len(names))
    ax.barh(list(y_pos), pct, color=colors, edgecolor='white', height=0.6)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names, fontsize=12)
    ax.set_xlabel('Validation loss reduction (%)', fontsize=12)
    ax.set_title('Training Progress — Val Loss Reduction (Epoch 1 → 30)',
                 fontsize=16, fontweight='bold', pad=12)
    ax.grid(True, axis='x', alpha=0.25)
    ax.axvline(0, color='#cbd5e1', lw=1)
    for i, p in enumerate(pct):
        ax.text(p + 0.8, i, '{:.1f}%'.format(p), va='center', fontsize=11)
    path = os.path.join(out_dir, '04_val_improvement_pct.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    return path


def fig_dashboard(rows, out_dir):
    """Full 2x2 dashboard (slide-ready duplicate of bc_train layout)."""
    epochs = [int(r['epoch']) for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.2), facecolor=BG)

    ax = axes[0, 0]
    ax.plot(epochs, [float(r['train_loss']) for r in rows],
            color=C_TRAIN, lw=2, label='train')
    ax.plot(epochs, [float(r['val_loss']) for r in rows],
            color=C_VAL, lw=2, label='val')
    _style_axes(ax, 'Total loss', ylabel='MSE')
    ax.legend(fontsize=10)
    ax2 = ax.twinx()
    ax2.plot(epochs, [float(r['lr']) for r in rows],
             color=C_LR, ls='--', lw=1.5)
    ax2.set_ylabel('LR', fontsize=10, color=C_LR)

    for ax, tk, vk, title in [
        (axes[0, 1], 'train_steer', 'val_steer', 'Steer MSE'),
        (axes[1, 0], 'train_throttle', 'val_throttle', 'Throttle MSE'),
        (axes[1, 1], 'train_brake', 'val_brake', 'Brake MSE'),
    ]:
        ax.plot(epochs, [float(r[tk]) for r in rows],
                color=C_TRAIN, lw=2, label='train')
        ax.plot(epochs, [float(r[vk]) for r in rows],
                color=C_VAL, lw=2, label='val')
        _style_axes(ax, title, ylabel='MSE')
        ax.legend(fontsize=10)

    fig.suptitle('CARLA PilotNet — 30-Epoch Training Dashboard',
                 fontsize=17, fontweight='bold')
    path = os.path.join(out_dir, '05_training_dashboard.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', default=DEFAULT_CSV)
    p.add_argument('--summary', default=DEFAULT_JSON)
    p.add_argument('--out', default=DEFAULT_OUT)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = load_rows(args.csv)
    summary = load_summary(args.summary)
    if not rows:
        raise SystemExit('No rows in {}'.format(args.csv))

    paths = [
        fig_total_loss(rows, summary, args.out),
        fig_per_axis(rows, args.out),
        fig_val_comparison_bar(rows, args.out),
        fig_improvement(rows, args.out),
        fig_dashboard(rows, args.out),
    ]
    print('Saved presentation plots to:', args.out)
    for path in paths:
        print(' ', path)


if __name__ == '__main__':
    main()
