#!/usr/bin/env python
"""Train PilotNet from a *streaming* Hugging Face CARLA dataset (no full download).

Uses ``datasets.load_dataset(..., streaming=True)`` so data is fetched in shards
instead of syncing the entire release (the multimodal set is hundreds of GB:
see the dataset card on Hugging Face).

Filters rows by substring on ``map_name`` (default: ``Town10``) so you only
train on that town before optional fine-tune on your own ``bc_data_collector``
data via ``bc_train.py``.

Depends on::
    pip install datasets

Optional: set ``HF_TOKEN`` or ``HUGGING_FACE_HUB_TOKEN`` for authenticated Hub reads
(see ``hf_token_env.py``). Never commit tokens.

Dataset (MIT): https://huggingface.co/datasets/immanuelpeter/carla-autopilot-multimodal-dataset

Example::
    pip install datasets
    python bc_train_hf_stream.py --town-substring Town10 --epochs 5 --device cuda

Keys used from each row: ``image_front``, ``image_front_left``, ``image_front_right``,
``throttle``, ``steer``, ``brake``, ``map_name``.
"""

from __future__ import print_function

import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset

try:
    from datasets import load_dataset
except ImportError:
    raise RuntimeError(
        'Install Hugging Face datasets: pip install datasets')

from bc_dataset import preprocess_pil, DEFAULT_SIDE_CAM_STEER_OFFSET
from bc_model import build_model
from hf_token_env import get_hf_hub_token


# ---------- loss (mirror bc_train.py) ----------


class WeightedBCLoss(nn.Module):
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


# ---------- iterable HF dataset ----------


def _pil_rgb(img):
    if img is None:
        return None
    if hasattr(img, 'convert'):
        return img.convert('RGB')
    raise TypeError('Unexpected image field type: %s' % type(img))


class HFTownCarlaIterable(IterableDataset):
    """Stream rows from HF, filter town, optionally 3-camera expand + augment."""

    def __init__(self,
                 split,
                 dataset_id,
                 town_substring,
                 seed,
                 use_side_cameras,
                 side_offset,
                 augment,
                 drop_zero_steer_prob,
                 hf_token=None):
        super(HFTownCarlaIterable, self).__init__()
        self.split = split
        self.dataset_id = dataset_id
        self.town_substring = town_substring
        self.seed = seed
        self.use_side_cameras = use_side_cameras
        self.side_offset = side_offset
        self.augment = augment
        self.drop_zero_steer_prob = drop_zero_steer_prob
        self.hf_token = hf_token

    def _iter_base_stream(self):
        load_kw = {"streaming": True}
        if self.hf_token:
            load_kw["token"] = self.hf_token
        ds = load_dataset(self.dataset_id, split=self.split, **load_kw)

        town = self.town_substring or ''

        def _keep(ex):
            m = ex.get('map_name')
            if m is None:
                return False
            return town in str(m)

        ds = ds.filter(_keep)
        if self.split == 'train':
            ds = ds.shuffle(seed=self.seed, buffer_size=5000)
        return ds

    def _augment(self, arr, steer, rng):
        if not self.augment:
            return arr, steer
        if rng.random() < 0.5:
            arr = arr[:, :, ::-1].copy()
            steer = -steer
        delta = float(rng.uniform(-0.25, 0.25))
        arr = np.clip(arr + delta, -1.0, 1.0)
        if rng.random() < 0.5:
            arr = arr + rng.normal(0, 0.02, arr.shape).astype(np.float32)
            arr = np.clip(arr, -1.0, 1.0)
        return arr.astype(np.float32), steer

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        stream = self._iter_base_stream()
        if info is not None:
            stream = stream.shard(
                num_shards=info.num_workers, index=info.id)

        rng = np.random.default_rng(self.seed + (0 if info is None else info.id))

        for row in stream:
            steer0 = float(row.get('steer', 0.0))
            thr0 = float(row.get('throttle', 0.0))
            brk0 = float(row.get('brake', 0.0))

            pils = {
                'center': _pil_rgb(row.get('image_front')),
                'left': _pil_rgb(row.get('image_front_left')),
                'right': _pil_rgb(row.get('image_front_right')),
            }
            if any(v is None for v in pils.values()):
                continue

            triples = []
            if self.use_side_cameras:
                triples.append((pils['center'], steer0, thr0, brk0))
                triples.append((pils['left'], steer0 + self.side_offset, thr0, brk0))
                triples.append((pils['right'], steer0 - self.side_offset, thr0, brk0))
            else:
                triples.append((pils['center'], steer0, thr0, brk0))

            for pil_img, st, th, br in triples:
                if (self.drop_zero_steer_prob > 0
                        and abs(st) < 0.02
                        and rng.random() < self.drop_zero_steer_prob):
                    continue

                arr = preprocess_pil(pil_img)
                if self.augment:
                    arr, st = self._augment(arr, st, rng)

                st = float(np.clip(st, -1.0, 1.0))
                th = float(np.clip(th, 0.0, 1.0))
                br = float(np.clip(br, 0.0, 1.0))

                yield {
                    'image': torch.from_numpy(arr).float(),
                    'steer': torch.tensor(st, dtype=torch.float32),
                    'throttle': torch.tensor(th, dtype=torch.float32),
                    'brake': torch.tensor(br, dtype=torch.float32),
                }


def run_epoch_limited(model, loader, criterion, optimizer, device, train,
                      max_batches):
    model.train() if train else model.eval()
    totals = {'loss': 0.0, 'steer': 0.0, 'throttle': 0.0, 'brake': 0.0, 'n': 0}
    ctx = torch.enable_grad() if train else torch.no_grad()
    n_batches = 0
    with ctx:
        for batch in loader:
            if max_batches is not None and n_batches >= max_batches:
                break
            n_batches += 1
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


def main():
    p = argparse.ArgumentParser(
        description='Stream-train PilotNet on HF CARLA autopilot (Town filter)')
    p.add_argument(
        '--dataset',
        default='immanuelpeter/carla-autopilot-multimodal-dataset',
        help='Hugging Face dataset id')
    p.add_argument(
        '--town-substring',
        default='Town10',
        help='keep rows where map_name contains this substring (e.g. Town10)')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument(
        '--train-batches-per-epoch', type=int, default=400,
        help='cap streaming train steps per epoch (each batch counts once)')
    p.add_argument(
        '--val-batches-per-epoch', type=int, default=120,
        help='cap validation batches per epoch')
    p.add_argument('--num-workers', type=int, default=0,
                   help='0 recommended for iterable HF streams (stable)')
    p.add_argument('--no-side-cameras', action='store_true')
    p.add_argument('--side-offset', type=float,
                   default=DEFAULT_SIDE_CAM_STEER_OFFSET)
    p.add_argument('--no-augment', action='store_true')
    p.add_argument('--drop-zero-steer-prob', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--out', default='models/hf_stream_bc.pt')

    args = p.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    hf_token = get_hf_hub_token()
    print('[HF-BC] Dataset:', args.dataset)
    print('[HF-BC] Town filter: substring in map_name:', repr(args.town_substring))
    print('[HF-BC] HF auth:', 'token from env' if hf_token else 'anonymous')
    print('[HF-BC] Streaming mode (partial download via shards)')
    print('[HF-BC] Train batches/epoch:', args.train_batches_per_epoch,
          '| val:', args.val_batches_per_epoch)

    train_ds = HFTownCarlaIterable(
        split='train',
        dataset_id=args.dataset,
        town_substring=args.town_substring,
        seed=args.seed,
        use_side_cameras=not args.no_side_cameras,
        side_offset=args.side_offset,
        augment=not args.no_augment,
        drop_zero_steer_prob=args.drop_zero_steer_prob,
        hf_token=hf_token,
    )
    val_ds = HFTownCarlaIterable(
        split='validation',
        dataset_id=args.dataset,
        town_substring=args.town_substring,
        seed=args.seed + 1,
        use_side_cameras=False,
        side_offset=args.side_offset,
        augment=False,
        drop_zero_steer_prob=0.0,
        hf_token=hf_token,
    )

    pin = device.type == 'cuda'
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=pin)

    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print('[HF-BC] PilotNet parameters: {:,}'.format(n_params))
    print('[HF-BC] Device:', device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs))
    criterion = WeightedBCLoss()

    out_dir = os.path.dirname(args.out) or '.'
    os.makedirs(out_dir, exist_ok=True)
    last_path = (args.out.replace('.pt', '.last.pt')
                 if args.out.endswith('.pt') else args.out + '.last.pt')

    best_val = math.inf
    print('[HF-BC] Training...')
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch_limited(
            model, train_loader, criterion, optimizer, device, train=True,
            max_batches=args.train_batches_per_epoch)
        vl = run_epoch_limited(
            model, val_loader, criterion, optimizer, device, train=False,
            max_batches=args.val_batches_per_epoch)
        scheduler.step()
        dt = time.time() - t0
        print(('[HF-BC] Epoch {ep:3d}/{tot}  '
               'train {trl:.4f} (s {trs:.4f} t {trt:.4f} b {trb:.4f})  '
               'val {vll:.4f} (s {vls:.4f} t {vlt:.4f} b {vlb:.4f})  '
               'lr {lr:.2e}  {dt:.1f}s').format(
            ep=epoch, tot=args.epochs,
            trl=tr['loss'], trs=tr['steer'], trt=tr['throttle'], trb=tr['brake'],
            vll=vl['loss'], vls=vl['steer'], vlt=vl['throttle'], vlb=vl['brake'],
            lr=optimizer.param_groups[0]['lr'], dt=dt))

        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'val_loss': vl['loss'],
            'dataset': args.dataset,
            'town_substring': args.town_substring,
        }, last_path)

        if vl['loss'] < best_val:
            best_val = vl['loss']
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': vl['loss'],
                'dataset': args.dataset,
                'town_substring': args.town_substring,
            }, args.out)
            print('[HF-BC]   * best val {:.4f} -> {}'.format(best_val, args.out))

    print('[HF-BC] Done. Best val:', best_val)
    print('[HF-BC] Next: adapt to your rigs with bc_train.py on dataset/run*')


if __name__ == '__main__':
    main()
