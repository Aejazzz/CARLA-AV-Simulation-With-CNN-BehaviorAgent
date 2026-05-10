"""PyTorch Dataset for the CARLA Behavioral Cloning project.

Reads one or more dataset directories produced by ``bc_data_collector.py``.

Each row in ``labels.csv`` is expanded into THREE training samples using the
NVIDIA 3-camera trick: the left/right cameras get a synthetic steering offset
that teaches the model to recover toward the lane center.

Augmentations (training only):
  * random horizontal flip (steer is negated)
  * random brightness jitter
  * Gaussian noise

Image preprocessing:
  * crop top sky band and bottom hood band
  * resize to (IMG_H, IMG_W) = (66, 200)  -> PilotNet input
  * normalise to [-1, 1]
"""

from __future__ import print_function

import csv
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from bc_model import IMG_H, IMG_W

# Crop fractions: how much of the raw frame to discard from top (sky) and
# bottom (vehicle hood). Tuned for the camera height/FoV used in
# bc_data_collector.py (320x180, FoV 90, z=1.7 m).
CROP_TOP = 0.35
CROP_BOTTOM = 0.10

# NVIDIA-style synthetic steer correction added to side-camera labels.
DEFAULT_SIDE_CAM_STEER_OFFSET = 0.20


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _resolve_image(run_dir, cam, frame_int):
    """Return the path to <run>/<cam>/<frame>.{jpg|png} if it exists, else None.

    The collector now writes JPEGs (much faster I/O); old datasets may still
    contain PNGs. We accept both transparently.
    """
    base = os.path.join(run_dir, cam, '{:08d}'.format(frame_int))
    for ext in ('.jpg', '.jpeg', '.png'):
        p = base + ext
        if os.path.exists(p):
            return p
    return None


def _read_labels(run_dir):
    """Return a list of dicts, one per CSV row, with absolute image paths."""
    csv_path = os.path.join(run_dir, 'labels.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError('No labels.csv in ' + run_dir)
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            frame_int = int(r['frame'])
            paths = {
                'center': _resolve_image(run_dir, 'center', frame_int),
                'left':   _resolve_image(run_dir, 'left',   frame_int),
                'right':  _resolve_image(run_dir, 'right',  frame_int),
            }
            if any(p is None for p in paths.values()):
                continue  # silently skip rows whose images were lost / dropped
            rows.append({
                'paths': paths,
                'steer': float(r['steer']),
                'throttle': float(r['throttle']),
                'brake': float(r['brake']),
                'speed': float(r.get('speed_kmh', 0.0)),
            })
    return rows


def discover_runs(root):
    """Find all subfolders of ``root`` that look like dataset runs."""
    if os.path.exists(os.path.join(root, 'labels.csv')):
        return [root]
    runs = []
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if os.path.isdir(full) and os.path.exists(os.path.join(full, 'labels.csv')):
            runs.append(full)
    if not runs:
        raise FileNotFoundError('No dataset runs found under ' + root)
    return runs


def load_all_rows(runs):
    """Concatenate label rows from every run. Returns list[dict]."""
    rows = []
    for run in runs:
        rows.extend(_read_labels(run))
    return rows


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def preprocess_pil(img):
    """Crop sky + hood, resize to PilotNet input, return float32 RGB array
    in [-1, 1] with shape (3, IMG_H, IMG_W)."""
    w, h = img.size
    top = int(h * CROP_TOP)
    bottom = h - int(h * CROP_BOTTOM)
    img = img.crop((0, top, w, bottom))
    img = img.resize((IMG_W, IMG_H), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0  # [-1, 1]
    return np.transpose(arr, (2, 0, 1))  # HWC -> CHW


def preprocess_numpy_rgb(rgb_uint8):
    """Same as preprocess_pil but takes a HxWx3 uint8 numpy array. Used by
    automatic_controll.py for live inference."""
    img = Image.fromarray(rgb_uint8)
    return preprocess_pil(img)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CarlaBCDataset(Dataset):
    """Behavioral cloning dataset.

    Pass either ``runs`` (a list of run directories) or ``rows`` (a list of
    pre-loaded label dicts from :func:`load_all_rows`). Passing ``rows`` is
    the recommended way to do clean train/val splits at the timestamp level.

    Parameters
    ----------
    runs, rows
        Source of label data. Provide exactly one.
    use_side_cameras : bool
        If True, expand each CSV row into 3 samples (center/left/right) with
        a steering offset on the sides. Disable for clean validation.
    side_offset : float
        Steering correction added to ``left`` and subtracted from ``right``.
    augment : bool
        Apply random brightness, noise, and horizontal flip (steer negation).
    drop_zero_steer_prob : float
        Probability of skipping a sample whose absolute steering is < 0.02.
        Mitigates the heavy bias toward straight-line driving. 0 disables it.
    """

    def __init__(self,
                 runs=None,
                 rows=None,
                 use_side_cameras=True,
                 side_offset=DEFAULT_SIDE_CAM_STEER_OFFSET,
                 augment=True,
                 drop_zero_steer_prob=0.0):
        super(CarlaBCDataset, self).__init__()
        if (runs is None) == (rows is None):
            raise ValueError('Pass exactly one of runs= or rows=')
        if rows is None:
            rows = load_all_rows(runs)

        self.augment = augment
        self.drop_zero_steer_prob = drop_zero_steer_prob

        self.samples = []
        for row in rows:
            if use_side_cameras:
                self.samples.append((row['paths']['center'],
                                     row['steer'], row['throttle'], row['brake']))
                self.samples.append((row['paths']['left'],
                                     row['steer'] + side_offset,
                                     row['throttle'], row['brake']))
                self.samples.append((row['paths']['right'],
                                     row['steer'] - side_offset,
                                     row['throttle'], row['brake']))
            else:
                self.samples.append((row['paths']['center'],
                                     row['steer'], row['throttle'], row['brake']))
        if not self.samples:
            raise RuntimeError('Dataset is empty - did you actually press R while driving?')

        self._rng = np.random.default_rng(0)

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def _augment(self, arr, steer):
        # Random horizontal flip (negate steer label)
        if self._rng.random() < 0.5:
            arr = arr[:, :, ::-1].copy()
            steer = -steer
        # Random brightness in [-0.25, +0.25] applied in normalised space
        delta = float(self._rng.uniform(-0.25, 0.25))
        arr = np.clip(arr + delta, -1.0, 1.0)
        # Light Gaussian noise
        if self._rng.random() < 0.5:
            arr = arr + self._rng.normal(0, 0.02, arr.shape).astype(np.float32)
            arr = np.clip(arr, -1.0, 1.0)
        return arr.astype(np.float32), steer

    def __getitem__(self, idx):
        path, steer, throttle, brake = self.samples[idx]

        # Optional under-sampling of zero-steer frames. We can't return None
        # from __getitem__, so we resample to a different index instead.
        if (self.drop_zero_steer_prob > 0
                and abs(steer) < 0.02
                and self._rng.random() < self.drop_zero_steer_prob):
            return self.__getitem__((idx + 1) % len(self.samples))

        with Image.open(path) as img:
            img = img.convert('RGB')
            arr = preprocess_pil(img)

        if self.augment:
            arr, steer = self._augment(arr, steer)

        steer = float(np.clip(steer, -1.0, 1.0))
        throttle = float(np.clip(throttle, 0.0, 1.0))
        brake = float(np.clip(brake, 0.0, 1.0))

        return {
            'image': torch.from_numpy(arr).float(),
            'steer': torch.tensor(steer, dtype=torch.float32),
            'throttle': torch.tensor(throttle, dtype=torch.float32),
            'brake': torch.tensor(brake, dtype=torch.float32),
        }


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Inspect a CARLA BC dataset')
    p.add_argument('root', help='dataset root or single run directory')
    args = p.parse_args()
    runs = discover_runs(args.root)
    ds = CarlaBCDataset(runs=runs, augment=False)
    print('Found {} run(s), {} samples (with 3-cam expansion).'.format(len(runs), len(ds)))
    sample = ds[0]
    for k, v in sample.items():
        if torch.is_tensor(v):
            print(' ', k, tuple(v.shape), v.dtype)
        else:
            print(' ', k, v)
