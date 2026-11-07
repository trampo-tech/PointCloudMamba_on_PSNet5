#!/usr/bin/env python3
"""Generate PSNet5 paper figures from trusted local data and saved evaluations."""
import argparse
import csv
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/psnet5-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NAMES = ['I-beam', 'Pipe', 'Pump', 'Rectangular beam', 'Tank']
COLORS = ['#d55e00', '#0072b2', '#cc79a7', '#009e73', '#e6b800']
CMAP = ListedColormap(COLORS)
LEGEND = [Patch(facecolor=c, label=n) for c, n in zip(COLORS, NAMES)]
RESULTS = ROOT / 'analysis/psnet5/results'


def save(fig, out, name):
    fig.savefig(out / (name + '.png'), dpi=250, bbox_inches='tight', facecolor='white')
    fig.savefig(out / (name + '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Saved', name, flush=True)


def semantic(ax, xy, labels, size=.2):
    ax.scatter(xy[:, 0], xy[:, 1], c=labels, cmap=CMAP, vmin=-.5, vmax=4.5,
               s=size, linewidths=0, rasterized=True)
    ax.set_aspect('equal'); ax.set_axis_off()


def subset(n, maximum, rng):
    return np.sort(rng.choice(n, min(n, maximum), replace=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT.parent / 'ResPointNet2/data/PSNet/PSNet5', help='PSNet5 directory with trusted processed/Area_N.pkl caches')
    parser.add_argument('--cache-root', type=Path, default=ROOT / 'data/PSNet5/processed_pcm/psnet5sphere_benchmark_benchmark-area-v1_test_v0.040')
    parser.add_argument('--output', type=Path, default=ROOT / 'analysis/psnet5/paper_figures')
    parser.add_argument('--seed', type=int, default=20261004)
    parser.add_argument('--points', type=int, default=200000, help='Maximum displayed points per area')
    args = parser.parse_args(); out = args.output; out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})

    # Original RGB and labels, from the same sampled points in each panel.
    fig, axes = plt.subplots(4, 2, figsize=(10, 14), constrained_layout=True)
    for area in range(1, 5):
        print('Loading Area', area, flush=True)
        with (args.data_root / 'processed' / f'Area_{area}.pkl').open('rb') as stream:
            xyz, rgb, labels = pickle.load(stream)
        idx = subset(len(xyz), args.points, rng)
        xy = np.asarray(xyz[idx, :2]); col = np.asarray(rgb[idx], dtype=float); lab = np.asarray(labels).reshape(-1)[idx]
        if col.max() > 1: col /= 255
        axes[area-1, 0].scatter(xy[:, 0], xy[:, 1], c=np.clip(col, 0, 1), s=.2, linewidths=0, rasterized=True)
        axes[area-1, 0].set_aspect('equal'); axes[area-1, 0].set_axis_off()
        semantic(axes[area-1, 1], xy, lab)
        axes[area-1, 0].set_title(f'Area {area} · RGB'); axes[area-1, 1].set_title(f'Area {area} · ground truth')
        del xyz, rgb, labels, idx, xy, col, lab
    fig.legend(handles=LEGEND, loc='outside lower center', ncol=5, frameon=False)
    save(fig, out, '01_dataset_overview')

    cache = args.cache_root
    xyz = np.load(cache / 'cloud_0_raw_points.npy', mmap_mode='r')
    cached_labels = np.load(cache / 'cloud_0_raw_labels.npy', mmap_mode='r').reshape(-1)
    pred_path = RESULTS / 'weighted_ce_epoch39_area3/Area_3_predictions.npz'
    with np.load(pred_path, allow_pickle=False) as data:
        truth = data['label'].reshape(-1); predicted = data['prediction'].reshape(-1); covered = data['covered'].reshape(-1)
    if len(xyz) != len(truth) or not np.array_equal(truth, cached_labels):
        raise ValueError('Area 3 cache and prediction labels do not match')
    if not covered.all(): raise ValueError('Area 3 prediction coverage is incomplete')
    idx = subset(len(xyz), args.points, rng); points = np.asarray(xyz[idx]); y = truth[idx]; p = predicted[idx]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for ax, labels, title in zip(axes, (y, p), ('Ground truth', 'Weighted CE prediction · epoch 39')):
        semantic(ax, points[:, :2], labels, .35); ax.set_title(title)
    fig.legend(handles=LEGEND, loc='outside lower center', ncol=5, frameon=False)
    save(fig, out, '02_area3_predictions')

    with (RESULTS / 'weighted_ce_epoch39_area3/confusion_vote_01.csv').open(newline='') as f:
        rows = list(csv.reader(f))
    matrix = np.array([[int(x) for x in row[1:]] for row in rows[1:]])
    fig, ax = plt.subplots(figsize=(7.5, 6), constrained_layout=True)
    percentages = matrix / matrix.sum(axis=1, keepdims=True) * 100
    im = ax.imshow(percentages, vmin=0, vmax=100, cmap='Blues')
    ax.set_xticks(range(5), NAMES, rotation=35, ha='right'); ax.set_yticks(range(5), NAMES)
    ax.set(xlabel='Predicted class', ylabel='Ground-truth class', title='Area 3 · weighted CE · epoch 39')
    for i in range(5):
        for j in range(5): ax.text(j, i, f'{percentages[i,j]:.1f}%', ha='center', va='center', color='white' if percentages[i,j]>55 else '#20252a')
    fig.colorbar(im, ax=ax, label='Share of ground-truth class (%)')
    save(fig, out, '03_confusion_matrix')

    base = json.loads((RESULTS / 'batchnorm_40_fixed_eval_area3/summary.json').read_text())['final']['per_class']
    weighted = json.loads((RESULTS / 'weighted_ce_epoch39_area3/summary.json').read_text())['final']['per_class']
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True); x = np.arange(5)
    ax.bar(x-.19, [r['iou']*100 for r in base], .38, label='Unweighted baseline', color='#8a99a6')
    ax.bar(x+.19, [r['iou']*100 for r in weighted], .38, label='Weighted CE', color='#0072b2')
    ax.set_xticks(x, NAMES, rotation=18, ha='right'); ax.set(ylabel='Intersection over union (%)', ylim=(0, 105), title='Area 3 · per-class IoU')
    ax.legend(frameon=False); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    save(fig, out, '04_per_class_iou')

    # Local same-class regions from the deterministic Area 3 display sample.
    fig = plt.figure(figsize=(15, 3.5), constrained_layout=True)
    for c in range(5):
        ax = fig.add_subplot(1, 5, c+1, projection='3d'); cloud = points[y == c]
        center = cloud[np.argmin(np.sum((cloud - np.median(cloud, axis=0))**2, axis=1))]
        local = cloud[np.linalg.norm(cloud-center, axis=1) < 2.0]
        if len(local)>10000: local = local[subset(len(local), 10000, rng)]
        ax.scatter(local[:,0], local[:,1], local[:,2], s=.8, color=COLORS[c], linewidth=0, rasterized=True)
        ax.set_title(NAMES[c]); ax.set_axis_off(); ax.view_init(elev=25, azim=-55)
        if len(local)>1: ax.set_box_aspect(np.maximum(np.ptp(local, axis=0), .01))
    fig.suptitle('Area 3 · representative local class regions (2 m radius)')
    save(fig, out, '05_class_examples')

    fig, axes = plt.subplots(2, 2, figsize=(9, 9), constrained_layout=True)
    for row, (a, b) in enumerate(((0, 1), (2, 1))):
        error = points[(y == a) & (p == b)]
        if len(error)==0: raise ValueError('No sampled points for selected error')
        center = np.median(error[:, :2], axis=0)
        mask = (np.abs(points[:,0]-center[0]) < 3) & (np.abs(points[:,1]-center[1]) < 3)
        for col, labels in enumerate((y, p)):
            semantic(axes[row,col], points[mask,:2], labels[mask], 1.3)
            axes[row,col].set_xlim(center[0]-3, center[0]+3); axes[row,col].set_ylim(center[1]-3, center[1]+3)
            axes[row,col].set_title(f'{NAMES[a]} → {NAMES[b]} · ' + ('ground truth' if col == 0 else 'prediction'))
    fig.legend(handles=LEGEND, loc='outside lower center', ncol=5, frameon=False)
    save(fig, out, '06_error_regions')
    (out / 'README.md').write_text('''# PSNet5 paper figures\n\nGenerated by `python analysis/psnet5/make_paper_figures.py` with seed %d and up to %d displayed points per area. Both PNG and PDF are provided. Point-cloud views are deterministic samples, not full-density renders; XY views can overlap vertically. Class examples are 2 m local regions, not isolated object instances. Error crops are illustrative regions selected from sampled errors.\n\n- 01: RGB and ground-truth labels for the four areas.\n- 02: Area 3 ground truth and weighted epoch-39 predictions, identical sampled points.\n- 03: Area 3 confusion matrix, row-normalized by ground-truth class; source counts are `../results/weighted_ce_epoch39_area3/confusion_vote_01.csv`.\n- 04: Per-class Area 3 IoU for the unweighted baseline and weighted model.\n- 05: Local examples of the five ground-truth classes in Area 3.\n- 06: Illustrative regions of i-beam-to-pipe and pump-to-pipe errors.\n\nSource data and the benchmark cache are external to Git; see script options and the repository README. Area 3 informed development and is not an untouched test set.\n''' % (args.seed, args.points))

if __name__ == '__main__': main()
