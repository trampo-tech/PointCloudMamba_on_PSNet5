"""Deterministic coverage-potential schedules for PSNet5 spheres."""
from __future__ import annotations

import os
import json
import tempfile
from pathlib import Path

import numpy as np


def _valid_jitter(rng, point, cloud, radius):
    for _ in range(100):
        noise = rng.normal(scale=radius / 10.0, size=(3,)).astype(np.float32)
        pick = point + noise
        axis = int(cloud['axis'])
        if axis < 0:
            return noise
        boundary = float(cloud['threshold'])
        buffer_radius = float(cloud['buffer_radius'])
        if cloud['role'] == 'train' and pick[axis] < boundary - buffer_radius:
            return noise
        if cloud['role'] != 'train' and pick[axis] >= boundary + buffer_radius:
            return noise
    raise RuntimeError('Could not draw a center jitter that respects the split buffer')


def make_schedule(clouds, num_epochs, num_steps, radius, num_points, seed, path,
                  class_sampling_ratio=0.0):
    ratio = float(class_sampling_ratio)
    if not 0 <= ratio <= 1:
        raise ValueError('class_sampling_ratio must be between 0 and 1')
    if ratio and any(cloud['role'] != 'train' for cloud in clouds):
        raise ValueError('Class-aware sampling is only allowed for training')
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if float(data.get('class_sampling_ratio', 0.0)) != ratio:
                raise ValueError('Cached schedule has a different class_sampling_ratio')
            return data['cloud_indices'], data['point_indices'], data['noise']
    rng = np.random.default_rng(seed)
    potentials, eligible_indices = [], []
    for cloud in clouds:
        eligible = np.flatnonzero(np.asarray(cloud['eligible']))
        eligible_indices.append(eligible)
        potentials.append(rng.random(len(cloud['sub_points'])) * 1e-3)
    # Separate RNG keeps center-choice draws separate from spatial jitter.
    class_rng = np.random.default_rng(seed + 1)
    class_candidates = {}
    if ratio:
        for cloud_index, eligible in enumerate(eligible_indices):
            labels = np.asarray(clouds[cloud_index]['sub_labels']).reshape(-1)
            for label in np.unique(labels[eligible]):
                candidates = eligible[labels[eligible] == label]
                class_candidates.setdefault(int(label), []).append((cloud_index, candidates))
        if not class_candidates:
            raise ValueError('No eligible training classes for balanced sampling')
    classes = sorted(class_candidates)
    center_counts = np.zeros((num_epochs, 5), dtype=np.int64)
    valid_counts = np.zeros_like(center_counts)
    balanced_counts = np.zeros_like(center_counts)
    cloud_indices = np.empty(num_epochs * num_steps, dtype=np.int16)
    point_indices = np.empty(num_epochs * num_steps, dtype=np.int32)
    noise_values = np.empty((num_epochs * num_steps, 3), dtype=np.float32)
    for schedule_index in range(len(cloud_indices)):
        minimums = [potential[eligible].min() for potential, eligible in zip(potentials, eligible_indices)]
        cloud_index = int(np.argmin(minimums))
        eligible = eligible_indices[cloud_index]
        point_index = int(eligible[np.argmin(potentials[cloud_index][eligible])])
        balanced = bool(ratio and class_rng.random() < ratio)
        if balanced:
            label = int(class_rng.choice(classes))
            candidates = class_candidates[label]
            # Pick the least-covered eligible point of this class globally.
            candidate_minima = [potentials[ci][inds].min() for ci, inds in candidates]
            cloud_index, eligible = candidates[int(np.argmin(candidate_minima))]
            point_index = int(eligible[np.argmin(potentials[cloud_index][eligible])])
        cloud = clouds[cloud_index]
        point = np.asarray(cloud['sub_points'][point_index])
        noise = _valid_jitter(rng, point, cloud, radius)
        pick = point + noise
        query = cloud['tree'].query_radius(pick.reshape(1, -1), r=radius,
                                           return_distance=True, sort_results=True)[0][0]
        query = query[:num_points]
        if not len(query):
            raise RuntimeError('Eligible center produced an empty neighborhood')
        if ratio:
            epoch = schedule_index // num_steps
            labels = np.asarray(cloud['sub_labels']).reshape(-1)
            label = int(labels[point_index])
            center_counts[epoch, label] += 1
            balanced_counts[epoch, label] += int(balanced)
            valid_counts[epoch] += np.bincount(labels[query], minlength=5)
        squared_distance = np.sum((np.asarray(cloud['sub_points'][query]) - pick) ** 2, axis=1)
        tukey = np.square(1.0 - squared_distance / radius ** 2)
        potentials[cloud_index][query] += tukey
        cloud_indices[schedule_index] = cloud_index
        point_indices[schedule_index] = point_index
        noise_values[schedule_index] = noise
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as stream:
        temporary = Path(stream.name)
        np.savez(stream, cloud_indices=cloud_indices, point_indices=point_indices, noise=noise_values,
                 class_sampling_ratio=ratio)
    os.replace(temporary, path)
    path.chmod(0o644)
    if ratio:
        diagnostics = {
            'class_sampling_ratio': ratio,
            'class_order': ['ibeam', 'pipe', 'pump', 'rectangularbeam', 'tank'],
            'available_center_classes': classes,
            'center_counts_per_epoch': center_counts.tolist(),
            'class_selected_center_counts_per_epoch': balanced_counts.tolist(),
            'valid_point_counts_per_epoch': valid_counts.tolist(),
            'note': 'Valid point counts include repeated exposure across spheres, exclude padding.'}
        path.with_suffix('.sampling.json').write_text(json.dumps(diagnostics, indent=2))
    return cloud_indices, point_indices, noise_values
