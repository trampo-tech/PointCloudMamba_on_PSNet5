# Point Cloud Mamba on PSNet5

This repository contains an adaptation of Point Cloud Mamba (PCM) for five-class semantic segmentation of industrial point clouds in PSNet5, together with dataset analysis, development experiments, evaluation outputs, and figures. The original implementations are [Point Cloud Mamba](https://github.com/zhang-tao-whu/PCM) and [ResPointNet++ / PSNet5](https://github.com/PointCloudYC/ResPointNet2). Our working repositories are [PCM](https://github.com/trampo-tech/PointCloudMamba_on_PSNet5.git) and [ResPointNet++](https://github.com/trampo-tech/respointnet_compute_proof).

## Artifact inventory

Paths are relative to this repository unless stated otherwise.

- `examples/segmentation/main.py`: training and validation entry point. `--cfg` selects the model configuration; the run directory under `log/psnet5/` receives resolved configurations, logs, metrics, and checkpoints.
- `openpoints/models/PCM/`, `openpoints/models/segmentation/`, and `openpoints/cpp/`: model implementation and required native operators.
- `openpoints/dataset/psnet5/`: PSNet5 preprocessing, sphere sampling, and voting implementation.
- `cfgs/psnet5/default.yaml` and `cfgs/psnet5/PCM-ce-weighted.yaml`: dataset, optimization and model settings for the weighted cross-entropy experiment. `PCM-ce-classmix.yaml` specifies the class-aware sampling experiment; `PCM-ce-nosmoothing.yaml` describes an incomplete experiment, not a completed result. `cfgs/psnet5/README.md` gives further protocol details.
- `analysis/psnet5/split.json` and `analysis/psnet5/results/Area_{1,2,4}_partition.npz`: development split and partition masks. `analysis/psnet5/REPORT.md` describes the data and split.
- `analysis/psnet5/evaluate_multivote.py`: evaluation of a supplied checkpoint. `--cfg` selects its model settings, `--checkpoint` the weights, `--benchmark-area3` the Area 3 evaluation, `--votes` the number of voting passes, `--num-steps` the spheres per pass, `--seed` the evaluation seed, and `--output` the results directory.
- `analysis/psnet5/results/weighted_ce_epoch39_area3/`: recorded Area 3 summary, per-class metrics, confusion matrix, vote metrics and prediction export for the weighted run. `analysis/psnet5/weighted_ce_40_report/` contains its report and figures. The matched baseline is in `analysis/psnet5/batchnorm_40_report/` and `analysis/psnet5/results/batchnorm_40_fixed_eval_area3/`.
- `analysis/psnet5/CURRENT_FINDINGS.md`, `PADDING_INVESTIGATION.md`, and `REFERENCE_EVALUATION_AUDIT.md`: findings and limitations, padding investigation, and evaluation protocol audit. Other subdirectories under `analysis/psnet5/results/` contain class-mix, sensitivity, exposure and earlier diagnostic results; analysis scripts are under `analysis/psnet5/`.
- `Dockerfile`, `requirements.txt`, and `INSTALL.md`: environment and installation resources. `tests/` contains implementation checks.

## Recorded best result

The weighted experiment trained for 40 epochs with seed `20261004`, batch size 2, 15,000 points per sphere, radius 2 m, 0.04 m voxel size, masked cross-entropy, label smoothing 0.2, and class weights computed from training-partition frequencies as `1 / (frequency + 0.02)`, normalized to mean 1. The selected checkpoint was epoch 39, with development mIoU 86.08%. The separate Area 3 evaluation used one vote, 2,000 spheres and current-sphere BatchNorm statistics, reporting 69.23% mIoU, 85.08% overall accuracy and 100% raw-point coverage. See the report and `summary.json` above for per-class results.

Training and development validation used spatial partitions of Areas 1, 2 and 4. 

## Reproduce the workflow

The following commands describe the current configuration. The Dockerfile uses PyTorch 2.1.2, CUDA 11.8 and cuDNN 8; building requires Docker BuildKit and the build downloads dependencies. Running requires an NVIDIA GPU and Docker GPU support.

1. Obtain the PSNet5 data and place the original `Area_1` through `Area_4` annotation directories and/or the matching `processed/Area_*.pkl` files under a local directory. The development split validates source digests against the audited data; different source caches may fail validation. The relative default data path in `cfgs/psnet5/default.yaml` resolves inside the container to `/workspace/ResPointNet2/data/PSNet/PSNet5`.
2. From this repository, build the image and create writable local directories for the generated cache, logs and evaluation outputs:

```bash
docker build -t pcm-psnet5 .
mkdir -p data/PSNet5 log/psnet5 output/psnet5
```

3. Replace `/absolute/path/to/PSNet5` with the data directory and start the container:

```bash
docker run --rm -it --gpus all --ipc=host --shm-size=16g \
  -v /absolute/path/to/PSNet5:/workspace/ResPointNet2/data/PSNet/PSNet5:ro \
  -v "$PWD/data/PSNet5:/workspace/PointCloudMamba/data/PSNet5" \
  -v "$PWD/log/psnet5:/workspace/PointCloudMamba/log/psnet5" \
  -v "$PWD/output/psnet5:/workspace/PointCloudMamba/output/psnet5" \
  pcm-psnet5
```

4. Inside the container, run the smoke configuration, then train the weighted experiment:

```bash
python examples/segmentation/main.py --cfg cfgs/psnet5/PCM-smoke.yaml
python examples/segmentation/main.py --cfg cfgs/psnet5/PCM-ce-weighted.yaml
```

The run creates its versioned processed cache on first use. Find its selected `*_ckpt_best.pth` under `log/psnet5/<run>/checkpoint/`. To evaluate a checkpoint accessible inside the container, substitute its path below:

```bash
python analysis/psnet5/evaluate_multivote.py \
  --cfg cfgs/psnet5/PCM-ce-weighted.yaml \
  --checkpoint /path/to/ckpt_best.pth \
  --benchmark-area3 --votes 1 --num-steps 2000 --seed 20261004 \
  --output output/psnet5/weighted-area3
```

The evaluator writes `summary.json`, per-class and vote metrics, confusion matrices, and per-area prediction exports to `--output`. Current-sphere BatchNorm is enabled by the default configuration. The output from a newly trained checkpoint is a new run, not the original saved result. See `cfgs/psnet5/README.md` for coverage and evaluation details.
