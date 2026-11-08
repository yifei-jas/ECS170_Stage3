# ECS 170 Stage 3 CNN experiments

This implementation extends the course base classes with three CNN subclasses
and an image loader. Stage 1 and Stage 2 sources/results remain separate.

## Data

Download the instructor's Stage 3 archive:
https://drive.google.com/file/d/1-udd9jkldM1gEQ-OO-9HKXds42AiLOL6/view

Extract it so stage_3_data/MNIST, stage_3_data/ORL and stage_3_data/CIFAR exist.
These trusted instructor files use pickle. Do not load arbitrary pickle files.
The archive/data files are excluded from the source submission package.

Original partitions:

| Dataset | Training | Testing | Original image | Classes |
|---|---:|---:|---|---:|
| MNIST | 60,000 | 10,000 | 28x28 grayscale | 10 |
| ORL | 360 | 40 | 112x92 RGB with identical channels | 40 |
| CIFAR-10 | 50,000 | 10,000 | 32x32 RGB | 10 |

The training holdout is stratified with seed 42: 6,000 validation MNIST images,
40 validation ORL images (one per person), and 5,000 validation CIFAR images.
Original test data is never used for training or configuration selection.
ORL uses one channel, resized to 56x46 with bilinear interpolation, and labels
are converted from 1-40 to 0-39. All pixels are scaled by 255; channel mean/std
are calculated on fitting images only. CIFAR training uses reflected four-pixel
padding, independently random 32x32 crops and horizontal flips. Validation/test
images receive no random augmentation.

## Architectures

Every convolution uses padding 1, kernel 3, stride 1, no bias, BatchNorm and
ReLU. Each block ends with 2x2 max pooling. Baseline has one convolution per
block. MNIST/CIFAR deeper variants have two convolutions per block. ORL's
deeper variant adds a third block. No pretrained weights are used.

| Dataset | Channels | Pool before classifier | Classifier |
|---|---|---|---|
| MNIST | 16,32 | 2x2 average pool; flatten 32x3x3 | 288 -> 128 -> 10 |
| ORL baseline | 16,32 | 2x2 average pool; flatten 32x7x5 | 1120 -> 128 -> 40 |
| ORL deeper | 16,32,64 | identity; flatten 64x7x5 | 2240 -> 128 -> 40 |
| CIFAR | 32,64,128 | identity; flatten 128x4x4 | 2048 -> 256 -> 10 |

Hidden classifier layers use ReLU and dropout 0.2; outputs are raw logits.
CrossEntropyLoss is used on integer labels. Initialization uses PyTorch defaults.

## Training

Dependencies: torch, numpy, scikit-learn, matplotlib (recorded environment in
requirements_stage3.txt and each run's metrics.json). GPU is chosen when present.

```sh
python script/stage_3_script/script_experiments.py
```

Default epoch budgets: MNIST 8, ORL 80, CIFAR 40, for both variants. Adam learning
rate 0.001 with cosine annealing to 0.00001; weight decay 0.0001 for CIFAR only.
Batch sizes: ORL 32, others 128. Seed 42 and deterministic algorithms enabled.
Hardware/software differences can still affect numerical reproducibility.

To run one dataset or change the budget:

```sh
python script/stage_3_script/script_experiments.py --datasets MNIST --mnist-epochs 8
```

Completed experiment outputs are under result/stage_3_result/comparison_v2.
Selection uses validation accuracy, then validation loss as a tiebreaker.
After selection is saved, both checkpoints are evaluated on the supplied test set.
ORL has only 40 test examples: results have 2.5 percentage point granularity.
The --resume option reuses completed runs in the named directory; it does not
resume an interrupted epoch sequence inside an incomplete run.

```sh
python script/stage_3_script/script_verify_results.py result/stage_3_result/comparison_v2
python script/stage_3_script/script_summarize.py result/stage_3_result/comparison_v2
```

Each model has model.pt, history.csv, metrics.json, split_indices.npz,
validation_predictions.csv, test_predictions.csv and test_metrics.json. All
precision/recall/F1 values include macro and weighted averages over the actual
class count (40 for ORL). Root comparison.json/csv summarize the runs.

## Additional CIFAR residual model

The additional CIFAR run trains a preactivation Wide ResNet-28-2 from scratch
with the same 45k/5k fitting/validation split. It has 1,467,610 parameters: a
16-channel stem, four residual blocks in each 32/64/128-channel group, and a
global-average-pool classifier. The latter two groups downsample by stride 2.
Convolution weights use Kaiming-normal initialization.

The run completed 126 epochs of a 200-epoch schedule. It used SGD 0.1,
Nesterov momentum 0.9, weight decay 0.0005,
five warmup epochs then cosine decay, Cutout16 and Mixup alpha0.2 (probability0.5).
Channels-last AMP float16 was enabled. Per-epoch validation averaged logits
from the original image and a horizontal flip.
This changes several factors together and is not an isolated depth ablation.

```sh
python script/stage_3_script/script_cifar_residual.py --width 2 --output-dir result/stage_3_result/cifar_residual_w2
```

During training, each epoch saves resumable state as `latest.pt`. The source
package includes the final inference checkpoint and evaluation records, and
omits the optimizer state. Validation selected averaged weights from epochs 121
and 126, BatchNorm statistics recalibrated on fitting images, and logits
averaged across the original and flipped images. The model achieved 93.02%
validation accuracy and 92.07% test accuracy. The selected MNIST and ORL models
achieved 99.40% and 97.50% test accuracy, respectively.

To verify the included CIFAR checkpoint with the instructor dataset and a CUDA GPU:

```sh
python script/stage_3_script/script_verify_residual.py result/stage_3_result/cifar_residual_w2/evaluation_epoch126
```
