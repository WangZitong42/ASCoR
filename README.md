# ASCoR

**Asymmetric Semantic--Spatial Learning for Remote Sensing Domain Generalization**

Source-only semantic segmentation with three training stages: TCSB (probe-once
LoRA rank allocation and SVD conversion), TSDB (spatial adaptation), and a
fusion router trained with frozen experts.

## Installation

Linux, Python 3.12, PyTorch 2.7.0 / CUDA 12.6. Fresh-environment installation
and GPU training, saving and reloading passed for both dataset families
using synthetic two-image datasets and one epoch per training stage.

Run from this repository's root. Building MMCV requires a C++ compiler, CUDA
toolkit and nvcc in addition to the PyTorch CUDA runtime.

```bash
conda create -n ascor python=3.12.12 pip -y
conda activate ascor
python -m pip install pip==26.0.1 setuptools==69.5.1 wheel==0.46.3 packaging==25.0
python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt

mkdir -p third_party
git clone https://github.com/open-mmlab/mmcv.git third_party/mmcv
git -C third_party/mmcv checkout a8073c74bf83d62ec36a103f835faa4837fb6585
MMCV_WITH_OPS=1 MAX_JOBS=4 python -m pip install -v \
  --no-build-isolation --no-deps ./third_party/mmcv

git clone --branch v3.3.0 https://github.com/open-mmlab/mmdetection.git \
  third_party/mmdetection
git clone https://github.com/open-mmlab/mmsegmentation.git \
  third_party/mmsegmentation
git -C third_party/mmsegmentation checkout b040e147adfa027bbc071b624bedf0ae84dfc922
python scripts/apply_mmcv_compat.py \
  --mmdet-root third_party/mmdetection \
  --mmseg-root third_party/mmsegmentation
python -m pip install --no-build-isolation --no-deps -e third_party/mmdetection
python -m pip install --no-build-isolation --no-deps -e third_party/mmsegmentation

git clone https://github.com/facebookresearch/sam3.git third_party/sam3
git -C third_party/sam3 checkout 86ed77094094e5cabb16b0414ec60c5ba9ce0a0f
git -C third_party/sam3 apply --check ../../patches/sam3_semantic_output.patch
git -C third_party/sam3 apply ../../patches/sam3_semantic_output.patch
python -m pip install --no-build-isolation --no-deps -e third_party/sam3
python -m pip check
python -c "import mmcv.ops, mmdet, mmseg; from sam3.model_builder import build_sam3_image_model; print('Imports passed')"
```

SAM3 uses official commit `86ed77094094e5cabb16b0414ec60c5ba9ce0a0f`.
The small patch exposes semantic output as `state["semantic_mask_logits"]`,
which the class-prompt branch consumes, while preserving its autograd graph.

The compatibility script changes only the MMCV upper bounds in
`mmdet/__init__.py` (`mmcv_maximum_version`) and `mmseg/__init__.py` (`MMCV_MAX`)
from 2.2.0 to 2.3.0, allowing MMCV 2.2.0. It does not change model code.
MMEngine is pinned to 0.10.7. OpenCV is pinned to 4.9.0.80 with NumPy 1.26.4;
install only one OpenCV distribution. Linux may need GL/GLib system libraries
for `cv2`. Build MMCV with the target GPU visible; a build without a visible GPU
requires explicit `FORCE_CUDA` and `TORCH_CUDA_ARCH_LIST` settings.

## Data

Place prepared datasets in the included `data/` directories. The `.gitkeep`
files preserve the folder layout; no images or labels are included. To use
a different location, copy the path template and set `data_root`:

```bash
cp paths.example.json paths.local.json
```

Required layout under `data_root`:

```text
LoveDA/
  Train/{Rural,Urban}/{images_png,masks_png}/
  Val/{Rural,Urban}/{images_png,masks_png}/
isprs_dg/
  potsdam_irrg/{img_dir,ann_dir}/{train,val}/
  potsdam_rgb/{img_dir,ann_dir}/{train,val}/
  vaihingen_irrg_cross_344/{img_dir,ann_dir}/train/
  vaihingen_irrg_cross_398/{img_dir,ann_dir}/val/
openearthmap_crossearth384_loveda7_pngview/
  img_dir/val/*.png
  ann_dir/val/*.png
```

Braces indicate alternative directories. Images and masks must be PNG files
with identical filenames. Potsdam RGB training uses the corresponding Potsdam
IRRG training masks, as specified in the configs. Paths are relative to
`data_root`; the dataset directory can be on any disk.

| Domain | Train pairs | Evaluation pairs |
| --- | ---: | ---: |
| Potsdam IRRG | 3456 | 2016 |
| Potsdam RGB | 3456 | 2016 |
| Vaihingen IRRG | 344 | 398 |
| LoveDA Rural | 1366 | 992 |
| LoveDA Urban | 1156 | 677 |
| OEM matched subset | — | 384 |

All network inputs are resized to 512. ISPRS prepared crops are 512 x 512;
LoveDA/OEM use the prepared 1024 x 1024 image/crop protocol before resizing.

**Masks:** ISPRS uses IDs 0..5 (impervious surface, building, low vegetation,
tree, car, clutter/background), with 255 ignored. LoveDA raw IDs 1..7 are
background, building, road, water, barren, forest and agricultural; raw 0 is
ignored and the reader maps 1..7 to train IDs 0..6.

For OEM, prepare the matched 384-pair subset and map original IDs to LoveDA
raw IDs: `0→0, 1→5, 2→1, 3→1, 4→3, 5→6, 6→4, 7→7, 8→2`.
The OEM config expects these LoveDA raw IDs. If masks have already been converted
to train IDs 0..6 with ignore 255, set `target_mask_encoding` to `train_ids`.
The `_pngview` suffix is a directory name for prepared PNG data.

## Weights and paths

Trained checkpoints will be published as GitHub Release assets; download
links will be added here when available. Extract each experiment into the
included `checkpoints/` directories. Obtain the SAM3 base checkpoint
separately from [SAM3](https://github.com/facebookresearch/sam3):

```text
checkpoints/
  pretrained/sam3.pt
  <experiment>/
    tcsb/final_lora.pth
    tsdb/final_tsdb.pth
    fusion/final_router.pth
```

In `paths.local.json`, set `data_root`, `checkpoint_root`, `output_root`, and
`sam_checkpoint`. The default source paths are `third_party/sam3` and
`third_party/mmsegmentation`. CLI path arguments override this local file.
Project paths resolve from the repository root, independent of the shell's
working directory. Data and weights are not included in the code archive.

## Training and evaluation

```bash
python run.py train --experiment loveda_r2u --gpu 0

python run.py test --experiment loveda_r2u --gpu 0

python run.py train --experiment loveda_r2u --dry-run

python run.py train --experiment loveda_r2u --stage tcsb --gpu 0

python run.py test --experiment loveda_u2oem --gpu 0
```

| Experiment | Source → target | TCSB epochs | TSDB epochs | Fusion epochs |
| --- | --- | ---: | ---: | ---: |
| loveda_r2u | Rural → Urban | 5 | 5 | 3 |
| loveda_u2r | Urban → Rural | 5 | 5 | 3 |
| isprs_pi2pr | Potsdam IRRG → Potsdam RGB | 5 | 5 | 3 |
| isprs_pi2vi | Potsdam IRRG → Vaihingen IRRG | 5 | 5 | 3 |
| isprs_pr2pi | Potsdam RGB → Potsdam IRRG | 5 | 5 | 3 |
| isprs_pr2vi | Potsdam RGB → Vaihingen IRRG | 5 | 5 | 3 |
| isprs_vi2pi | Vaihingen IRRG → Potsdam IRRG | 5 | 5 | 3 |
| isprs_vi2pr | Vaihingen IRRG → Potsdam RGB | 5 | 15 | 3 |

TCSB adds one source-domain probe epoch at rank 128, allocates a total rank
budget of 512 over blocks 3,7,11,15,19,23,27,31 (bounds 4..256), and performs
one SVD conversion before final-stage training. The allocation stays fixed.
TSDB uses RA on blocks 0..23 and FDRA on 24..31. The router uses fused CE,
selective XDED and appearance consistency, with TSDB weight bounded by 0.2.

`class_weights` are fixed loss weights in dataset class order. TCSB uses
unit weights for pixel-wise cross-entropy. TSDB uses the configured values
for Mask2Former query classification; mask and Dice losses keep their own
coefficients. Evaluation metrics are unweighted.

Training saves final-epoch weights and configured interval snapshots. Target
metrics are for monitoring; evaluation uses a single resized image without TTA.
Logs show rank allocation, epoch losses and saved paths; only final fusion
prints mIoU and per-class IoU. Seeds are unset by default; use `--seed N` when
needed. Training refuses to overwrite a nonempty stage output directory.

## Acknowledgements and license

Built with [SAM3](https://github.com/facebookresearch/sam3),
[MMCV](https://github.com/open-mmlab/mmcv),
[MMEngine](https://github.com/open-mmlab/mmengine),
[MMDetection](https://github.com/open-mmlab/mmdetection) and
[MMSegmentation](https://github.com/open-mmlab/mmsegmentation).

SAM3 and the OpenMMLab dependencies retain their upstream licenses. Dependency
source and model weights are downloaded separately and are not bundled here.
The license for the ASCoR project code and the paper citation will be added
when confirmed.
