# ASCoR

Source-only domain generalization for remote sensing semantic segmentation.

## Installation

Requires Linux, an NVIDIA GPU, a CUDA toolkit with `nvcc`, a C++ compiler, Git and Conda.

```bash
git clone https://github.com/WangZitong42/ASCoR.git
cd ASCoR

conda create -n ascor python=3.12.12 pip -y
conda activate ascor
python -m pip install pip==26.0.1 setuptools==69.5.1 wheel==0.46.3 packaging==25.0
python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt

mkdir -p third_party
git clone https://github.com/open-mmlab/mmcv.git third_party/mmcv
git -C third_party/mmcv checkout a8073c74bf83d62ec36a103f835faa4837fb6585
MMCV_WITH_OPS=1 MAX_JOBS=2 python -m pip install -v \
  --no-build-isolation --no-deps ./third_party/mmcv

git clone --branch v3.3.0 https://github.com/open-mmlab/mmdetection.git third_party/mmdetection
git clone https://github.com/open-mmlab/mmsegmentation.git third_party/mmsegmentation
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

## Data preparation

Download and prepare the datasets separately, then place them under `data/`:

```text
data/
  LoveDA/
    Train/{Rural,Urban}/{images_png,masks_png}/
    Val/{Rural,Urban}/{images_png,masks_png}/
  isprs_dg/
    potsdam_irrg/{img_dir,ann_dir}/{train,val}/
    potsdam_rgb/{img_dir,ann_dir}/{train,val}/
    vaihingen_irrg_cross_344/{img_dir,ann_dir}/train/
    vaihingen_irrg_cross_398/{img_dir,ann_dir}/val/
  OpenEarthMap/
    img_dir/val/
    ann_dir/val/
```

Braces list the required subdirectories. Use PNG images and masks with matching filenames. Potsdam RGB training uses the corresponding Potsdam IRRG training masks, as specified in the configs.

- **LoveDA:** keep raw mask IDs 0..7. ID 0 is ignored; IDs 1..7 are mapped to training IDs 0..6 by the loader.
- **ISPRS:** use mask IDs 0..5, with 255 ignored.
- **OpenEarthMap:** use the prepared 384-pair subset. Map original mask IDs to LoveDA raw IDs: `0→0, 1→5, 2→1, 3→1, 4→3, 5→6, 6→4, 7→7, 8→2`.

To use a different data location, run:

```bash
cp paths.example.json paths.local.json
```

Set `data_root` in `paths.local.json` to your dataset directory, keeping the same folder layout.
