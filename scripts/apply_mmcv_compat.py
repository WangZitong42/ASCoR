import argparse
from pathlib import Path


def patch_file(path, variable):
    text = path.read_text()
    old = f"{variable} = '2.2.0'"
    new = f"{variable} = '2.3.0'"
    if new in text and old not in text:
        print(f'Already applied: {path}')
        return
    if text.count(old) != 1:
        raise ValueError(f'Unexpected version-bound definition in {path}')
    path.write_text(text.replace(old, new))
    print(f'Patched: {path}')


def main():
    parser = argparse.ArgumentParser(description="Allow MMCV 2.2.0 in the pinned OpenMMLab dependencies.")
    parser.add_argument('--mmdet-root', type=Path, required=True)
    parser.add_argument('--mmseg-root', type=Path, required=True)
    args = parser.parse_args()
    patch_file(args.mmdet_root / 'mmdet/__init__.py', 'mmcv_maximum_version')
    patch_file(args.mmseg_root / 'mmseg/__init__.py', 'MMCV_MAX')


if __name__ == '__main__':
    main()
