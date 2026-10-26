import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def resolve(path, base=ROOT):
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (base / p).resolve()


def cli_options(values):
    result = []
    for key, value in values.items():
        if value is None or value is False:
            continue
        result.append("--" + key.replace("_", "-"))
        if value is not True:
            result.append(str(value))
    return result


def build_plan(args):
    config_path = resolve(args.config) if args.config else ROOT / "configs" / (args.experiment + ".json")
    config = json.loads(config_path.read_text())
    if args.seed is not None:
        config["seed"] = args.seed
    paths = {"data_root": "data", "checkpoint_root": "checkpoints", "output_root": "outputs"}
    local = ROOT / "paths.local.json"
    if local.is_file():
        paths.update(json.loads(local.read_text()))
    for key in tuple(paths):
        value = getattr(args, key, None)
        if value:
            paths[key] = value
    data_root = resolve(paths["data_root"])
    checkpoint_root = resolve(paths["checkpoint_root"])
    output_root = resolve(paths["output_root"])
    sam = resolve(args.sam_checkpoint or paths.get("sam_checkpoint") or
                  str(checkpoint_root / "pretrained/sam3.pt"))
    experiment = config["checkpoint_experiment"]
    weight_dir = checkpoint_root / experiment
    weights = {
        "tcsb": weight_dir / "tcsb/final_lora.pth",
        "tsdb": weight_dir / "tsdb/final_tsdb.pth",
        "fusion": weight_dir / "fusion/final_router.pth",
    }
    train_img = resolve(config["data"]["train_images"], data_root)
    train_ann = resolve(config["data"]["train_labels"], data_root)
    target_img = resolve(config["data"]["target_images"], data_root)
    target_ann = resolve(config["data"]["target_labels"], data_root)
    shared = {"resize": config["resize"], "device": args.device, "seed": config["seed"]}
    data_train = {"train_img_dir": train_img, "train_ann_dir": train_ann}
    data_val = {"val_img_dir": target_img, "val_ann_dir": target_ann}
    data_target = {"target_img_dir": target_img, "target_ann_dir": target_ann}
    family = config["family"]
    modules = {
        "tcsb": "method.tcsb_" + family,
        "tsdb": "method.tsdb_" + family,
        "fusion": config["fusion_module"],
    }
    a_params = {k: v for k, v in config["tcsb"].items()
                if k not in {"probe_epochs", "rank_conversion", "eval_interval"}}
    a_options = {**a_params, **shared, **data_train,
                 "checkpoint": sam, "work_dir": weight_dir / "tcsb"}
    b_options = {**config["tsdb"], **shared, **data_train, **data_val,
                 "sam_ckpt": sam, "tcsb_lora_ckpt": weights["tcsb"],
                 "work_dir": weight_dir / "tsdb"}
    fusion_options = {**config["fusion"], **shared, **data_train, **data_target,
                      "sam_ckpt": sam, "tcsb_lora_ckpt": weights["tcsb"],
                      "tsdb_ckpt": weights["tsdb"], "work_dir": weight_dir / "fusion"}
    options = {"tcsb": a_options, "tsdb": b_options, "fusion": fusion_options}
    if args.mode == "test":
        stages = ["fusion"]
        fusion_options.update(eval_only=True, resume_router_ckpt=weights["fusion"],
                              work_dir=output_root / config["name"])
        fusion_options.pop("train_img_dir")
        fusion_options.pop("train_ann_dir")
    else:
        if config.get("evaluation_only"):
            raise ValueError("This is an evaluation-only transfer; train loveda_u2r instead")
        stages = ["tcsb", "tsdb", "fusion"] if args.stage == "all" else [args.stage]
    plan = []
    for stage in stages:
        required = [sam]
        if stage != "tcsb":
            required += [target_img, target_ann]
        if args.mode == "train":
            required += [train_img, train_ann]
        if stage in ("tsdb", "fusion"):
            required.append(weights["tcsb"])
        if stage == "fusion":
            required.append(weights["tsdb"])
        if args.mode == "test":
            required.append(weights["fusion"])
        plan.append({"stage": stage, "module": modules[stage], "options": options[stage],
                     "required": required, "artifact": weights[stage]})
    return config, plan


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate ASCoR.")
    parser.add_argument("mode", choices=["train", "test"])
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--experiment", choices=[p.stem for p in sorted((ROOT / "configs").glob("*.json"))])
    choice.add_argument("--config", help="Custom JSON config; relative paths resolve from repo root")
    parser.add_argument("--stage", choices=["all", "tcsb", "tsdb", "fusion"], default="all")
    parser.add_argument("--data-root")
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--output-root")
    parser.add_argument("--sam-checkpoint")
    parser.add_argument("--sam3-source", help="Optional local SAM3 source checkout")
    parser.add_argument("--mmseg-source", help="Optional local MMSegmentation source checkout")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=None, help="Optional controlled experiment only")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config, plan = build_plan(args)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    local_path = ROOT / "paths.local.json"
    local_paths = json.loads(local_path.read_text()) if local_path.is_file() else {}
    dependency_sources = []
    for key in ("sam3_source", "mmseg_source"):
        value = getattr(args, key) or local_paths.get(key)
        if value:
            dependency_sources.append(str(resolve(value)))
    if dependency_sources:
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT), *dependency_sources,
                                             env.get("PYTHONPATH", "")])
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    if config["seed"] is not None:
        env["PYTHONHASHSEED"] = str(config["seed"])
    print(f"[{config['name']}] {args.mode}", flush=True)
    for item in plan:
        command = [sys.executable, "-u", "-m", item["module"], *cli_options(item["options"])]
        if args.dry_run:
            print(shlex.join(command), flush=True)
            continue
        print(f"[{item['stage'].upper()}] Output: {item['options']['work_dir']}", flush=True)
        missing = [str(p) for p in item["required"] if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))
        work = Path(item["options"]["work_dir"])
        if args.mode == "train" and work.exists() and any(work.iterdir()):
            raise RuntimeError(f"Refusing to overwrite existing stage: {work}")
        work.mkdir(parents=True, exist_ok=True)
        (work / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
        with (work / ("test.log" if args.mode == "test" else "train.log")).open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                status = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        if status:
            raise SystemExit(status)
        if args.mode == "train" and not item["artifact"].is_file():
            raise RuntimeError(f"Stage finished without expected artifact: {item['artifact']}")


if __name__ == "__main__":
    main()
