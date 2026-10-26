import sys


def progress(*args, **kwargs):
    from tqdm import tqdm

    kwargs.setdefault("disable", not sys.stderr.isatty())
    return tqdm(*args, **kwargs)


def log_epoch(stage, epoch, epochs, metrics):
    keys = ("loss_total", "loss_ce", "loss_xded", "loss_cons") if stage == "Fusion" else ("loss_total", "loss_rescue")
    values = " ".join(
        f"{key}={float(metrics[key]):.4f}" for key in keys if key in metrics
    )
    print(f"[{stage}] Epoch {epoch}/{epochs} | {values}", flush=True)


def log_fusion_evaluation(label, result):
    print(f"[{label}] mIoU={100.0 * float(result['fusion']['mIoU']):.2f}%",
          flush=True)
    print("Fusion IoU (%):", {
        name: round(100.0 * float(value), 2)
        for name, value in result["fusion"]["per_class_iou"].items()
    }, flush=True)
