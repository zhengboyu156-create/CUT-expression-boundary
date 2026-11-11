"""PASB-inspired BCI HER2 four-class ConvNeXt-T downstream experiment.

Project-defined protocol: image-level fixed split, 6 image inputs, 3 seeds.
This is not a strict reproduction of PASB's unpublished training details.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support,
                             recall_score, roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import convnext_tiny

ROOT = None
EXP = None
WEIGHTS = None
SPLIT = None
BASELINE_MANIFEST = None
LABELS = [0, 1, 2, 3]
SEEDS = [1234, 2345, 3456]
METHODS = ["he", "maps_v3", "cyclegan", "cut", "m2plgan", "real_ihc"]
MAX_EPOCHS = 50
PATIENCE = 10
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2


def parse_args():
    p = argparse.ArgumentParser(description="Train/evaluate PASB-inspired BCI HER2 ConvNeXt-T classifiers.")
    p.add_argument("--split-csv", type=Path, required=True, help="CSV with id, split, label, he_path, real_ihc_path, virtual_ihc_path")
    p.add_argument("--baseline-manifest", type=Path, required=True, help="CSV with method, id, image_path for non-MAPS generated images")
    p.add_argument("--weights", type=Path, required=True, help="TorchVision ConvNeXt-Tiny ImageNet-1K state dict")
    p.add_argument("--output-dir", type=Path, required=True, help="New, empty experiment output directory")
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    return p.parse_args()


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ImageRows(Dataset):
    def __init__(self, rows, method, training):
        self.rows = rows
        self.method = method
        self.training = training
        if training:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        path = Path(row["paths"][self.method])
        with Image.open(path) as im:
            x = self.transform(im.convert("RGB"))
        return x, int(row["label"]), row["id"]


def build_rows():
    with SPLIT.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 4873 or len({r["id"] for r in rows}) != len(rows):
        raise RuntimeError(f"Unexpected split row count or duplicate ids: {len(rows)}")
    baseline = {m: {} for m in ("cyclegan", "cut", "m2plgan")}
    with BASELINE_MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            baseline[r["method"]][Path(r["id"]).stem] = r["image_path"]
    for method in baseline:
        if len(baseline[method]) != 4873:
            raise RuntimeError(f"{method}: expected 4873 manifest entries, got {len(baseline[method])}")
    for r in rows:
        sid = Path(r["id"]).stem
        r["paths"] = {
            "he": r["he_path"],
            "maps_v3": r["virtual_ihc_path"],
            "real_ihc": r["real_ihc_path"],
            "cyclegan": baseline["cyclegan"][sid],
            "cut": baseline["cut"][sid],
            "m2plgan": baseline["m2plgan"][sid],
        }
        for method, path in r["paths"].items():
            if not Path(path).is_file():
                raise FileNotFoundError(f"{method}: {path}")
    counts = {s: len([r for r in rows if r["split"] == s]) for s in ("fit", "val", "test")}
    if counts != {"fit": 3311, "val": 585, "test": 977}:
        raise RuntimeError(f"Unexpected split counts: {counts}")
    for split in ("fit", "val", "test"):
        c = {str(k): sum(int(r["label"]) == k for r in rows if r["split"] == split) for k in LABELS}
        if any(v == 0 for v in c.values()):
            raise RuntimeError(f"Missing class in {split}: {c}")
    return rows, counts


def metric(y, p):
    pred = p.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=LABELS, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_ovr_auc": float(roc_auc_score(y, p, labels=LABELS, multi_class="ovr", average="macro")),
        "macro_f1": float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y, pred, labels=LABELS, weights="quadratic")),
        "per_class": {str(c): {"precision": float(precision[c]), "recall": float(recall[c]),
                               "f1": float(f1[c]), "support": int(support[c])} for c in LABELS},
        "confusion_matrix": confusion_matrix(y, pred, labels=LABELS).tolist(),
    }


def load_pretrained():
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    model = convnext_tiny(weights=None)
    model.load_state_dict(state, strict=True)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 4)
    return model


def get_probs(model, loader, device):
    model.eval()
    probs, ys, ids = [], [], []
    with torch.inference_mode():
        for x, y, sid in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
            # Softmax in float32 avoids half-precision rows failing sklearn's
            # multiclass probability-sum validation for macro OvR AUC.
            probs.append(logits.float().softmax(1).cpu().numpy())
            ys.extend(y.numpy().tolist())
            ids.extend(list(sid))
    return np.concatenate(probs), np.asarray(ys), ids


def train_one(method, seed, rows, device, run_root):
    out = run_root / method / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=False)
    fit = [r for r in rows if r["split"] == "fit"]
    val = [r for r in rows if r["split"] == "val"]
    test = [r for r in rows if r["split"] == "test"]
    seed_all(seed)
    train_ds = ImageRows(fit, method, True)
    eval_fit = ImageRows(fit, method, False)
    val_ds = ImageRows(val, method, False)
    test_ds = ImageRows(test, method, False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                          pin_memory=True, persistent_workers=True)
    fit_dl = DataLoader(eval_fit, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS,
                        pin_memory=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS,
                        pin_memory=True, persistent_workers=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS,
                         pin_memory=True, persistent_workers=True)
    counts = np.bincount([int(r["label"]) for r in fit], minlength=4)
    weights = torch.tensor(counts, device=device, dtype=torch.float32).rsqrt()
    weights = weights / weights.mean()
    model = load_pretrained().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=weights)
    cfg = {
        "method": method, "seed": seed, "input": method,
        "split_csv": str(SPLIT), "n_fit": len(fit), "n_val": len(val), "n_test": len(test),
        "class_counts_fit": counts.tolist(), "labels": {"0": 0, "1": 1, "2": 2, "3": 3},
        "pretrained_weights": str(WEIGHTS),
        "pretrained_sha256": hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
        "architecture": "torchvision convnext_tiny; full fine-tuning; 4-class head",
        "preprocessing": "RGB; resize short edge to 256; train random 224 crop + H/V flips; eval center crop 224; ImageNet mean/std",
        "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY,
        "loss": "inverse_sqrt_fit_class_frequency weighted cross entropy",
        "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "amp": "CUDA float16 autocast", "best_rule": "highest validation macro-F1; ties keep earlier epoch",
        "seeds": SEEDS, "device": str(device),
        "generator_provenance": "MAPS-GAN uses existing V3 prediction path; CycleGAN/CUT/M2PL-GAN use canonical_retry1 predictions",
    }
    atomic_json(out / "train_config.json", cfg)
    best, best_epoch, stale = -1.0, 0, 0
    log = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total, n = 0.0, 0
        for x, y, _ in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(y)
            n += len(y)
        vp, vy, _ = get_probs(model, val_dl, device)
        vm = metric(vy, vp)
        record = {"epoch": epoch, "fit_loss": total / n, "val_accuracy": vm["accuracy"],
                  "val_macro_f1": vm["macro_f1"], "val_macro_ovr_auc": vm["macro_ovr_auc"]}
        log.append(record)
        with (out / "training_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        if vm["macro_f1"] > best + 1e-12:
            best, best_epoch, stale = vm["macro_f1"], epoch, 0
            torch.save({"state_dict": model.state_dict(), "seed": seed, "epoch": epoch}, out / "best_model.pt")
        else:
            stale += 1
        atomic_json(run_root / "run_status.json", {
            "status": "RUNNING", "method": method, "seed": seed, "epoch": epoch,
            "best_epoch": best_epoch, "completed_runs": len(list(run_root.glob("*/seed_*/metrics.json"))),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
        if stale >= PATIENCE:
            break
    ck = torch.load(out / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(ck["state_dict"])
    p, y, ids = get_probs(model, test_dl, device)
    metrics = metric(y, p)
    metrics.update({"method": method, "seed": seed, "best_epoch": best_epoch, "epochs_run": len(log)})
    with (out / "test_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label", "p0", "p1", "p2", "p3", "pred"])
        for sid, yi, pi in zip(ids, y, p):
            writer.writerow([sid, int(yi), *map(float, pi), int(pi.argmax())])
    atomic_json(out / "metrics.json", metrics)
    for dl in (train_dl, fit_dl, val_dl, test_dl):
        if getattr(dl, "_iterator", None) is not None:
            dl._iterator._shutdown_workers()
    del model, opt, train_dl, fit_dl, val_dl, test_dl
    torch.cuda.empty_cache()
    return metrics


def main():
    global ROOT, EXP, WEIGHTS, SPLIT, BASELINE_MANIFEST
    global METHODS, SEEDS, MAX_EPOCHS, PATIENCE, BATCH_SIZE, LR, WEIGHT_DECAY, NUM_WORKERS
    args = parse_args()
    SPLIT = args.split_csv.resolve()
    ROOT = SPLIT.parent
    BASELINE_MANIFEST = args.baseline_manifest.resolve()
    WEIGHTS = args.weights.resolve()
    EXP = args.output_dir.resolve()
    METHODS, SEEDS = args.methods, args.seeds
    MAX_EPOCHS, PATIENCE = args.max_epochs, args.patience
    BATCH_SIZE, LR, WEIGHT_DECAY, NUM_WORKERS = args.batch_size, args.lr, args.weight_decay, args.num_workers
    if (EXP / "experiment_manifest.json").exists() or (EXP / "results").exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment results: {EXP}")
    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)
    run_root = EXP / "results"
    run_root.mkdir(parents=True)
    rows, split_counts = build_rows()
    # Small model/weight compatibility check before the long training job.
    model = load_pretrained()
    del model
    torch.cuda.empty_cache()
    cfg = {
        "status": "STARTING", "split_counts": split_counts, "methods": METHODS, "seeds": SEEDS,
        "protocol": "PASB-inspired project experiment; not a strict paper reproduction",
        "input_groups": {
            "he": "real H&E", "maps_v3": "existing MAPS-GAN V3 virtual IHC",
            "cyclegan": "CycleGAN virtual IHC", "cut": "CUT virtual IHC",
            "m2plgan": "M2PL-GAN virtual IHC", "real_ihc": "real HER2 IHC reference",
        },
        "limits": "Image-level split; case-level independence is unverified. MAPS V3 generator was trained on the full 3896-pair training partition; classifier validation IDs are within that partition.",
        "source_split_sha256": hashlib.sha256(SPLIT.read_bytes()).hexdigest(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_json(EXP / "experiment_manifest.json", cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    results = {}
    atomic_json(run_root / "run_status.json", {"status": "RUNNING", "methods": METHODS, "seeds": SEEDS,
                                                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    for method in METHODS:
        results[method] = []
        for seed in SEEDS:
            results[method].append(train_one(method, seed, rows, device, run_root))
        for key in ("accuracy", "macro_ovr_auc", "macro_f1", "quadratic_weighted_kappa"):
            values = [r[key] for r in results[method]]
            results[f"{method}_mean_{key}"] = float(np.mean(values))
            results[f"{method}_std_{key}"] = float(np.std(values, ddof=1))
        atomic_json(run_root / "classifier_summary.json", results)
    atomic_json(run_root / "run_status.json", {"status": "COMPLETE", "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                                "summary": str(run_root / "classifier_summary.json")})
    atomic_json(EXP / "experiment_manifest.json", {**cfg, "status": "COMPLETE", "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        exp = EXP
        if exp is not None and exp.exists():
            atomic_json(exp / "results" / "run_status.json", {"status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc(),
                                                               "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        raise
