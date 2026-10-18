"""Train a U-Net to find roads in satellite images (DeepGlobe Road Extraction).

Kaggle (recommended, free GPU):
    !pip install -q segmentation-models-pytorch
    !python train_roads.py --epochs 15
Mac (Apple GPU via MPS, slower):
    python train_roads.py --data-dir ~/data/deepglobe --epochs 5 --max-images 1500
Quick check that everything runs (no download, a few minutes on CPU):
    python make_fake_data.py && python train_roads.py --data-dir fake_data --size 256 \
        --epochs 2 --batch 2 --weights none --out outputs_fake

Writes to --out: best.pt (model), history.csv (per epoch), metrics.json (final scores
with and without post-processing), predictions.png (examples), split.csv.
"""
import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # ImageNet statistics
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=None, help="folder containing *_sat.jpg and *_mask.png")
    p.add_argument("--out", default="outputs")
    p.add_argument("--size", type=int, default=512, help="training crop size (pixels)")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weights", default="imagenet", choices=["imagenet", "none"])
    p.add_argument("--max-images", type=int, default=0, help="use only this many images (0 = all)")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def find_pairs(data_dir):
    """Every *_sat.jpg that has a matching *_mask.png (DeepGlobe's val/test have no masks)."""
    if data_dir is None:
        roots = [Path("/kaggle/input"), Path("data/deepglobe")]
        data_dir = next((r for r in roots if r.exists()), None)
        if data_dir is None:
            raise SystemExit("No data found. Pass --data-dir pointing at the DeepGlobe train folder.")
    pairs = []
    for sat in sorted(Path(data_dir).expanduser().rglob("*_sat.jpg")):
        mask = sat.with_name(sat.name.replace("_sat.jpg", "_mask.png"))
        if mask.exists():
            pairs.append((sat, mask))
    if not pairs:
        raise SystemExit(f"No *_sat.jpg / *_mask.png pairs under {data_dir}")
    return pairs


class Roads(Dataset):
    def __init__(self, pairs, size, train):
        self.pairs, self.size, self.train = pairs, size, train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        sat, mask = self.pairs[i]
        img = cv2.cvtColor(cv2.imread(str(sat)), cv2.COLOR_BGR2RGB)
        m = (cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
        if self.train:
            # random crop keeps the full 50 cm/pixel detail (thin roads vanish if we shrink)
            h, w = m.shape
            y, x = random.randint(0, h - self.size), random.randint(0, w - self.size)
            img, m = img[y:y + self.size, x:x + self.size], m[y:y + self.size, x:x + self.size]
            if random.random() < 0.5:
                img, m = img[:, ::-1], m[:, ::-1]
            if random.random() < 0.5:
                img, m = img[::-1], m[::-1]
            k = random.randint(0, 3)
            img, m = np.rot90(img, k), np.rot90(m, k)
        img = (img.astype(np.float32) / 255 - MEAN) / STD
        return (torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(m))[None])


def postprocess(binary, close_px=7, min_area=300):
    """Close small gaps (roads broken by trees/shadows), then drop tiny blobs."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    closed = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return closed.astype(bool), keep[lab]


def scores(tp, fp, fn):
    return {"iou": tp / max(tp + fp + fn, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}


@torch.no_grad()
def evaluate(model, loader, device, full=False):
    """Pixel counts summed over the whole validation set (full images, native resolution)."""
    model.eval()
    c = {k: np.zeros(3) for k in ("raw", "closed", "closed_clean")}
    for x, y in loader:
        prob = torch.sigmoid(model(x.to(device))).float().cpu().numpy()[:, 0]
        gt = y.numpy()[:, 0].astype(bool)
        for p, g in zip(prob > 0.5, gt):
            variants = {"raw": p}
            if full:
                variants["closed"], variants["closed_clean"] = postprocess(p)
            for k, v in variants.items():
                c[k] += [(v & g).sum(), (v & ~g).sum(), (~v & g).sum()]
    model.train()
    return {k: scores(*v) for k, v in c.items() if v.sum() > 0}


def save_examples(model, ds, device, path, n=4):
    model.eval()
    rows = []
    with torch.no_grad():
        for i in range(min(n, len(ds))):
            x, y = ds[i]
            p = torch.sigmoid(model(x[None].to(device))).float().cpu().numpy()[0, 0] > 0.5
            img = ((x.numpy().transpose(1, 2, 0) * STD + MEAN) * 255).clip(0, 255).astype(np.uint8)
            to3 = lambda a: np.repeat((a * 255).astype(np.uint8)[..., None], 3, axis=2)
            rows.append(np.hstack([img, to3(y.numpy()[0]), to3(p)]))
    grid = cv2.resize(np.vstack(rows), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))   # image | truth | prediction


def main():
    a = parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(a.data_dir)
    random.shuffle(pairs)
    if a.max_images:
        pairs = pairs[:a.max_images]
    n_val = max(1, int(len(pairs) * a.val_frac))
    val, train = pairs[:n_val], pairs[n_val:]
    pd.DataFrame([(str(s), "val") for s, _ in val] + [(str(s), "train") for s, _ in train],
                 columns=["image", "split"]).to_csv(out / "split.csv", index=False)
    print(f"Device {device} | {len(train)} training images, {len(val)} validation images")

    tl = DataLoader(Roads(train, a.size, True), batch_size=a.batch, shuffle=True,
                    num_workers=a.workers, drop_last=True, pin_memory=device == "cuda")
    vl = DataLoader(Roads(val, a.size, False), batch_size=2, num_workers=a.workers)

    model = smp.Unet("resnet34", encoder_weights=None if a.weights == "none" else a.weights,
                     in_channels=3, classes=1).to(device)
    dice, bce = smp.losses.DiceLoss("binary"), torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * len(tl))
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, history = -1.0, []
    for epoch in range(1, a.epochs + 1):
        t0, losses = time.time(), []
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(x)
                loss = dice(logits, y) + bce(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            losses.append(loss.item())
        m = evaluate(model, vl, device)["raw"]
        history.append({"epoch": epoch, "loss": np.mean(losses), "val_iou": m["iou"],
                        "val_f1": m["f1"], "minutes": (time.time() - t0) / 60})
        print(f"epoch {epoch:2d}  loss {np.mean(losses):.3f}  val IoU {m['iou']:.3f}  "
              f"F1 {m['f1']:.3f}  ({history[-1]['minutes']:.1f} min)", flush=True)
        if m["iou"] > best:
            best = m["iou"]
            torch.save(model.state_dict(), out / "best.pt")
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    final = evaluate(model, vl, device, full=True)
    best_epoch = int(pd.DataFrame(history)["val_iou"].idxmax()) + 1
    json.dump({"best_epoch": best_epoch, "n_train": len(train), "n_val": len(val),
               "crop_size": a.size, "epochs": a.epochs, "encoder_weights": a.weights,
               **{k: {m: round(float(v), 4) for m, v in d.items()} for k, d in final.items()}},
              open(out / "metrics.json", "w"), indent=2)
    print("\nValidation (best epoch", best_epoch, ")")
    for k, d in final.items():
        print(f"  {k:13s} IoU {d['iou']:.3f}  F1 {d['f1']:.3f}  "
              f"precision {d['precision']:.3f}  recall {d['recall']:.3f}")
    save_examples(model, Roads(val, a.size, False), device, out / "predictions.png")
    print(f"Saved {out}/metrics.json, history.csv, predictions.png, best.pt")


if __name__ == "__main__":
    main()
