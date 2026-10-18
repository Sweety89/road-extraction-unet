"""Make a few fake 'satellite' images in DeepGlobe's file layout, so train_roads.py can be
tested without downloading anything. Roads are grey lines, partly hidden by 'trees'.

    python make_fake_data.py          # writes fake_data/train/*_sat.jpg and *_mask.png
"""
from pathlib import Path
import cv2
import numpy as np

S, N = 1024, 16
out = Path("fake_data/train")
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
for i in range(N):
    img = cv2.GaussianBlur((rng.normal(0.45, 0.12, (S, S, 3)) * 255).clip(0, 255).astype(np.uint8), (9, 9), 0)
    mask = np.zeros((S, S), np.uint8)
    for _ in range(rng.integers(3, 7)):
        p1, p2 = tuple(map(int, rng.integers(0, S, 2))), tuple(map(int, rng.integers(0, S, 2)))
        w = int(rng.integers(8, 18))
        cv2.line(mask, p1, p2, 255, w)
        cv2.line(img, p1, p2, (150, 150, 150), w)
    for _ in range(12):                                     # trees hiding parts of roads
        cv2.circle(img, tuple(map(int, rng.integers(0, S, 2))), int(rng.integers(10, 22)), (60, 90, 60), -1)
    cv2.imwrite(str(out / f"{1000 + i}_sat.jpg"), img)
    cv2.imwrite(str(out / f"{1000 + i}_mask.png"), mask)
print(f"Wrote {N} fake image/mask pairs to {out}")
