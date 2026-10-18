# Finding roads in satellite images with a U-Net

Maps go out of date as new roads are built. One way to catch missing roads is to detect
roads in fresh satellite imagery and compare them with the map. This project trains a U-Net
(ResNet-34 encoder, pretrained on ImageNet) to label every pixel of a satellite image as road
or not road. It then tests whether simple image post-processing reconnects roads that trees
and shadows break apart.

## Data

The [DeepGlobe Road Extraction dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset)
on Kaggle has 6,226 labelled 1024×1024 RGB images at 50 cm per pixel. Its validation and test
folders have no masks, so the script holds out 15% of the labelled images for validation.
The data comes under DeepGlobe's own licence: don't re-upload the dataset to GitHub.

## Run on Kaggle (recommended: free GPU)

1. Create a new Kaggle notebook. In **Add Input**, search "deepglobe road extraction" and add it.
2. Turn on a GPU accelerator and internet access in the notebook's settings. Internet is needed
   to download the pretrained encoder. Kaggle asks for phone verification before allowing either.
3. In the first cell, type `%%writefile train_roads.py` on the first line, paste the whole
   script below it, and run the cell. This saves the script inside the notebook.
4. Run:
   ```
   !pip install -q segmentation-models-pytorch
   !python train_roads.py --epochs 1          # first, time one epoch
   !python train_roads.py --epochs 15         # then the real run
   ```
5. Download `outputs/metrics.json`, `history.csv` and `predictions.png` from the output panel.

Each epoch prints its time, so you can choose how many epochs fit in your GPU quota.

## Run on your Mac (slower, Apple GPU)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python make_fake_data.py
python train_roads.py --data-dir fake_data --size 256 --epochs 2 --batch 2 --weights none --out outputs_fake
```

That checks everything works in a few minutes. For real data, download the dataset from
Kaggle (about 4 GB) and train on a subset, e.g. `--max-images 1500 --epochs 5`.

## Outputs

| File | What it holds |
| --- | --- |
| `metrics.json` | IoU, F1, precision and recall on validation, for `raw`, `closed` (gaps closed) and `closed_clean` (gaps closed, small blobs removed) |
| `history.csv` | Loss and validation IoU/F1 per epoch |
| `predictions.png` | Image, true roads, predicted roads for a few validation images |
| `best.pt` | Model weights from the best epoch (don't commit: large) |

Headline numbers are written up in [RESULTS.md](RESULTS.md): validation IoU and F1 from
`metrics.json` → `raw`, and whether morphological post-processing (`closed`) beat `raw`.
"It didn't help" is reported too.

## How it works

- **Training crops:** random 512×512 crops at full resolution, with flips and rotations.
  Shrinking whole images would make thin roads disappear.
- **Loss:** Dice + binary cross-entropy. Roads are only a few percent of pixels, and Dice keeps
  the model from predicting "no road" everywhere.
- **Validation:** full 1024×1024 images. IoU and F1 are summed over all pixels of the validation set.

## Ideas to extend

- Tune the 0.5 threshold on validation data.
- Measure road connectivity, not just pixels (the APLS metric is used for road networks).
- Run on imagery of an area you know and overlay the result on OpenStreetMap roads.

## Limitations

- Pixel IoU punishes a road predicted a few pixels off, even when the map would be fine.
- DeepGlobe covers Thailand, Indonesia and India; results may differ elsewhere.
