# Results

## Status

Code verified end to end on synthetic data (`make_fake_data.py`): the model trains, the
metrics and post-processing run, and all outputs are written. Those numbers measure nothing
about real roads and are deliberately not recorded here.

The DeepGlobe training run is pending. Once it finishes, `outputs/metrics.json` fills in:

| Variant | IoU | F1 | Precision | Recall |
| --- | --- | --- | --- | --- |
| `raw` | | | | |
| `closed` (gaps closed) | | | | |
| `closed_clean` (gaps closed, small blobs removed) | | | | |

The question the `closed` rows answer is whether morphological closing reconnects roads
broken by trees and shadows, or whether it mostly thickens false positives. Either answer
gets reported.

## Setup to record with the numbers

- Encoder, number of training images, crop size, epochs, batch size
- Train/validation split (15% of the labelled set, held out by the script)
- Decision threshold (0.5 unless tuned on validation)

## Limitations

- Pixel IoU penalises a road predicted a few pixels off, even where the resulting map would
  be fine. Connectivity metrics such as APLS measure the thing a map actually cares about.
- DeepGlobe covers Thailand, Indonesia and India. Results elsewhere may differ.
