# MVAM — Multi-View Attention Model for Steering Angle Prediction

MVAM is a small deep learning project that predicts a car's steering angle from three dashboard camera images (center, left, right). It combines a shared ResMLP image backbone with a multi-head attention module that fuses the three views, followed by an MLP that outputs the final steering value. The project was built and trained on the classic Udacity self-driving-car simulator data.

## What this project does

Given three images taken at the same moment from a center, a left, and a right-facing camera on a simulated car, the model predicts a single number: the steering angle. This setup mimics how a human driver uses peripheral vision together with what is directly ahead. Instead of just averaging the three views, the model learns an attention pattern that decides how much weight to give each camera when making a prediction.

## Repository contents

| File | Purpose |
|---|---|
| `multiview-streeing-resmlp.ipynb` | Main notebook: downloads the dataset, builds the dataset/dataloader pipeline, defines the model, and runs training. |
| `data_preprocessing.py` | Standalone version of the data pipeline (steering angle analysis, normalization, balanced sampling, `Dataset`/`DataLoader` classes). |
| `evaluation_results.json` | Final metrics computed on the validation set. |
| `Data-analysis.png` | Steering angle distribution, box plot, and cumulative distribution. |
| `Results.png` | Full grid of evaluation plots (predictions vs. targets, residuals, error distribution, attention matrices, etc.). |
| `Predictions-vs-true-values.png` | Time-series view of predicted vs. true steering angles on a sample sequence. |

## Data

The data comes from the `training-car` dataset on Kaggle (based on the Udacity self-driving-car simulator), loaded through `kagglehub`. Each row of `driving_log.csv` has paths to a center, left, and right image plus the recorded steering angle, throttle, brake, and speed at that moment.

A quick look at the steering angle distribution (see `Data-analysis.png`) shows that most of the driving is close to a straight line: the great majority of steering values sit near zero, with sharper turns much rarer. To stop the model from just learning to predict "go straight" all the time, the pipeline:

- normalizes the steering angle (min-max, z-score, or tanh, selectable),
- builds a **weighted sampler** that gives rarer, larger steering angles more chances to appear during training,
- applies light image augmentation during training only (color jitter, random horizontal flip, small Gaussian noise).

## Model architecture

1. **Backbone** — a `resmlp_12_224` model (from the `timm` library, ImageNet pretrained) is shared across all three camera views, so the same weights process the center, left, and right image. The first 60% of the backbone's layers are frozen; the rest are fine-tuned.
2. **Attention fusion** — the three per-view feature vectors are stacked and passed through a multi-head scaled dot-product attention block. This lets the model learn, for example, that the left/right cameras matter more during a turn than during straight driving. The three attended outputs are averaged into a single fused feature vector.
3. **Regressor head** — a small MLP with a bottleneck shape (512 → 256 → 128 → 1), using ReLU, dropout, and batch normalization, maps the fused feature to a single steering angle prediction.
4. **Loss** — a custom loss combining standard MSE with a smoothness term (the mean absolute magnitude of the prediction), which discourages large, jerky steering outputs.

Training uses AdamW, a cosine annealing learning rate schedule, and gradient clipping, for up to 50 epochs, saving the checkpoint with the best validation loss.

## Results

Evaluated on a held-out validation split of 1,608 samples (1,015 straight-driving, 593 turning: 300 left / 293 right):

| Metric | Value |
|---|---|
| MSE | 0.0100 |
| MAE | 0.0647 |
| RMSE | 0.0999 |
| R² | 0.365 |
| Directional accuracy | 36.1% |
| Within 0.05 rad (≈2.9°) | 62.7% |
| Within 0.10 rad (≈5.7°) | 78.5% |
| Within 0.15 rad (≈8.6°) | 89.0% |
| Within 0.20 rad (≈11.5°) | 93.8% |
| MAE, straight driving | 0.0335 |
| MAE, turns | 0.1183 |

The model tracks the general steering pattern reasonably well (see `Predictions-vs-true-values.png`, correlation ≈ 0.63) but tends to smooth out sharp turns: the standard deviation of its predictions (0.067) is noticeably smaller than that of the true angles (0.125). It reacts a bit slowly to sudden steering changes, which shows up as visible lag around a few peaks in the time-series plot.

The attention matrix (bottom-left panel of `Results.png`) shows an interesting pattern: all three views end up paying the most attention to the **right** camera (attention weight around 0.57–0.65), more than to themselves or to the left camera. This asymmetry was not something we designed on purpose, and it is worth digging into further — it might reflect something about how the simulator track is laid out, or a quirk of how the fusion layer converged during training.

The low directional accuracy (36%) looks worse than it is: since most steering angles are extremely close to zero, a tiny difference in sign between a true angle of, say, +0.001 and a prediction of ‑0.001 counts as a directional "miss" even though both are effectively "drive straight." Looking at the angular accuracy numbers above gives a fairer picture of how close the predictions really are.

## Limitations

- The model is noticeably more conservative than the ground truth during sharp turns, which is the main thing to improve next (a stronger smoothness penalty, more turn examples, or a small temporal model such as an LSTM/GRU over consecutive frames could help).
- Directional accuracy as defined here is misleading for a steering angle distribution this concentrated near zero; a threshold-based or magnitude-weighted variant would be a better metric.
- The dataset is a driving simulator, so results are not directly evidence of real-road performance.

## Getting started

```bash
pip install torch torchvision timm pandas scikit-learn pillow matplotlib seaborn kagglehub tqdm
```

Then open `multiview-streeing-resmlp.ipynb` and run the cells in order. The first cell downloads the dataset via `kagglehub`; the rest builds the data pipeline, model, and training loop. `data_preprocessing.py` can also be imported on its own if you just need the dataset/dataloader logic.

## Acknowledgements

- Dataset: `roydatascience/training-car` on Kaggle, based on the Udacity self-driving-car simulator.
- Backbone model: ResMLP (Touvron et al., 2021), used via the `timm` library (Wightman, 2019).
- [Mohammad Hossein Shamsipoor](https://github.com/Mound21k) — **co-developed this project**

## Citation

If you want to reference this project, you can use:

```bibtex
@misc{shamsipour_khaleghi_2025_mvam,
  author       = {Shamsipour, MohammadHossein and Khaleghi, Aida},
  title        = {MVAM: Multi-View Attention Model for Steering Angle Prediction},
  year         = {2025},
  howpublished = {GitHub repository},
  note         = {Authors' emails: mohammadhossein.shamsipoor@gmail.com, aiida.khaleghi@gmail.com}
}
```
