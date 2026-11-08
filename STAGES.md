# Dataset-usage detection: stages

Marks a dataset with an owner-specific invisible pattern, then checks whether a suspect image generator
reproduces that pattern. Reimplemented from the DiffusionShield paper (Cui et al.); no code reused.

| Stage | Script | What it achieves |
|---|---|---|
| 0. Timing | `timing_test.py` | Measures what this machine can actually run |
| 1. Engine + E1 | `stage1_engine.py` | Trains the faint patches and the small decoder once, then checks the mark is invisible and survives JPEG and resizing |
| 2. Marking | `stage2_mark.py` | Turns an owner's images into marked PNGs carrying their secret message, and logs it in a tamper-evident registry |
| 3. Detector | `stage3_detect.py` | Reads the mark from images and gives a verdict from two statistical tests |
| 4. Audit | `stage4_audit.py` | Runs a suspect from a crawler record to an evidence report with no manual steps; anything untestable is logged "Not testable", never "clean" |
| 5. Transfer (E2/E3) | `stage5_transfer.py` | The decisive test: fine-tune one generator on clean images and one on marked images, and see whether the mark appears in the second's outputs |
| Owner scan | `exp_owner_scan.py` | Scores saved outputs against thousands of other owners, to check decoder bias and false accusations |
| E7 | `exp_e7_reuse.py` | Checks that one trained engine serves different owners and image domains |

## What runs where (measured on an 8-core Ryzen 7730U, no GPU, 9.9 GB RAM)

| Experiment | 256 px (design) | 32 px (reduced) |
|---|---|---|
| One fine-tune, assuming 2,000 steps | 70 h | 3.2 h |
| 500 generated images | 17 h | 0.5 h |
| E2 | 7 days | 7.4 h |
| E3, five marked shares | 18 days | 19 h |
| E4, 30 clean fine-tunes | 109 days | 4.7 days |
| Peak RAM | 9.0 GB | 3.5 GB |

The 256 px model needs a GPU (about 3 h for E2 on a Colab T4), and is on hold for lack of time. The notebook `colab/run_pipeline.ipynb` runs the 32 px experiments. A 32 px pass supports the
claim at 32 px only. Not implemented: the latent-diffusion engine (D3) and real crawler parsing (D5 needs a
sample record), so latent-diffusion suspects are reported as "Not testable".

## Results so far

From one Colab T4 run: 32 px, CIFAR-10 birds, budget 4/255, 2,000 fine-tuning steps, 500 images per model.

- **E1 fails.** Bit accuracy is 99.9% with no attack, but JPEG 90/80/60, WebP 80 and resize to 50% all fall to 46.1%,
  the same as unmarked images, so the mark is gone. This engine has no corruption training; the paper's 94.45% at
  JPEG used 8/255 with corruptions added to training. Invisibility: PSNR 37.0 dB, SSIM 0.980, LPIPS 0.00016. Uniformity Z 0.95.
- **E2 (reduced) passes.** The model fine-tuned on 100% marked images reaches 98.0% bit accuracy on generated images
  and is Detected. The clean model scores 61.7% and is Not detected, but that score is above chance:
  `exp_owner_scan.py` investigates it.
- **Not established:** false-accusation rate (E4), anything at 256 px, repeat runs.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
export SHIELD_KEY=$(python3 -c "import secrets;print(secrets.token_hex(32))")   # never commit it
.venv/bin/python -m pytest -q tests
```
