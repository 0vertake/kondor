"""Stage 0: can this machine fine-tune and sample google/ddpm-celebahq-256?

Measures real seconds per training step and per sampling step, then extrapolates to the design's
experiments. Writes out/timing.json. Random tensors stand in for images; speed does not depend on them.

Usage: python timing_test.py [--batch 4] [--train-steps 5] [--threads 8]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel


def rss_gb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM"):
                return int(line.split()[1]) / 1024 ** 2
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/ddpm-celebahq-256")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--train-steps", type=int, default=5)
    ap.add_argument("--sample-steps", type=int, default=5)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    unet = UNet2DModel.from_pretrained(a.model).to(dev)
    res = {"model": a.model, "params_million": sum(p.numel() for p in unet.parameters()) / 1e6,
           "image_size": unet.config.sample_size, "threads": a.threads, "device": dev, "load_seconds": time.time() - t0}
    print(res, flush=True)
    sched = DDPMScheduler(num_train_timesteps=1000)
    size = unet.config.sample_size

    # ----- training steps
    unet.train()
    opt = torch.optim.AdamW(unet.parameters(), lr=1e-5)
    times = []
    for i in range(a.train_steps):
        t0 = time.time()
        x = torch.randn(a.batch, 3, size, size, device=dev)
        noise = torch.randn_like(x)
        t = torch.randint(0, 1000, (a.batch,), device=dev)
        loss = F.mse_loss(unet(sched.add_noise(x, noise, t), t).sample, noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
        times.append(time.time() - t0)
        print(f"train step {i}: {times[-1]:.1f}s  peak RSS {rss_gb():.1f} GB", flush=True)
    train_s = sum(times[1:]) / max(len(times) - 1, 1)
    res.update(batch=a.batch, seconds_per_train_step=train_s, seconds_per_image_train=train_s / a.batch,
               peak_rss_gb_train=rss_gb())

    # ----- sampling steps (one UNet forward per step)
    unet.eval()
    times = []
    with torch.no_grad():
        for i in range(a.sample_steps):
            t0 = time.time()
            unet(torch.randn(1, 3, size, size, device=dev), torch.tensor([500], device=dev)).sample
            times.append(time.time() - t0)
    s = sum(times[1:]) / max(len(times) - 1, 1)
    res["seconds_per_unet_forward"] = s

    # ----- extrapolation to the design's experiments
    gen_ddim50 = 50 * s
    ft_steps = 2000                       # assumption: a modest fine-tune; the design does not fix this
    ft_h = ft_steps * train_s / 3600
    gen500_h = 500 * gen_ddim50 / 3600
    res["extrapolation"] = {
        "assumed_finetune_steps": ft_steps,
        "one_finetune_hours": ft_h,
        "one_image_ddim50_seconds": gen_ddim50,
        "500_images_ddim50_hours": gen500_h,
        "E2_two_finetunes_plus_sampling_hours": 2 * (ft_h + gen500_h),
        "E3_five_shares_hours": 5 * (ft_h + gen500_h),
        "E4_30_clean_finetunes_hours": 30 * (ft_h + gen500_h),
    }
    Path("out").mkdir(exist_ok=True)
    Path("out/timing.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
