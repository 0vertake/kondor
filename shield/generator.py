"""Fine-tuning and sampling a pixel-space diffusion model (used by experiments and the probe runner)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from .config import WORK, device
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel


def fine_tune(base: str, images: torch.Tensor, out: Path, steps: int, batch: int = 32, lr: float = 1e-4,
              seed: int = 0, log=print, save_every: int = 200) -> dict:
    """Noise-prediction training continued from pretrained weights. images in [0, 1].

    Resumable: a checkpoint is written every save_every steps, because Colab sessions can disconnect.
    Each step's randomness comes from (seed, step), so a resumed run draws the same batches.
    """
    out = Path(out)
    if (out / "train_log.json").exists():
        return json.loads((out / "train_log.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    dev = device()
    amp = dev == "cuda"
    torch.manual_seed(seed)
    unet = UNet2DModel.from_pretrained(base).to(dev)
    sched = DDPMScheduler.from_pretrained(base) if _has_scheduler(base) else DDPMScheduler(num_train_timesteps=1000)
    opt = torch.optim.AdamW(unet.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(enabled=amp)
    ck, start, spent = out / "resume.pt", 0, 0.0
    if ck.exists():
        state = torch.load(ck, map_location=dev, weights_only=True)
        unet.load_state_dict(state["unet"])
        opt.load_state_dict(state["opt"])
        scaler.load_state_dict(state["scaler"])
        start, spent = int(state["step"]), float(state["seconds"])
        log(f"[fine-tune {out.name}] resuming at step {start}")
    unet.train()
    t0, losses = time.time(), []
    gen = torch.Generator()
    for step in range(start, steps):
        gen.manual_seed(seed * 1_000_003 + step)
        idx = torch.randint(0, len(images), (batch,), generator=gen)
        x = (images[idx] * 2 - 1).to(dev)
        noise = torch.randn(x.shape, generator=gen).to(dev)
        t = torch.randint(0, sched.config.num_train_timesteps, (batch,), generator=gen).to(dev)
        with torch.autocast(dev, enabled=amp):
            loss = F.mse_loss(unet(sched.add_noise(x, noise, t), t).sample, noise)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(float(loss))
        if step % 100 == 0 or step == steps - 1:
            log(f"[fine-tune {out.name}] step {step}/{steps} loss {sum(losses[-100:]) / len(losses[-100:]):.4f} ({time.time() - t0:.0f}s)")
        if (step + 1) % save_every == 0 and step + 1 < steps:
            tmp = ck.with_suffix(".tmp")           # write then rename: a cut-off session cannot corrupt it
            torch.save({"unet": unet.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                        "step": step + 1, "seconds": spent + time.time() - t0}, tmp)
            tmp.replace(ck)
    unet.save_pretrained(out)
    ck.unlink(missing_ok=True)
    info = {"base": base, "steps": steps, "batch": batch, "lr": lr, "seed": seed, "n_images": len(images), "device": dev,
            "seconds": spent + time.time() - t0, "final_loss": sum(losses[-100:]) / max(len(losses[-100:]), 1)}
    (out / "train_log.json").write_text(json.dumps(info, indent=2))
    return info


def _has_scheduler(model: str) -> bool:
    try:
        DDPMScheduler.load_config(model)
        return True
    except Exception:
        return False


@torch.no_grad()
def sample(model: str | Path, n: int, seed: int = 0, ddim_steps: int = 50, batch: int = 32, log=print) -> torch.Tensor:
    """n images in [0, 1] with fixed per-image seeds, so a reference model can reuse the same seeds."""
    dev = device()
    unet = UNet2DModel.from_pretrained(model).to(dev).eval()
    sched = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear", clip_sample=True, prediction_type="epsilon")
    sched.set_timesteps(ddim_steps)
    s = unet.config.sample_size
    out, t0 = [], time.time()
    for i in range(0, n, batch):
        seeds = range(seed + i, seed + min(i + batch, n))
        x = torch.cat([torch.randn(1, 3, s, s, generator=torch.Generator().manual_seed(q)) for q in seeds]).to(dev)
        for t in sched.timesteps:
            x = sched.step(unet(x, t).sample, t, x).prev_sample
        out.append((((x.clamp(-1, 1) + 1) / 2 * 255).round() / 255).cpu())
        log(f"[sample] {min(i + batch, n)}/{n} ({time.time() - t0:.0f}s)")
    return torch.cat(out)


def cached_sample(model: str | Path, n: int, seed: int = 100_000, ddim_steps: int = 50, log=print,
                  batch: int = 32) -> torch.Tensor:
    """Probe images for a model, cached on disk so a reference is only generated once."""
    tag = str(model).replace("/", "_").strip("_")
    path = WORK / "probes" / f"{tag}_n{n}_s{seed}_d{ddim_steps}.pt"
    if path.exists():
        return torch.load(path, weights_only=True)
    x = sample(model, n, seed=seed, ddim_steps=ddim_steps, batch=batch, log=log)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(x, path)
    return x
