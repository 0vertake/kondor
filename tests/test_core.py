import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shield.config import CONFIGS, EngineConfig  # noqa: E402
from shield.engine import Decoder, Engine, random_blocks, train_engine  # noqa: E402
from shield import marking, verdict  # noqa: E402

KEY = b"unit-test-key"


def test_message_deterministic_length_and_independent():
    a = marking.message_bits(KEY, "acme", "d1", 512)
    assert len(a) == 512 and np.array_equal(a, marking.message_bits(KEY, "acme", "d1", 512))
    assert 0.35 < (a == marking.message_bits(KEY, "acme", "d2", 512)).mean() < 0.65
    assert not np.array_equal(a, marking.message_bits(b"other", "acme", "d1", 512))
    # the separator stops ("ab","c") colliding with ("a","bc")
    assert not np.array_equal(marking.message_bits(KEY, "ab", "c", 128), marking.message_bits(KEY, "a", "bc", 128))


def test_config_capacity_matches_paper():
    assert CONFIGS["px32"].message_bits == 128
    assert CONFIGS["px256"].message_bits == 256
    assert CONFIGS["px32"].hash() != CONFIGS["px256"].hash()


def _engine(cfg):
    patches = torch.zeros(cfg.B, 3, cfg.patch, cfg.patch)
    for i in range(1, cfg.B):
        patches[i] = (torch.rand(3, cfg.patch, cfg.patch) * 2 - 1) * cfg.eps
    return Engine(cfg, patches, Decoder(cfg.B))


@pytest.mark.parametrize("name", ["px32", "px256"])
def test_symbol_bit_roundtrip_and_pattern_shape(name):
    cfg = CONFIGS[name]
    e = _engine(cfg)
    bits = marking.message_bits(KEY, "o", "d", cfg.message_bits)
    sym = e.symbols_from_bits(bits)
    assert sym.max() < cfg.B and len(sym) == cfg.blocks
    assert np.array_equal(e.bits_from_symbols(sym), bits)
    W = e.pattern(bits)
    assert W.shape == (3, cfg.image_size, cfg.image_size)
    assert W.abs().max() <= cfg.eps + 1e-6
    # block (r, c) of W is the patch chosen by symbol r*n + c
    n, p = cfg.image_size // cfg.patch, cfg.patch
    r, c = n - 1, 1
    assert torch.equal(W[:, r * p:(r + 1) * p, c * p:(c + 1) * p], e.patches[int(sym[r * n + c])])


def test_decode_block_order_matches_pattern():
    """An oracle decoder that reads the mean of each block must return the message in order."""
    cfg = EngineConfig("t", 32, 4, 2)
    e = _engine(cfg)
    e.patches[1] = torch.full((3, 4, 4), 0.5)

    class Oracle(torch.nn.Module):
        def forward(self, x):
            m = x.mean(dim=(1, 2, 3))
            return torch.stack([-m, m - 0.25], 1)
    e.decoder = Oracle()
    bits = marking.message_bits(KEY, "o", "d", cfg.message_bits)
    assert np.array_equal(e.decode_bits(e.pattern(bits)[None])[0], bits)


def test_training_learns_on_tiny_problem():
    cfg = EngineConfig("tiny", 32, 4, 4, eps255=8, steps=150, batch=64)
    imgs = torch.rand(32, 3, 32, 32) * 0.5 + 0.25
    res = train_engine(cfg, imgs, log=lambda s: None)
    assert res["patches"][0].abs().max() == 0                      # patch 1 stays zero
    assert res["patches"].abs().max() <= cfg.eps + 1e-6            # budget holds
    assert torch.allclose(res["patches"] * 255, (res["patches"] * 255).round())  # whole 8-bit levels
    e = Engine(cfg, res["patches"], res["decoder"])
    msg = marking.message_bits(KEY, "o", "d", cfg.message_bits)
    x = ((imgs[:8] + e.pattern(msg)).clamp(0, 1) * 255).round() / 255
    # 150 steps on noise images only has to show learning; the >= 90% bar is E1, on real images
    assert verdict.bit_accuracy(e.decode_bits(x), msg).mean() > 0.65


def test_registry_chain_detects_tampering(tmp_path):
    log = tmp_path / "r.jsonl"
    for o in "abc":
        marking.registry_append({"owner": o, "dataset": "d"}, log)
    assert marking.registry_verify(log)
    assert marking.registry_find("b", "d", log)["owner"] == "b"
    lines = log.read_text().splitlines()
    r = json.loads(lines[1]); r["owner"] = "mallory"
    lines[1] = json.dumps(r, sort_keys=True)
    log.write_text("\n".join(lines) + "\n")
    assert not marking.registry_verify(log)


def test_random_message_scores_match_direct_computation():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, (50, 64), dtype=np.uint8)
    msg = rng.integers(0, 2, 64, dtype=np.uint8)
    r = verdict.random_message_test(bits, msg, K=200)
    assert r["owner_score"] == pytest.approx(verdict.bit_accuracy(bits, msg).mean())


def test_test1_is_calibrated_under_the_null():
    """With unrelated bits, p-values must be uniform: about alpha of them fall below alpha."""
    rng = np.random.default_rng(1)
    bias = rng.uniform(0.2, 0.8, 128)                              # a strongly biased decoder
    ps = []
    for i in range(400):
        bits = (rng.random((100, 128)) < bias).astype(np.uint8)
        ps.append(verdict.random_message_test(bits, rng.integers(0, 2, 128, dtype=np.uint8), K=500, seed=i)["p"])
    assert 0.05 < np.mean(np.array(ps) <= 0.1) < 0.15


def test_decide_all_verdicts():
    rng = np.random.default_rng(2)
    msg = rng.integers(0, 2, 128, dtype=np.uint8)
    clean = rng.integers(0, 2, (200, 128), dtype=np.uint8)
    flip = rng.random((200, 128)) < 0.05
    marked = np.where(flip, 1 - msg, msg).astype(np.uint8)
    other = rng.integers(0, 2, (200, 128), dtype=np.uint8)
    K = 5000
    assert verdict.decide(marked, msg, clean, K=K).verdict == "Detected"
    assert verdict.decide(other, msg, clean, K=K).verdict == "Not detected"
    assert verdict.decide(marked, msg, None, K=K).verdict == "Inconclusive"
    assert verdict.decide(marked[:20], msg, clean, K=K).verdict == "Inconclusive"
    # a decoder always returning the owner's message on BOTH models: test 1 passes, test 2 does not
    assert verdict.decide(marked, msg, marked, K=K).verdict == "Inconclusive"
    with pytest.raises(ValueError):
        verdict.decide(marked, msg, clean, K=100)


def test_intake_never_calls_untestable_clean():
    from shield.intake import Suspect, eligibility
    base = dict(suspect_id="s", source_url="u")
    assert eligibility(Suspect(**base, access="open-weights", family="pixel-diffusion", claimed_base_model="m"))[0]
    assert not eligibility(Suspect(**base, access="api", family="pixel-diffusion", claimed_base_model="m"))[0]
    assert not eligibility(Suspect(**base, access="open-weights", family="latent-diffusion", claimed_base_model="m"))[0]
    assert not eligibility(Suspect(**base, access="open-weights", family="pixel-diffusion"))[0]


def test_report_written(tmp_path):
    from shield import report
    cfg = CONFIGS["px32"]
    rng = np.random.default_rng(3)
    msg = rng.integers(0, 2, cfg.message_bits, dtype=np.uint8)
    bits = np.tile(msg, (120, 1))
    ref = rng.integers(0, 2, (120, cfg.message_bits), dtype=np.uint8)
    v = verdict.decide(bits, msg, ref, K=5000).as_dict()
    reg = {"owner": "o", "dataset": "d", "message_hash": "a" * 64, "prev_hash": "b" * 64}
    sus = {"suspect_id": "x", "source_url": "u", "family": "pixel-diffusion", "claimed_base_model": "m"}
    imgs = torch.rand(120, 3, 32, 32)
    path = report.write_report(tmp_path, sus, reg, cfg, v, {"n": 120, "seed": 0, "ddim_steps": 50, "image_hash": "h"},
                               imgs, bits, ref, msg, rng.normal(0.5, 0.01, 1000))
    text = path.read_text()
    assert "Detected" in text and "does not mean the model is clean" in text and "E4" in text


def test_interrupted_fine_tune_resumes_to_identical_weights(tmp_path):
    """A disconnected Colab session must not change the result."""
    from diffusers import UNet2DModel
    from shield.generator import fine_tune
    base = tmp_path / "base"
    torch.manual_seed(0)
    UNet2DModel(sample_size=8, block_out_channels=(4, 8), layers_per_block=1, norm_num_groups=2,
                down_block_types=("DownBlock2D", "DownBlock2D"), up_block_types=("UpBlock2D", "UpBlock2D")).save_pretrained(base)
    imgs = torch.rand(8, 3, 8, 8)
    kw = dict(steps=4, batch=2, save_every=2)
    fine_tune(str(base), imgs, tmp_path / "straight", log=lambda s: None, **kw)

    def die(msg):
        if "step 3/4" in msg:
            raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        fine_tune(str(base), imgs, tmp_path / "cut", log=die, **kw)
    assert (tmp_path / "cut" / "resume.pt").exists()
    seen = []
    fine_tune(str(base), imgs, tmp_path / "cut", log=seen.append, **kw)
    assert any("resuming at step 2" in s for s in seen)
    assert not (tmp_path / "cut" / "resume.pt").exists()
    a = UNet2DModel.from_pretrained(tmp_path / "straight").state_dict()
    b = UNet2DModel.from_pretrained(tmp_path / "cut").state_dict()
    assert all(torch.allclose(a[k], b[k], atol=1e-6) for k in a)
