"""
DeepSeek V4 Flash FP8 abliteration.
Weights: float8_e4m3fn with e8m0fnu block scales [rows/128, cols/128].
Strategy:
  1. Cast float8 weights to float32 (torch handles fp8→fp32 natively)
  2. Compute refusal direction via SVD of shared_expert gate weights (layers 0-8)
  3. Project direction out of w1 (gate) and w3 (up) for all shared_experts
  4. Optionally also project from attention output wo_b
  5. Clamp to float8_e4m3fn range [-448, 448] and cast back
  6. Keep scale tensors unchanged
"""

import os, gc, json, shutil, torch
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

MODEL_DIR = Path("/home/sunil/models/DeepSeek-V4-Flash")
OUT_DIR   = Path("/home/sunil/models/DeepSeek-V4-Flash-abliterated")
FP8_MAX   = 448.0   # float8_e4m3fn max value

ALPHA = 1.0  # Projection strength (1.0 = full removal)


def dequant_fp8(w_fp8):
    """Cast float8_e4m3fn to float32 — torch handles this directly."""
    return w_fp8.to(torch.float32)


def requant_fp8(w_f32, orig_fp8):
    """Clamp to FP8 range and cast back, preserving dtype."""
    clamped = torch.clamp(w_f32, -FP8_MAX, FP8_MAX)
    return clamped.to(orig_fp8.dtype)


def compute_refusal_direction(model_dir, layers=range(9)):
    """
    SVD of shared_expert gate (w1) weights across early layers.
    Returns the top right singular vector (input-space direction).
    """
    with open(model_dir / "model.safetensors.index.json") as f:
        wm = json.load(f)["weight_map"]

    directions = []
    print("Computing refusal direction from shared expert gate weights...")

    for li in layers:
        key = f"layers.{li}.ffn.shared_experts.w1.weight"
        if key not in wm:
            continue
        shard = model_dir / wm[key]
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            if key not in f.keys():
                continue
            w = dequant_fp8(f.get_tensor(key))   # [out, in]

        # Top right singular vector = principal direction in input space
        try:
            _, _, Vh = torch.linalg.svd(w, full_matrices=False)
            directions.append(Vh[0])
            print(f"  Layer {li}: w1 {list(w.shape)} → direction {list(Vh[0].shape)}")
        except Exception as e:
            print(f"  Layer {li}: SVD failed: {e}")
        del w; gc.collect()

    if not directions:
        raise RuntimeError("No shared_expert w1 weights found")

    d = torch.stack(directions).mean(dim=0)
    d = d / d.norm()
    print(f"Refusal direction: shape={list(d.shape)}, norm={d.norm():.6f}")
    return d


def project_out(W_f32, direction, alpha=ALPHA):
    """
    Remove `direction` from W.
    W shape: [out, in]
    direction: [in] (input-space) or [out] (output-space)
    """
    if direction.shape[0] == W_f32.shape[1]:
        # Input-space projection: W -= W @ d @ dT
        proj = (W_f32 @ direction).unsqueeze(1) * direction.unsqueeze(0)
        return W_f32 - alpha * proj
    elif direction.shape[0] == W_f32.shape[0]:
        # Output-space projection
        proj = direction.unsqueeze(1) * (direction @ W_f32).unsqueeze(0)
        return W_f32 - alpha * proj
    else:
        # Resize direction
        d = torch.nn.functional.interpolate(
            direction.float().unsqueeze(0).unsqueeze(0),
            size=W_f32.shape[1], mode='linear', align_corners=False
        ).squeeze()
        d = d / d.norm()
        proj = (W_f32 @ d).unsqueeze(1) * d.unsqueeze(0)
        return W_f32 - alpha * proj


def abliterate():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Copy config files
    print("Copying config files...")
    for f in MODEL_DIR.iterdir():
        if f.is_file() and f.suffix in ['.json', '.py', '.md', '.tiktoken', '.txt', '']:
            try:
                shutil.copy2(f, OUT_DIR / f.name)
            except Exception:
                pass

    # Compute refusal direction
    refusal_dir = compute_refusal_direction(MODEL_DIR)

    # Load shard index
    with open(MODEL_DIR / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]

    # Target keys to modify
    TARGET_SUFFIXES = [
        "ffn.shared_experts.w1.weight",  # gate proj — most important
        "ffn.shared_experts.w3.weight",  # up proj
        "ffn.shared_experts.w2.weight",  # down proj
        "attn.wo_b.weight",              # attention output
    ]

    all_shards = sorted(set(weight_map.values()))
    print(f"\nProcessing {len(all_shards)} shards...")

    for shard_name in all_shards:
        shard_path = MODEL_DIR / shard_name
        out_path   = OUT_DIR   / shard_name

        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            tensors = {}

            for key in keys:
                t = f.get_tensor(key)

                is_target = (
                    any(key.endswith(sfx) for sfx in TARGET_SUFFIXES)
                    and t.ndim == 2
                    and t.dtype == torch.float8_e4m3fn
                )

                if is_target:
                    w_f = dequant_fp8(t)
                    w_new = project_out(w_f, refusal_dir)
                    tensors[key] = requant_fp8(w_new, t)
                    print(f"  ✓ {key}  {list(t.shape)}")
                    del w_f, w_new
                else:
                    tensors[key] = t

        save_file(tensors, str(out_path))
        del tensors; gc.collect()
        print(f"  Saved {shard_name}")

    # Copy index
    shutil.copy2(
        MODEL_DIR / "model.safetensors.index.json",
        OUT_DIR   / "model.safetensors.index.json"
    )
    print(f"\n✅ Done! Abliterated model at: {OUT_DIR}")


if __name__ == "__main__":
    abliterate()
