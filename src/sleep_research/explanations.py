"""Bounded saliency, compatible Conv1d Grad-CAM and explicit attention-weight capture."""

from pathlib import Path
import json
import numpy as np
import torch

from .common import atomic_file, digest, guard_evaluation, json_write


def saliency(model, inputs, targets=None):
    model.eval()
    values = inputs.detach().clone().requires_grad_(True)
    logits = model(values)
    if targets is None:
        targets = logits.argmax(dim=1)
    objective = logits[torch.arange(len(values), device=values.device), targets].sum()
    gradients = torch.autograd.grad(objective, values)[0]
    return gradients.detach().cpu().numpy().astype(np.float32), targets.detach().cpu().numpy()


def grad_cam_1d(model, inputs, layer_name, targets=None):
    layer = dict(model.named_modules()).get(layer_name)
    if not isinstance(layer, torch.nn.Conv1d):
        raise ValueError("1D Grad-CAM requires a compatible named Conv1d layer")
    activation = []
    hook = layer.register_forward_hook(lambda module, values, output: activation.append(output))
    try:
        model.eval()
        values = inputs.detach().clone().requires_grad_(True)
        logits = model(values)
        if targets is None:
            targets = logits.argmax(dim=1)
        objective = logits[torch.arange(len(values), device=values.device), targets].sum()
        gradient = torch.autograd.grad(objective, activation[0])[0]
        weights = gradient.mean(dim=-1, keepdim=True)
        cams = torch.relu((weights * activation[0]).sum(dim=1, keepdim=True))
        cams = torch.nn.functional.interpolate(cams, size=inputs.shape[-1], mode="linear", align_corners=False)
        maximum = cams.amax(dim=-1, keepdim=True)
        cams = torch.where(maximum > 0, cams / maximum.clamp_min(1e-12), cams)
        return cams.detach().cpu().numpy().astype(np.float32), targets.detach().cpu().numpy()
    finally:
        hook.remove()


def attention_weights(model, inputs, layer_name, max_tokens=128):
    layer = dict(model.named_modules()).get(layer_name)
    if layer is None:
        raise ValueError("Unknown attention layer")
    captured = []
    def capture(module, values, output):
        if not isinstance(output, tuple) or len(output) < 2 or not isinstance(output[1], torch.Tensor):
            raise ValueError("Architecture must explicitly return attention weights (e.g. need_weights=True)")
        weights = output[1]
        if weights.ndim not in (3, 4) or max(weights.shape[-2:]) > max_tokens:
            raise ValueError("Attention weights exceed the configured token limit or have unsupported layout")
        captured.append(weights.detach().cpu().numpy().astype(np.float32))
    hook = layer.register_forward_hook(capture)
    try:
        model.eval()
        with torch.no_grad():
            model(inputs)
        if len(captured) != 1:
            raise ValueError("Configured attention layer must execute once per explanation batch")
        return captured[0]
    finally:
        hook.remove()


def export_explanations(model, batch, output, checkpoint_path, conv_layer=None, attention_layer=None,
                        max_samples=8, max_attention_tokens=128, final_evaluation=False):
    guard_evaluation(batch["split"], final_evaluation)
    checkpoint_path = Path(checkpoint_path)
    sidecar = json.loads(checkpoint_path.with_suffix(checkpoint_path.suffix + ".json").read_text())
    if digest(checkpoint_path) != sidecar["sha256"]:
        raise ValueError("Explanation checkpoint is corrupt")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    current = model.state_dict()
    if set(current) != set(state["model"]) or any(not torch.equal(current[key].detach().cpu(), state["model"][key]) for key in current):
        raise ValueError("Explanation model weights differ from the named checkpoint")
    if max_samples < 1:
        raise ValueError("Explanation sample cap must be positive")
    count = min(max_samples, len(batch["label"]))
    device = next(model.parameters()).device
    values = batch["eeg"][:count].to(device)
    gradients, targets = saliency(model, values)
    arrays = {"signed_saliency": gradients, "absolute_saliency": np.abs(gradients), "target_labels": targets,
              "epoch_ids": np.array(list(batch["epoch_id"][:count]), dtype=str)}
    if conv_layer:
        arrays["grad_cam_1d"] = grad_cam_1d(model, values, conv_layer)[0]
    if attention_layer:
        arrays["attention_weights"] = attention_weights(model, values, attention_layer, max_attention_tokens)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    with atomic_file(output / "explanations.npz", binary=True) as handle:
        np.savez_compressed(handle, **arrays)
    json_write(output / "manifest.json", {"epoch_ids": arrays["epoch_ids"].tolist(), "sample_count": count,
               "checkpoint_sha256": digest(checkpoint_path), "conv_layer": conv_layer, "attention_layer": attention_layer,
               "maximum_attention_tokens": max_attention_tokens, "files": {"explanations.npz": digest(output / "explanations.npz")},
               "synthetic": bool(state["configuration"].get("synthetic", False)),
               "note": "Post-hoc inference explanations, not evidence of causal physiology"})
    # Figures are generated from the saved bounded arrays, without training.
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(count, 1, figsize=(10, max(3, count * 2)), squeeze=False, constrained_layout=True)
    for i in range(count):
        axis = axes[i, 0]
        axis.plot(arrays["absolute_saliency"][i].reshape(-1), label="Absolute input gradient")
        if "grad_cam_1d" in arrays:
            secondary = axis.twinx(); secondary.plot(arrays["grad_cam_1d"][i].reshape(-1), color="orange", alpha=0.6)
            secondary.set_ylabel("Grad-CAM [0,1]")
        axis.set(title=arrays["epoch_ids"][i], xlabel="Sample index", ylabel="Gradient magnitude")
    fig.savefig(output / "saliency_grad_cam.png", dpi=120); plt.close(fig)
    if "attention_weights" in arrays:
        weights = arrays["attention_weights"]
        weights = weights[0].mean(axis=0) if weights.ndim == 4 else weights[0]
        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        ax.imshow(weights, cmap="viridis"); ax.set(xlabel="Key token", ylabel="Query token", title="Saved attention, sample 0 (head mean)")
        fig.savefig(output / "attention.png", dpi=140); plt.close(fig)
    return output
