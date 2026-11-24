"""The `ablation` request field: per-request runtime directions (refusal-vector ablation) for an engine loaded
with --directions.  `ablation` is a strength in 0..1, or {"strength": S, "layers": [L, ...]} to ablate only those
layers (0..63); absent or 0 is the exact baseline.  The keys reach the engine as `abl=S ablmask=HEX` on the GEN
line; an engine without directions refuses them there."""
from __future__ import annotations


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def checked_ablation(value) -> None:
    """ValueError naming what is wrong, so the request gets a 400 before streaming."""
    if _num(value):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"ablation: expected a strength 0..1, not {value!r}")
        return
    if isinstance(value, dict):
        unknown = set(value) - {"strength", "layers"}
        if unknown:
            raise ValueError(f"ablation: unknown key {sorted(unknown)[0]!r} (strength, layers)")
        strength = value.get("strength")
        if not _num(strength) or not 0.0 <= float(strength) <= 1.0:
            raise ValueError(f"ablation.strength: expected a number 0..1, not {strength!r}")
        layers = value.get("layers")
        if layers is not None:
            if not isinstance(layers, list) or not layers or \
                    any(not isinstance(l, int) or isinstance(l, bool) or not 0 <= l < 64 for l in layers):
                raise ValueError(f"ablation.layers: expected a nonempty list of layer numbers 0..63, not {layers!r}")
        return
    raise ValueError(f"ablation: expected a strength 0..1 or {{'strength': S, 'layers': [...]}}, not {value!r}")


def ablation_key(sampling: dict) -> str:
    """The GEN line's `abl=S ablmask=HEX` keys; empty when the request does not steer."""
    value = sampling.get("ablation")
    if value is None:
        return ""
    if _num(value):
        strength, layers = float(value), None
    else:
        strength, layers = float(value.get("strength") or 0.0), value.get("layers")
    if strength <= 0.0:
        return ""   # the exact baseline: nothing sent, nothing applied
    keys = f" abl={strength!r}"
    if layers:
        mask = 0
        for layer in layers:
            mask |= 1 << layer
        keys += f" ablmask={mask:x}"
    return keys
