#!/usr/bin/env python3
"""Convert an ablation-directions tensor (pqhaz's ablation_directions.pt, a PyTorch zip) into a Basalt
runtime-directions bundle (BSLDIR01) for `basalt --directions`.

Only the standard library; no pickle execution: the tensor's raw f32 storage is read out of the zip member
(directions/data/3 in the pinned pqhaz file).  Every row must be a unit vector or zero; the layer mask is the
nonzero rows.  Usage:

    python3 tools/deepseek41_directions.py ablation_directions.pt deepseek41.directions

The pinned source is pqhaz/DeepSeek-V4.1-Flash-1017-heretic at revision
12d58ad0878b5008e80d720251adb969a0069412 (40 layers x 5120, layers 2..39); a different file converts the same way
but is reported, so a silent upstream change is seen."""
import argparse
import hashlib
from pathlib import Path
import struct
import sys
from zipfile import ZipFile

MAGIC = b"BSLDIR01"
PINNED_SHA256 = "87e81bffab36b046786eb7a3888c8d12c69ddef145235e363e0d4fb2c8fe31b4"
MEMBER_PREFIX = "directions/data/"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pt", type=Path, help="the ablation_directions.pt (a PyTorch zip)")
    parser.add_argument("output", type=Path, help="the BSLDIR01 bundle to write")
    parser.add_argument("--hidden", type=int, default=5120, help="the model's hidden size (default 5120)")
    parser.add_argument("--member", default="directions/data/3",
                        help="the zip member with the tensor (default directions/data/3, the 'projected' tensor "
                             "of the pinned pqhaz file)")
    parser.add_argument("--layers", default="2:40", metavar="FIRST:END",
                        help="the layers that keep their direction, FIRST inclusive to END exclusive (default "
                             "2:40, the layers pqhaz's heretic model and the TensorFold recipe ablate; rows "
                             "outside are zeroed)")
    args = parser.parse_args()
    try:
        first, end = (int(x) for x in args.layers.split(":"))
    except ValueError:
        parser.error(f"--layers: expected FIRST:END, not {args.layers!r}")

    data = args.pt.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != PINNED_SHA256:
        print(f"warning: {args.pt} is not the pinned pqhaz directions file (sha256 {digest}); converting anyway",
              file=sys.stderr)

    with ZipFile(args.pt) as archive:
        try:
            tensor = archive.read(args.member)
        except KeyError:
            parser.error(f"{args.pt} has no member {args.member} "
                         f"({', '.join(n for n in archive.namelist() if '/data/' in n)})")
    if len(tensor) % (args.hidden * 4) != 0 or len(tensor) < args.hidden * 4:
        parser.error(f"{args.member}: {len(tensor)} bytes is not whole rows of {args.hidden} f32")
    floats = struct.unpack(f"<{len(tensor) // 4}f", tensor)
    n_layers = len(floats) // args.hidden
    if not 1 <= n_layers <= 64:
        parser.error(f"{member}: {n_layers} layers, expected 1..64")

    mask = 0
    rows = []
    for layer in range(n_layers):
        row = floats[layer * args.hidden:(layer + 1) * args.hidden]
        norm = sum(v * v for v in row) ** 0.5
        if norm == 0.0 or not first <= layer < end:
            rows.append((0.0,) * args.hidden if norm != 0.0 else row)
            continue
        if abs(norm - 1.0) > 1e-5:
            parser.error(f"layer {layer} is not a unit vector (norm {norm:.6f})")
        mask |= 1 << layer
        rows.append(row)
    if mask == 0:
        parser.error("every row is zero: no directions")

    packet = MAGIC + struct.pack("<IIQ", n_layers, args.hidden, mask) + b"".join(
        struct.pack(f"<{len(row)}f", *row) for row in rows)
    args.output.write_bytes(packet)
    layers = [layer for layer in range(n_layers) if (mask >> layer) & 1]
    print(f"{args.output}: {n_layers} layers x {args.hidden}, directions on layers "
          f"{layers[0]}..{layers[-1]} ({len(layers)}), sha256 {hashlib.sha256(packet).hexdigest()}"
          + ("" if digest == PINNED_SHA256 else " (from an UNPINNED source)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
