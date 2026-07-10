# vanish/cli.py
"""Argparse CLI. Parses args, calls io + features, writes output. No algorithm
logic lives here."""
import argparse
import sys
import numpy as np
from vanish import core, io, features


def _target(current, absolute, delta):
    """Resolve an absolute target or a delta into an absolute dimension."""
    if absolute is not None:
        return absolute
    if delta is not None:
        return current + delta
    return None


def _build_parser():
    p = argparse.ArgumentParser(prog="vanish", description="Seam-carving resizer")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name in ("shrink", "enlarge"):
        sp = sub.add_parser(name)
        sp.add_argument("input")
        sp.add_argument("output")
        sp.add_argument("--width", type=int)
        sp.add_argument("--height", type=int)
        sp.add_argument("--dw", type=int, help="delta width (+enlarge/-shrink)")
        sp.add_argument("--dh", type=int, help="delta height")

    rm = sub.add_parser("remove")
    rm.add_argument("input")
    rm.add_argument("output")
    rm.add_argument("--mask", required=True)
    rm.add_argument("--shrink", action="store_true")

    en = sub.add_parser("energy")
    en.add_argument("input")
    en.add_argument("output")

    sm = sub.add_parser("seams")
    sm.add_argument("input")
    sm.add_argument("output")
    sm.add_argument("--count", type=int, required=True)
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    img = io.load_image(args.input)
    h, w = img.shape[:2]

    if args.cmd in ("shrink", "enlarge"):
        tw = _target(w, args.width, args.dw)
        th = _target(h, args.height, args.dh)
        if tw is None and th is None:
            sys.exit("Specify --width/--height or --dw/--dh")
        if args.cmd == "shrink":
            if tw is not None and not (1 <= tw < w):
                sys.exit(f"shrink width must be between 1 and {w - 1}")
            if th is not None and not (1 <= th < h):
                sys.exit(f"shrink height must be between 1 and {h - 1}")
        elif args.cmd == "enlarge":
            if tw is not None and tw <= w:
                sys.exit(f"enlarge width must be > {w}")
            if th is not None and th <= h:
                sys.exit(f"enlarge height must be > {h}")
        out = features.resize(img, width=tw, height=th)

    elif args.cmd == "remove":
        mask = io.load_mask(args.mask, (h, w))
        out = features.remove_object(img, mask, shrink=args.shrink)

    elif args.cmd == "energy":
        e = core.energy_map(img)
        e = (255 * e / e.max()).astype("uint8") if e.max() > 0 else e.astype("uint8")
        out = np.stack([e, e, e], axis=-1)

    elif args.cmd == "seams":
        if not (1 <= args.count < w):
            sys.exit(f"--count must be between 1 and {w - 1}")
        out = _overlay_seams(img, args.count)

    io.save_image(args.output, out)


def _overlay_seams(img, count):
    """Draw `count` lowest-energy seams in red over a copy of the image."""
    out = img.copy()
    for seam in core.compute_seams(img, count):
        out[range(out.shape[0]), seam] = [255, 0, 0]
    return out
