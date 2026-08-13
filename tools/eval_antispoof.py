"""
Evaluate the anti-spoofing model on a captured dataset.

Reports the ISO/IEC 30107-3 presentation-attack-detection metrics, which is
the vocabulary this field actually uses:

  APCER  attack presentations wrongly accepted as live, computed per attack
         species and reported worst-case (the weakest species is your real
         exposure, so averaging across species would flatter the model)
  BPCER  genuine presentations wrongly rejected as attacks
  ACER   (APCER + BPCER) / 2

The model is only usable if some threshold drives both APCER and BPCER low at
once. If the score distributions overlap, no threshold exists and no amount of
threshold tuning will help — the model has to be replaced or retrained. That
verdict is printed explicitly, because it is the failure mode that looks like
a tuning problem and isn't.

    python -m tools.eval_antispoof --dataset Dataset/pad

Scores come from the bbox recorded at capture time, so the model sees exactly
what production would have handed it.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.antispoof import load_antispoof  # noqa: E402

BONA_FIDE = "live"
MANIFEST = "manifest.jsonl"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="Dataset/pad", help="dataset root")
    p.add_argument("--manifest", default=MANIFEST,
                   help="manifest filename inside the dataset root; point this at a "
                        "filtered copy to exclude samples without touching the raw one")
    p.add_argument("--json-out", help="write raw per-sample scores here")
    return p.parse_args()


def load_manifest(root, name=MANIFEST):
    path = root / name
    if not path.exists():
        sys.exit(f"No manifest at {path}.\nCapture a dataset first:\n"
                 f"  python -m tools.capture_pad_dataset --label live   --count 100\n"
                 f"  python -m tools.capture_pad_dataset --label print  --count 100\n"
                 f"  python -m tools.capture_pad_dataset --label replay --count 100")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def score_dataset(backend, root, rows):
    """-> {label: [scores]}, skipped_count"""
    by_label, skipped = {}, 0
    for r in rows:
        img = cv2.imread(str(root / r["file"]))
        if img is None:
            skipped += 1
            continue
        s = backend.score(img, r["bbox"])
        if s is None:
            skipped += 1
            continue
        by_label.setdefault(r["label"], []).append(s)
    return by_label, skipped


def rates(live, attacks, t):
    """BPCER, worst-case APCER, and the per-species APCERs at threshold t.
    Decision rule: accept as live when score >= t."""
    bpcer = float(np.mean(np.asarray(live) < t))
    per = {k: float(np.mean(np.asarray(v) >= t)) for k, v in attacks.items()}
    apcer = max(per.values()) if per else 0.0
    return bpcer, apcer, per


def auc(live, attack):
    """Probability a random live sample outscores a random attack (Mann-Whitney)."""
    live, attack = np.asarray(live), np.asarray(attack)
    if not len(live) or not len(attack):
        return float("nan")
    order = np.argsort(np.concatenate([live, attack]), kind="mergesort")
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks over ties so the statistic stays unbiased
    allv = np.concatenate([live, attack])
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    r1 = ranks[:len(live)].sum()
    return float((r1 - len(live) * (len(live) + 1) / 2) / (len(live) * len(attack)))


def report(by_label):
    print(f"\n{'='*74}\n  ANTI-SPOOFING EVALUATION\n{'='*74}")
    if BONA_FIDE not in by_label:
        print(f"  no '{BONA_FIDE}' samples — cannot evaluate")
        return None
    live = by_label[BONA_FIDE]
    attacks = {k: v for k, v in by_label.items() if k != BONA_FIDE}
    if not attacks:
        print("  no attack samples — capture at least one attack class")
        return None

    print(f"\n  score distributions ({len(live)} bona fide, "
          f"{sum(len(v) for v in attacks.values())} attack)")
    print(f"    {'class':12s}{'n':>5s}{'min':>9s}{'median':>9s}{'mean':>9s}{'max':>9s}")
    for k, v in [(BONA_FIDE, live)] + sorted(attacks.items()):
        a = np.asarray(v)
        print(f"    {k:12s}{len(a):5d}{a.min():9.4f}{np.median(a):9.4f}"
              f"{a.mean():9.4f}{a.max():9.4f}")

    all_attack = [s for v in attacks.values() for s in v]
    print(f"\n  AUC (live vs all attacks): {auc(live, all_attack):.4f}"
          "   [0.5 = coin flip, 1.0 = perfect]")

    cands = np.unique(np.concatenate([np.asarray(live), np.asarray(all_attack), [0.0, 1.0]]))
    rows = [(t,) + rates(live, attacks, t)[:2] for t in cands]
    best_t, best_b, best_a = min(rows, key=lambda r: (r[1] + r[2]) / 2)
    best_acer = (best_a + best_b) / 2

    print(f"\n  best operating point (minimises ACER)")
    print(f"    threshold {best_t:.4f}   APCER {best_a:7.2%}   BPCER {best_b:7.2%}"
          f"   ACER {best_acer:7.2%}")
    _, _, per = rates(live, attacks, best_t)
    for k, v in sorted(per.items()):
        print(f"      APCER[{k}] {v:.2%}")

    print(f"\n  BPCER at fixed APCER  (how many real people get rejected to hold "
          "attacks at a given rate)")
    for target in (0.01, 0.05, 0.10):
        ok = [(b, t) for t, b, a in rows if a <= target]
        if ok:
            b, t = min(ok)
            print(f"    APCER <= {target:4.0%}  ->  BPCER {b:7.2%}  (threshold {t:.4f})")
        else:
            print(f"    APCER <= {target:4.0%}  ->  unreachable at any threshold")

    live_min, atk_max = min(live), max(all_attack)
    print()
    if live_min > atk_max:
        print(f"  VERDICT: separable — every live score ({live_min:.4f}) exceeds every "
              f"attack score ({atk_max:.4f}).")
    elif best_acer < 0.10:
        print(f"  VERDICT: usable — ACER {best_acer:.2%} at threshold {best_t:.4f}. "
              f"Set SPOOF_THRESH={best_t:.2f}.")
    else:
        print(f"  VERDICT: NOT usable — best achievable ACER is {best_acer:.2%}. The "
              "distributions overlap;\n           no threshold fixes this. Replace or "
              "retrain the model.")
    return {"threshold": best_t, "apcer": best_a, "bpcer": best_b, "acer": best_acer}


def main():
    args = parse_args()
    root = Path(args.dataset)
    rows = load_manifest(root, args.manifest)
    print(f"Loaded {len(rows)} samples from {root/args.manifest}")

    backend = load_antispoof()
    if not backend.available:
        sys.exit("Anti-spoofing weights unavailable — check the download log above.")
    by_label, skipped = score_dataset(backend, root, rows)
    if skipped:
        print(f"  ({skipped} sample(s) unreadable or un-scoreable, skipped)")
    res = report(by_label)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"summary": res, "scores": by_label}, indent=2))
        print(f"\nRaw scores written to {args.json_out}")


if __name__ == "__main__":
    main()
