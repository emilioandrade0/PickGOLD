from __future__ import annotations

from pathlib import Path
import json
import sys


SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT
SUMMARY_FILE = BASE_DIR / "data" / "tennis" / "reports" / "walkforward_summary_tennis.json"


def main() -> None:
    if not SUMMARY_FILE.exists():
        print(f"ERROR: No existe {SUMMARY_FILE}")
        return

    summary = json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
    print("=" * 54)
    print("REPORTE BASELINE TENNIS")
    print("=" * 54)
    print(f"Status        : {summary.get('status', '-')}")
    print(f"Rows          : {summary.get('rows', 0)}")
    print(f"Resueltas     : {summary.get('resolved_rows', 0)}")
    acc = summary.get("accuracy")
    print(f"Accuracy      : {acc:.2%}" if isinstance(acc, (int, float)) else "Accuracy      : N/A")
    tiers = summary.get("tiers") or {}
    if tiers:
        print("-" * 54)
        print("POR TIER")
        for tier, values in tiers.items():
            tier_acc = values.get("accuracy")
            print(f"{tier:<12} rows={values.get('rows', 0):>4} | acc={tier_acc:.2%}" if isinstance(tier_acc, (int, float)) else f"{tier:<12} rows={values.get('rows', 0):>4} | acc=N/A")
    print("=" * 54)
    print(f"Fuente: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
