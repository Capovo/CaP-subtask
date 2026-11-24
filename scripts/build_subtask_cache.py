#!/usr/bin/env python3
"""Join observation metadata and real Tau0 outputs; reject incomplete frame coverage."""
import argparse
from agenticwam.coding.subtask_data import build_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    doc = build_cache(args.observations, args.predictions, args.output)
    print(f"Wrote {args.output}; datasets={len(doc['datasets'])}")


if __name__ == "__main__":
    main()
