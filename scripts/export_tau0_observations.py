#!/usr/bin/env python3
"""Extract current three-camera observations from a RoboTwin LeRobot dataset."""
import argparse
import json
from agenticwam.coding.subtask_data import export_observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--episodes", help="optional JSON list of original episode indices")
    args = parser.parse_args()
    selected = json.load(open(args.episodes)) if args.episodes else None
    path, count = export_observations(args.dataset, args.output, stride=args.stride, episodes=selected)
    print(f"Wrote {count} observations: {path}")


if __name__ == "__main__":
    main()
