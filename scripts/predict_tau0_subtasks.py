#!/usr/bin/env python3
"""Query the frozen Tau0 service in episode order, carrying only past planner memory."""
import argparse
import json
from pathlib import Path
from agenticwam.coding.subtask_data import read_jsonl
from agenticwam.coding.tau0_client import ProposalClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--planner-config", default="configs/subtask/planner.yaml")
    args = parser.parse_args()
    rows = sorted(read_jsonl(args.input), key=lambda row: (row["dataset_root"], row["episode_index"], row["frame_start"]))
    client = ProposalClient.from_config(args.planner_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    previous = None
    with args.output.open("x") as handle:
        for row in rows:
            episode = (row["dataset_root"], row["episode_index"])
            if episode != previous:
                client.reset()
            images = {view: (args.input.resolve().parent / value).resolve() for view, value in row["images"].items()}
            result = client.predict(row["instruction"], images)
            handle.write(json.dumps({"id": row["id"], **result}, ensure_ascii=False) + "\n")
            handle.flush()
            previous = episode


if __name__ == "__main__":
    main()
