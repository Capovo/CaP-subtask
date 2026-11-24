#!/usr/bin/env python3
"""Derive a conditioned dataset config from the original RoboTwin code-labeled data."""
import argparse
from pathlib import Path
import yaml


def generate(source, tasks):
    config = yaml.safe_load(Path(source).read_text())
    entries = config["train"]["dataset"]["datasets"]
    selected = []
    names = set()
    for entry in entries:
        # Published cap dataset layout: <root>/<task>/lerobot/robotwin2.0.
        name = Path(entry["dataset_dirs"][0]).parent.parent.name
        if tasks and name not in tasks:
            continue
        entry["subtask_cache_path"] = "${paths.robotwin_subtasks}/" + name + "/subtask_cache.json"
        entry["require_high_level_subtask"] = True
        entry["pretrained_norm_stats"] = "${paths.robotwin_dataset_stats}"
        entry["processor"]["norm_default_mode"] = "q01/q99"
        selected.append(entry)
        names.add(name)
    if not selected or (tasks and set(tasks) - names):
        raise ValueError(f"Unknown/empty tasks: {sorted(set(tasks or []) - names)}")
    # Keep the original DeadlineDataset/ConcatDataset wrappers, with selected leaves.
    config["train"]["dataset"]["datasets"] = selected
    config.pop("_common", None)
    config.pop("_processor", None)
    config.pop("_shape_meta", None)
    return config


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(root / "configs/robotwin/data/robotwin_cap.yaml"))
    parser.add_argument("--tasks", default="click_alarmclock", help="comma-separated tasks, or all")
    parser.add_argument("--output", default=str(root / "configs/robotwin/data/cap_subtask.yaml"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output}; use --overwrite to regenerate")
    config = generate(args.source, None if args.tasks == "all" else args.tasks.split(","))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("# Derived from robotwin_cap.yaml; adds episode-aligned Tau0 conditioning.\n" + yaml.safe_dump(config, sort_keys=False))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
