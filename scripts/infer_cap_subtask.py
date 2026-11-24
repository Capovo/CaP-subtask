#!/usr/bin/env python3
"""One observation -> Tau0 subtask -> joint AgenticWAM inference (no robot motion)."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset-stats", default="/home/yilianli/project/models/agentic-wam-robotwin-release/dataset_stats.json")
    parser.add_argument("--proposal-json", type=Path, help="use a saved, valid Tau0 output instead of HTTP")
    parser.add_argument("--planner-config", default=str(ROOT / "configs/subtask/planner.yaml"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not Path(args.ckpt).is_file():
        raise FileNotFoundError(args.ckpt)
    row = json.loads(args.input.read_text())
    state = np.asarray(row["state"], dtype=np.float32)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError("state must contain 14 finite absolute joint positions")
    observation = {"observation": {}, "joint_action": {"vector": state}}
    for view, camera in {"head": "head_camera", "left": "left_camera", "right": "right_camera"}.items():
        path = args.input.resolve().parent / row["images"][view]
        with Image.open(path) as image:
            observation["observation"][camera] = {"rgb": np.asarray(image.convert("RGB"))}
    if args.proposal_json:
        from agenticwam.coding.subtask_conditioning import normalize_subtask
        proposal = json.loads(args.proposal_json.read_text())
        if not row.get("id") or proposal.get("id") != row["id"]:
            raise ValueError("Saved proposal id must match the input observation id")
        if proposal.get("format_valid") is not True:
            raise ValueError("Saved Tau0 prediction is not format_valid")
        observation["high_level_subtask"] = normalize_subtask(proposal.get("subtask"), required=True)
    os.environ["CAP_SUBTASK_PLANNER_CONFIG"] = args.planner_config
    os.environ["AGENTICWAM_HYBRID"] = "0"  # This tool generates actions; it executes no code/motion.
    os.environ["AGENTICWAM_CODE_PREAMBLE_FILE"] = str(ROOT / "configs/robotwin/tasks_indep" / row["task"] / "preamble.txt")
    os.environ["AGENTICWAM_CODE_NO_HISTORY"] = "1"
    sys.path.insert(0, str(ROOT / "robotwin/deploy"))
    from deploy_policy import get_model
    policy = get_model({"sim_task": "cap_subtask_robotwin", "sim_paths_overrides": "paths=local",
                        "use_code_expert": True, "ckpt_setting": args.ckpt,
                        "dataset_stats_path": args.dataset_stats, "device": args.device,
                        "num_inference_steps": args.steps})
    actions = policy._infer_action_chunk(observation, row["instruction"])
    result = {"instruction": row["instruction"], "high_level_subtask": policy._last_subtask,
              "code": policy._last_code, "action_chunk": actions.tolist(),
              "action_type": "qpos_abs", "action_dim": 14,
              "conditioning": "tau0_subtask_text_v1"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
