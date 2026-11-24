"""Portable observation manifests and episode-aligned Tau0 proposal caches."""
import hashlib
import json
from pathlib import Path

from .subtask_cache import SubtaskCache, dataset_fingerprint
from .subtask_conditioning import SubtaskConditionError, normalize_subtask


def read_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids) or not ids:
        raise SubtaskConditionError("Manifest IDs must be unique and the manifest non-empty")
    return rows


def build_cache(observations, predictions, output):
    provenance = {"observations_sha256": hashlib.sha256(Path(observations).read_bytes()).hexdigest(),
                  "predictions_sha256": hashlib.sha256(Path(predictions).read_bytes()).hexdigest()}
    observations, predictions = read_jsonl(observations), read_jsonl(predictions)
    predicted = {row["id"]: row for row in predictions}
    if set(predicted) != {row["id"] for row in observations}:
        raise SubtaskConditionError("Observation and prediction ID sets differ")
    datasets = {}
    for row in observations:
        result = predicted[row["id"]]
        if result.get("format_valid") is not True:
            raise SubtaskConditionError(f"Invalid Tau0 prediction: {row['id']}")
        subtask = normalize_subtask(result.get("subtask"), required=True)
        root = str(Path(row["dataset_root"]).resolve())
        if root not in datasets:
            datasets[root] = {"root": root, "metadata_sha256": dataset_fingerprint(root), "episodes": {}}
        dataset = datasets[root]
        if row.get("metadata_sha256") != dataset["metadata_sha256"]:
            raise SubtaskConditionError("Dataset changed after exporting the camera observations")
        ep = str(int(row["episode_index"]))
        record = dataset["episodes"].setdefault(ep, {"length": row["episode_length"], "segments": []})
        if record["length"] != row["episode_length"]:
            raise SubtaskConditionError(f"Conflicting lengths for episode {ep}")
        record["segments"].append({"start": row["frame_start"], "end": row["frame_end"],
                                   "subtask": subtask, "proposal_id": row["id"]})
    for dataset in datasets.values():
        for record in dataset["episodes"].values():
            record["segments"].sort(key=lambda segment: segment["start"])
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance["planner_models"] = sorted({row["planner_model"] for row in predictions if row.get("planner_model")})
    doc = {"schema_version": 1, "planner": "tau0_proposal", "conditioning": "subtask_text_only", "provenance": provenance,
           "datasets": list(datasets.values())}
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(doc, handle, ensure_ascii=False, indent=2)
        cache = SubtaskCache(temporary)  # Reject gaps, overlaps and stale metadata before publishing.
        for dataset in doc["datasets"]:
            cache.validate_selection(dataset["root"], [int(ep) for ep in dataset["episodes"]])
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return doc


def export_observations(dataset_root, output_dir, *, stride=24, episodes=None):
    """LeRobot v2.x: extract ONLY each interval's current observation, never future frames.

    Cache keys retain original episode/frame identities, so selecting/reordering training
    episodes cannot silently move a subtask label onto a different trajectory.
    """
    import av
    from PIL import Image

    if stride <= 0:
        raise ValueError("stride must be positive")
    root, out = Path(dataset_root).resolve(), Path(output_dir).resolve()
    manifest = out / "observations.jsonl"
    if manifest.exists():
        raise FileExistsError(manifest)
    info = json.loads((root / "meta/info.json").read_text())
    all_episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    selected = None if episodes is None else {int(ep) for ep in episodes}
    if selected is not None and selected - {int(ep["episode_index"]) for ep in all_episodes}:
        raise ValueError("Selected episode does not exist")
    fingerprint = dataset_fingerprint(root)
    chunk_size = int(info.get("chunks_size", 1000))
    video_template = info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    camera_keys = {"head": "observation.images.cam_high", "left": "observation.images.cam_left_wrist",
                   "right": "observation.images.cam_right_wrist"}
    missing = set(camera_keys.values()) - set(info["features"])
    if missing:
        raise ValueError(f"Expected the three RoboTwin camera keys, missing: {sorted(missing)}")
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    with manifest.open("x") as handle:
        for episode in sorted(all_episodes, key=lambda ep: int(ep["episode_index"])):
            ep, length = int(episode["episode_index"]), int(episode["length"])
            if selected is not None and ep not in selected:
                continue
            if length <= 0 or not episode.get("tasks"):
                raise ValueError(f"Episode {ep} has no frames or instruction")
            starts = list(range(0, length, stride))
            image_paths = {frame: {} for frame in starts}
            for view, key in camera_keys.items():
                video = root / video_template.format(episode_chunk=ep // chunk_size, video_key=key, episode_index=ep)
                remaining = set(starts)
                with av.open(str(video)) as container:
                    for frame_index, frame in enumerate(container.decode(video=0)):
                        if frame_index in remaining:
                            image_path = out / "images" / f"ep{ep:06d}_f{frame_index:06d}_{view}.png"
                            image_path.parent.mkdir(parents=True, exist_ok=True)
                            if image_path.exists():
                                raise FileExistsError(image_path)
                            Image.fromarray(frame.to_ndarray(format="rgb24")).save(image_path)
                            image_paths[frame_index][view] = str(image_path)
                            remaining.remove(frame_index)
                        if not remaining:
                            break
                if remaining:
                    raise ValueError(f"Video is shorter than metadata: {video}")
            prefix = hashlib.sha256(str(root).encode()).hexdigest()[:12]
            for start in starts:
                row = {"id": f"{prefix}:ep{ep}:f{start}", "instruction": episode["tasks"][0],
                       "task_type": "full_qa", "memory": "(empty)", "images": image_paths[start],
                       "dataset_root": str(root), "metadata_sha256": fingerprint,
                       "episode_index": ep, "episode_length": length,
                       "frame_start": start, "frame_end": min(start + stride, length)}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    return manifest, count
