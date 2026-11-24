"""Episode/frame keyed planner cache, independent of loader splits, ordering and retries."""
import bisect
import hashlib
import json
from pathlib import Path

from .subtask_conditioning import SubtaskConditionError, normalize_subtask


def dataset_fingerprint(root):
    digest = hashlib.sha256()
    for name in ("info.json", "episodes.jsonl", "tasks.jsonl"):
        path = Path(root) / "meta" / name
        if not path.is_file():
            raise SubtaskConditionError(f"Missing LeRobot metadata: {path}")
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class SubtaskCache:
    def __init__(self, path):
        self.path = Path(path)
        doc = json.loads(self.path.read_text())
        if doc.get("schema_version") != 1 or not doc.get("datasets"):
            raise SubtaskConditionError(f"Invalid subtask cache schema: {self.path}")
        self.datasets = {}
        for dataset in doc["datasets"]:
            root = str(Path(dataset["root"]).resolve())
            if root in self.datasets:
                raise SubtaskConditionError(f"Duplicate cache dataset: {root}")
            if dataset.get("metadata_sha256") != dataset_fingerprint(root):
                raise SubtaskConditionError(f"Dataset metadata changed since proposal generation: {root}")
            episodes = {}
            for ep, record in dataset["episodes"].items():
                segments = record["segments"]
                cursor = 0
                for segment in segments:
                    a, b = segment["start"], segment["end"]
                    if type(a) is not int or type(b) is not int or a != cursor or b <= a:
                        raise SubtaskConditionError(f"Gap/overlap in {root} episode {ep}")
                    normalize_subtask(segment["subtask"], required=True)
                    cursor = b
                if cursor != record["length"] or cursor <= 0:
                    raise SubtaskConditionError(f"Incomplete coverage in {root} episode {ep}")
                episodes[int(ep)] = (record["length"], [s["start"] for s in segments], segments)
            self.datasets[root] = episodes

    def validate_selection(self, root, episodes):
        key = str(Path(root).resolve())
        available = self.datasets.get(key, {})
        metadata = {int(row["episode_index"]): int(row["length"])
                    for line in (Path(root) / "meta/episodes.jsonl").read_text().splitlines()
                    if line.strip() for row in [json.loads(line)]}
        for ep in episodes:
            if ep not in available or available[ep][0] != metadata[ep]:
                raise SubtaskConditionError(f"Missing/wrong-length subtask labels: {root}, episode {ep}")

    def lookup(self, root, episode, frame):
        try:
            length, starts, segments = self.datasets[str(Path(root).resolve())][int(episode)]
            if not 0 <= int(frame) < length:
                raise KeyError(frame)
            return segments[bisect.bisect_right(starts, int(frame)) - 1]["subtask"]
        except (KeyError, IndexError) as exc:
            raise SubtaskConditionError(f"No subtask for dataset={root}, episode={episode}, frame={frame}") from exc
