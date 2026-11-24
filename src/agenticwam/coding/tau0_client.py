"""HTTP boundary isolates Tau0's Transformers 5 environment from AgenticWAM's 4.x."""
import base64
import io
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from .subtask_conditioning import SubtaskConditionError, normalize_subtask


CAMERAS = {"head": "head_camera", "left": "left_camera", "right": "right_camera"}


def encode_image(image):
    if isinstance(image, (str, Path)):
        with Image.open(image) as loaded:
            image = loaded.convert("RGB")
    elif not isinstance(image, Image.Image):
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
            raise SubtaskConditionError("Proposal images must be HWC uint8 RGB")
        image = Image.fromarray(array)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class ProposalClient:
    def __init__(self, url="http://127.0.0.1:10089/predict", timeout_s=120, task_type="full_qa"):
        if task_type not in ("full_qa", "subtask_only"):
            raise ValueError("task_type must be full_qa or subtask_only")
        self.url, self.timeout_s, self.task_type = url, float(timeout_s), task_type
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.reset()

    @classmethod
    def from_config(cls, path):
        import yaml
        config = yaml.safe_load(Path(path).read_text())
        return cls(**{key: config[key] for key in ("url", "timeout_s", "task_type") if key in config})

    def reset(self):
        self.memory = "(empty)"
        self.last_result = None

    def health(self):
        try:
            with self.opener.open(self.url.rsplit("/", 1)[0] + "/health", timeout=self.timeout_s) as response:
                result = json.load(response)
            if result.get("ready") is not True:
                raise ValueError("planner is not ready")
            return result
        except Exception as exc:
            raise SubtaskConditionError(f"Tau0 Proposal health check failed: {exc}") from exc

    def predict(self, instruction, images):
        if not isinstance(instruction, str) or not instruction.strip():
            raise SubtaskConditionError("Proposal requires the original task instruction")
        if set(images) != set(CAMERAS):
            raise SubtaskConditionError("Proposal requires exactly head, left and right camera views")
        memory_in = self.memory
        payload = {"instruction": instruction, "memory": memory_in, "task_type": self.task_type,
                   "images": {view: encode_image(images[view]) for view in CAMERAS}}
        request = urllib.request.Request(self.url, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        started = time.perf_counter()
        try:
            with self.opener.open(request, timeout=self.timeout_s) as response:
                result = json.load(response)
        except Exception as exc:
            raise SubtaskConditionError(f"Tau0 Proposal request failed: {type(exc).__name__}: {exc}") from exc
        if not isinstance(result, dict) or result.get("format_valid") is not True:
            raise SubtaskConditionError("Tau0 returned an invalid proposal; refusing unconditioned inference")
        subtask = normalize_subtask(result.get("subtask"), required=True)
        next_memory = result.get("next_memory", memory_in)
        if not isinstance(next_memory, str):
            raise SubtaskConditionError("Tau0 next_memory must be a string")
        self.memory = next_memory.strip() or "(empty)"
        self.last_result = {**result, "subtask": subtask, "memory_in": memory_in,
                            "next_memory": self.memory,
                            "client_latency_ms": round((time.perf_counter() - started) * 1000, 2)}
        return self.last_result

    def predict_robotwin(self, instruction, observation):
        return self.predict(instruction, {view: observation["observation"][camera]["rgb"]
                                          for view, camera in CAMERAS.items()})
