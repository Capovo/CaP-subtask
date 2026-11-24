import base64
import io
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from agenticwam.coding.subtask_cache import SubtaskCache, dataset_fingerprint
from agenticwam.coding.subtask_conditioning import SubtaskConditionError, condition_code_prompt
from agenticwam.coding.subtask_data import build_cache, export_observations
from agenticwam.coding.tau0_client import ProposalClient
from agenticwam.models.experts.code_expert import ByteTokenizer, _build_mock_code_expert
from agenticwam.models.cosmos_mot.agenticwam_model import AgenticWAMCosmos

torch.set_num_threads(1)


def metadata(root, lengths):
    meta = root / "meta"
    meta.mkdir(parents=True)
    info = {"fps": 10, "chunks_size": 1000,
            "video_path": "videos/{video_key}/episode_{episode_index:06d}.mkv",
            "features": {"observation.images." + key: {} for key in ("cam_high", "cam_left_wrist", "cam_right_wrist")}}
    (meta / "info.json").write_text(json.dumps(info))
    (meta / "episodes.jsonl").write_text("".join(json.dumps({"episode_index": ep, "length": length, "tasks": ["Move the cube"]}) + "\n"
                                                  for ep, length in lengths.items()))
    (meta / "tasks.jsonl").write_text('{"task_index":0,"task":"Move the cube"}\n')


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class PromptAndModelTests(unittest.TestCase):
    def test_factory_enables_the_condition_on_model_and_expert(self):
        from agenticwam.models.cosmos_mot.build import create_agenticwam_cosmos
        fake = SimpleNamespace(code_expert=self.expert())
        with patch("agenticwam.models.cosmos_mot.build.build_agenticwam_cosmos", return_value=fake):
            result = create_agenticwam_cosmos(use_code_expert=True, code_objective="joint",
                                              high_level_subtask_conditioning=True, code_vision="none")
        self.assertTrue(result.require_high_level_subtask)
        self.assertTrue(result.code_expert.require_high_level_subtask)
        with self.assertRaises(ValueError):
            create_agenticwam_cosmos(use_code_expert=True, code_objective="ar", high_level_subtask_conditioning=True)
        with patch.dict(os.environ, {"AGENTICWAM_FAST_JOINT": "1"}), self.assertRaises(ValueError):
            create_agenticwam_cosmos(use_code_expert=True, code_objective="joint", high_level_subtask_conditioning=True)

    def test_checkpoint_records_and_enforces_conditioning(self):
        class Tiny(AgenticWAMCosmos):
            def __init__(self, required):
                torch.nn.Module.__init__(self)
                self.weight = torch.nn.Parameter(torch.tensor([3.0]))
                self.require_high_level_subtask = required

            def _trainable_state_dict(self):
                return self.state_dict()

            def extra_trainable_modules(self):
                return []

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            Tiny(True).save_checkpoint(path, step=12)
            record = torch.load(path, weights_only=True)
            self.assertEqual(record["conditioning"]["kind"], "tau0_subtask_text_v1")
            with self.assertRaises(SubtaskConditionError):
                Tiny(False).load_checkpoint(path)
            restored = Tiny(True)
            restored.load_checkpoint(path)
            self.assertEqual(float(restored.weight.detach()[0]), 3.0)

    def expert(self):
        torch.manual_seed(7)
        return _build_mock_code_expert(1, 2, 16, 32, max_code_tokens=768)

    def test_condition_is_not_a_label_and_changes_real_code_logits(self):
        expert = self.expert()
        expert.require_high_level_subtask = True
        prompt = "# Instruction: Move cube\n# Helper library:\ndef reach(hand): ...\n# Next:\n"
        code = "reach('left')"
        ids, labels = expert.build_code_batch([prompt], [code], "cpu", high_level_subtasks=["Use left arm"])
        prefix_length = int((labels[0] == -100).sum())
        expected = condition_code_prompt(prompt, "Use left arm")
        self.assertEqual(bytes(ids[0, :prefix_length].tolist()).decode(), expected)
        self.assertTrue(torch.all(labels[0, :prefix_length] == -100))
        self.assertEqual(labels[0, prefix_length].item(), ord('r'))
        changed, _ = expert.build_code_batch([prompt], [code], "cpu", high_level_subtasks=["Use right arm"])
        logits = expert._forward_code_only(ids)
        changed_logits = expert._forward_code_only(changed)
        self.assertFalse(torch.allclose(logits[:, -2], changed_logits[:, -2]))
        loss = expert.code_loss(logits, labels)["loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(expert.embed_tokens.weight.grad.abs().sum()), 0)

    def test_joint_noise_never_masks_the_condition(self):
        expert = self.expert()
        expert.code_objective, expert.mask_token_id = "joint", 257
        expert.require_high_level_subtask = True
        ids, labels = expert.build_code_batch(["Move cube"], ["reach('left')"], "cpu", high_level_subtasks=["Use left arm"])
        prefix = ByteTokenizer().encode(condition_code_prompt("# Instruction: Move cube\n", "Use left arm"))
        self.assertEqual(ids[0, :len(prefix)].tolist(), prefix)
        self.assertTrue(torch.all(labels[0, :len(prefix)] == -100))
        self.assertGreater(int((labels != -100).sum()), 0)

    def test_missing_condition_and_prompt_overflow_fail(self):
        expert = self.expert()
        expert.require_high_level_subtask = True
        with self.assertRaises(SubtaskConditionError):
            expert.build_code_batch(["Move"], ["reach()"], "cpu")
        with self.assertRaises(SubtaskConditionError):
            expert.build_code_batch(["Move"], ["reach()"], "cpu", max_length=12, high_level_subtasks=["Use left"])
        with self.assertRaises(SubtaskConditionError):
            condition_code_prompt("Move", "x" * 200, tokenizer=ByteTokenizer())
        self.assertEqual(condition_code_prompt("original\n", None), "original\n")

    def test_actual_joint_inference_uses_identical_conditioned_prompt(self):
        class CapturedPrompt(Exception):
            pass

        expert = self.expert()
        instruction, subtask = "Move cube", "Use left arm"
        preamble = "def reach(hand): ...\n# Setup defines: hand\n"
        training = f"# Instruction: {instruction}\n# Helper library:\n{preamble}# Next:\n"
        ids, labels = expert.build_code_batch([training], ["reach(hand)"], "cpu", high_level_subtasks=[subtask])
        expected = bytes(ids[0, :(labels[0] == -100).sum()].tolist()).decode()
        captured = []

        class Tokenizer(ByteTokenizer):
            def encode(self, text, **kwargs):
                if text.startswith("# Instruction:"):
                    captured.append(text)
                    raise CapturedPrompt()
                return super().encode(text)

        expert.tokenizer = Tokenizer()
        fake = SimpleNamespace(dit=torch.nn.Identity(), device=torch.device("cpu"),
                               action_expert=torch.nn.Linear(1, 1), n_cameras=3,
                               code_expert=expert, require_high_level_subtask=True,
                               cond_timestep=0.1, subtask_max_tokens=128,
                               _encode_video_latents=lambda x: torch.zeros(1, 1, 3, 1, 1),
                               _vit_prefix=lambda x: None, _pixel_to_latent=lambda x: 1,
                               _action_context=lambda p, q: (p, None))
        fake.action_expert.action_dim = 14
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preamble.txt"
            path.write_text(preamble)
            with patch.dict(os.environ, {"AGENTICWAM_CODE_PREAMBLE_FILE": str(path), "AGENTICWAM_CODE_NO_HISTORY": "1"}):
                with self.assertRaises(CapturedPrompt):
                    AgenticWAMCosmos._infer_action_joint(fake, torch.zeros(1), None, torch.zeros(1, 1, 1), 32,
                                                        instruction=instruction, high_level_subtask=subtask)
        self.assertEqual(captured, [expected])


class CacheAndVideoTests(unittest.TestCase):
    def test_loader_preserves_identity_after_processor_and_random_retry(self):
        from agenticwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
        class Multi:
            num_frames = 2

            def __len__(self):
                return 2

            def __getitem__(self, index):
                if index == 0:
                    raise OSError("fixture: unreadable frame")
                return {"task": "Move cube", "action": torch.zeros(32, 14),
                        "observation.state": torch.zeros(33, 14),
                        "observation.images.cam_high": torch.zeros(33, 3, 16, 16),
                        "action_is_pad": torch.zeros(32, dtype=torch.bool),
                        "observation.state_is_pad": torch.zeros(33, dtype=torch.bool),
                        "observation.images.cam_high_is_pad": torch.zeros(33, dtype=torch.bool),
                        "dataset_index": torch.tensor(1), "episode_index": torch.tensor(9),
                        "frame_index": torch.tensor(2)}

        base = BaseLerobotDataset.__new__(BaseLerobotDataset)
        base.multi_dataset = Multi()
        base.action_meta = [{"key": "default", "lerobot_key": "action", "raw_shape": 14}]
        base.state_meta = [{"key": "default", "lerobot_key": "observation.state", "raw_shape": 14}]
        base.image_meta = [{"key": "cam_high", "lerobot_key": "observation.images.cam_high", "raw_shape": [3, 16, 16]}]
        base.processor = SimpleNamespace(preprocess=lambda data: {"idx": data["idx"]})
        with patch("numpy.random.randint", return_value=1), patch("traceback.format_exc", return_value="expected fixture retry"):
            sample = base[0]
        self.assertEqual(sample["idx"], 1)
        self.assertEqual(int(sample["dataset_index"]), 1)
        self.assertEqual(int(sample["episode_index"]), 9)
        self.assertEqual(int(sample["frame_index"]), 2)

    def test_cache_handles_nonzero_reordered_episodes_and_rejects_staleness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            metadata(root, {5: 4, 9: 4})
            observations, predictions = [], []
            for ep in (9, 5):
                for start in (0, 2):
                    key = f"{ep}:{start}"
                    observations.append({"id": key, "dataset_root": str(root), "metadata_sha256": dataset_fingerprint(root),
                                         "episode_index": ep, "episode_length": 4, "frame_start": start, "frame_end": start + 2})
                    predictions.append({"id": key, "subtask": f"ep{ep} frame{start}", "format_valid": True})
            a, b, output = Path(directory) / "observations.jsonl", Path(directory) / "predictions.jsonl", Path(directory) / "cache.json"
            write_jsonl(a, observations)
            write_jsonl(b, list(reversed(predictions)))
            build_cache(a, b, output)
            cache = SubtaskCache(output)
            cache.validate_selection(root, [9, 5])
            self.assertEqual(cache.lookup(root, 5, 3), "ep5 frame2")
            self.assertEqual(cache.lookup(root, 9, 0), "ep9 frame0")
            with self.assertRaises(SubtaskConditionError):
                cache.lookup(root, 5, 4)
            with (root / "meta/tasks.jsonl").open("a") as handle:
                handle.write('\n')
            with self.assertRaises(SubtaskConditionError):
                SubtaskCache(output)

    def test_cache_rejects_gap_and_invalid_proposal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            metadata(root, {0: 4})
            a, b, output = Path(directory) / "a.jsonl", Path(directory) / "b.jsonl", Path(directory) / "out.json"
            write_jsonl(a, [{"id": "0", "dataset_root": str(root), "metadata_sha256": dataset_fingerprint(root),
                             "episode_index": 0, "episode_length": 4, "frame_start": 1, "frame_end": 4}])
            write_jsonl(b, [{"id": "0", "subtask": "Pick cube", "format_valid": True}])
            with self.assertRaises(SubtaskConditionError):
                build_cache(a, b, output)
            self.assertFalse(output.exists())
            write_jsonl(b, [{"id": "0", "subtask": "Pick cube", "format_valid": False}])
            with self.assertRaises(SubtaskConditionError):
                build_cache(a, b, output)

    def test_export_reads_current_frames_from_real_three_camera_videos(self):
        import av
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            metadata(root, {5: 4})
            for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                video = root / f"videos/observation.images.{camera}/episode_000005.mkv"
                video.parent.mkdir(parents=True, exist_ok=True)
                with av.open(str(video), "w") as container:
                    stream = container.add_stream("ffv1", rate=10)
                    stream.width, stream.height, stream.pix_fmt = 16, 16, "bgr0"
                    for index in range(4):
                        frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), 20 + index * 30, dtype=np.uint8), format="rgb24")
                        for packet in stream.encode(frame):
                            container.mux(packet)
                    for packet in stream.encode():
                        container.mux(packet)
            manifest, count = export_observations(root, Path(directory) / "export", stride=2)
            rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            self.assertEqual(count, 2)
            self.assertEqual([row["frame_start"] for row in rows], [0, 2])
            self.assertEqual(Image.open(rows[0]["images"]["head"]).getpixel((0, 0)), (20, 20, 20))
            self.assertEqual(Image.open(rows[1]["images"]["right"]).getpixel((0, 0)), (80, 80, 80))


class HttpPlannerTests(unittest.TestCase):
    def test_http_contract_memory_reset_and_invalid_format(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"ready":true,"contract":"tau0_proposal_v1"}')

            def do_POST(self):
                row = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(row)
                result = {"format_valid": row["instruction"] != "invalid", "subtask": "Pick cube with left arm",
                          "next_memory": "The cube is being picked"}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = ProposalClient(f"http://127.0.0.1:{server.server_port}/predict")
            self.assertTrue(client.health()["ready"])
            images = {view: np.full((16, 16, 3), color, dtype=np.uint8)
                      for view, color in (("head", 20), ("left", 40), ("right", 60))}
            self.assertEqual(client.predict("Move cube", images)["subtask"], "Pick cube with left arm")
            client.predict("Move cube", images)
            self.assertEqual(requests[0]["memory"], "(empty)")
            self.assertEqual(requests[1]["memory"], "The cube is being picked")
            client.reset()
            client.predict("Move cube", images)
            self.assertEqual(requests[-1]["memory"], "(empty)")
            pixel = Image.open(io.BytesIO(base64.b64decode(requests[0]["images"]["left"]))).getpixel((0, 0))
            self.assertEqual(pixel, (40, 40, 40))
            with self.assertRaises(SubtaskConditionError):
                client.predict("invalid", images)
            self.assertEqual(client.memory, "The cube is being picked")
        finally:
            server.shutdown()
            thread.join()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
