"""Deterministic offline plumbing fixtures, not visual reasoning or task validation."""

import hashlib
import json
from pathlib import Path

import av
from PIL import Image, ImageDraw

from video_gen_qc.providers.base import VLM, ImageGenerator, VideoGenerator, VLMRequest


class MockVLM(VLM):
    provider_name = "mock"
    is_mock = True

    def __init__(self, model: str | None = "mock-vlm"):
        self.model = model

    def complete(self, request: VLMRequest) -> str:
        if request.purpose == "qc":
            check = {
                "status": "uncertain",
                "reason": "MOCK: No VLM inspection performed; this is an offline pipeline test.",
                "evidence_frames": [],
            }
            return json.dumps(
                {
                    "checks": dict.fromkeys(
                        ["task_compliance", "scene_consistency", "visual_anomalies"], check
                    )
                }
            )
        task = json.loads(request.text)["original_task"]
        if request.purpose == "image_prompt":
            constraints = "; ".join(task["initial_state_constraints"])
            scene = "; ".join(task["scene_constraints"])
            return (
                "MOCK prompt template (no model reasoning). Show a starting state before "
                "the requested action begins, never its completed outcome. "
                f"Initial-state requirements: {constraints or 'action has not started'}. "
                f"Scene: {scene or 'keep relevant objects visible'}. "
                f"Original task, for context only: {task['instruction']}"
            )
        initial = next(image for image in request.images if image.label == "initial_image")
        with Image.open(initial.path) as image:
            size = image.size
        digest = hashlib.sha256(initial.path.read_bytes()).hexdigest()
        return (
            "MOCK prompt template (no visual reasoning). Use the supplied actual initial "
            f"image ({size[0]}x{size[1]}, sha256={digest}) as the starting frame. "
            f"Original task remains authoritative: {task['instruction']} "
            f"Required event order: {'; '.join(task['required_events'])}. "
            f"Preserve: {'; '.join(task['scene_constraints'])}. "
            "The mock cannot identify or resolve image/task conflicts."
        )


class MockImageGenerator(ImageGenerator):
    def generate(
        self, prompt: str, output_path: Path, *, reference_image: Path | None = None
    ) -> Path:
        # Reference conditioning is intentionally not simulated by this offline fixture.
        image = Image.new("RGB", (640, 384), "#e8edf2")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 250, 640, 384), fill="#9aa9b5")
        draw.rectangle((250, 150, 370, 250), fill="#b67c47", outline="#6a452a", width=3)
        draw.rounded_rectangle((440, 185, 605, 220), radius=15, fill="#d7aa8b")
        draw.text((20, 20), "MOCK INITIAL IMAGE - synthetic fixture", fill="#15283b")
        draw.text(
            (20, 42), "Not a task-conditioned generation or a quality judgment", fill="#15283b"
        )
        draw.text(
            (20, 355),
            "prompt sha256: " + hashlib.sha256(prompt.encode()).hexdigest()[:16],
            fill="#15283b",
        )
        image.save(output_path, format="PNG")
        return output_path


class MockVideoGenerator(VideoGenerator):
    def generate(self, image_path: Path, prompt: str, output_path: Path) -> Path:
        with Image.open(image_path) as source:
            initial = source.convert("RGB")
        initial.thumbnail((640, 480))
        width, height = (max(2, value + value % 2) for value in initial.size)
        canvas = Image.new("RGB", (width, height), "white")
        canvas.paste(initial, (0, 0))
        with av.open(str(output_path), mode="w", format="mp4") as container:
            stream = container.add_stream("mpeg4", rate=8)
            stream.width, stream.height = width, height
            stream.pix_fmt = "yuv420p"
            for index in range(32):
                image = canvas.copy()
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, width, 20), fill="black")
                draw.text(
                    (5, 4), f"MOCK VIDEO / frame {index:02d} / no simulated task", fill="white"
                )
                frame = av.VideoFrame.from_image(image)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        return output_path
