"""Three entry modes sharing only artifact handling and independent inspection."""

import json
import shutil
from pathlib import Path

from video_gen_qc.artifacts import copy_image, new_run_dir, run_record, write_json
from video_gen_qc.config import AppConfig
from video_gen_qc.errors import InputError, ProviderError
from video_gen_qc.frame_sampler import sample_video
from video_gen_qc.prompts import IMAGE_PROMPT_SYSTEM, VIDEO_PROMPT_SYSTEM
from video_gen_qc.providers.base import VLM, ImageInput, VLMRequest
from video_gen_qc.providers.factory import (
    create_image_generator,
    create_video_generator,
    create_vlm,
)
from video_gen_qc.qc import judge
from video_gen_qc.schemas import TaskSpec, load_task


def _require_files(**paths: Path | None) -> dict[str, str]:
    inputs = {}
    for label, path in paths.items():
        if path is not None:
            if not path.is_file():
                raise InputError(f"Missing {label} file: {path}")
            inputs[label] = str(path.resolve())
    return inputs


def _design_prompt(vlm: VLM, request: VLMRequest, run_dir: Path) -> str:
    write_json(
        run_dir / f"{request.purpose}_request.json",
        {
            "purpose": request.purpose,
            "system": request.system,
            "payload": json.loads(request.text),
            "images": [{"label": image.label, "path": image.path.name} for image in request.images],
        },
    )
    prompt = vlm.complete(request)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ProviderError(f"VLM returned an empty {request.purpose}.")
    (run_dir / f"{request.purpose}.txt").write_text(prompt + "\n", encoding="utf-8")
    return prompt


def _inspect(
    task: TaskSpec,
    config: AppConfig,
    run_dir: Path,
    vlm: VLM,
    video: Path,
    initial: Path | None,
    reference: Path | None,
) -> str:
    sampling = sample_video(video, run_dir / "sampled_frames", config.qc.sample_frames)
    write_json(run_dir / "sampled_frames.json", sampling.model_dump())
    report = judge(task, sampling, run_dir, vlm, initial_image=initial, reference_image=reference)
    return report.decision


def run_generation(
    task_path: Path,
    config: AppConfig | None = None,
    *,
    initial_image: Path | None = None,
    reference_image: Path | None = None,
    output_root: Path | None = None,
    output_dir: Path | None = None,
    allow_paid: bool = False,
) -> Path:
    config = config or AppConfig()
    task = load_task(task_path)
    inputs = _require_files(
        task=task_path, initial_image=initial_image, reference_image=reference_image
    )
    # Preflight all ACTIVE providers before the first call. Skipped providers need no secrets.
    vlm = create_vlm(config.vlm, allow_paid=allow_paid)
    video_generator = create_video_generator(config.video_generation, allow_paid=allow_paid)
    image_generator = (
        create_image_generator(config.image_generation, allow_paid=allow_paid)
        if initial_image is None
        else None
    )
    run_dir = new_run_dir(output_root or Path(config.output.root), task.task_id, output_dir)
    mode = "existing_image" if initial_image is not None else "full"
    active = ["vlm", "video_generation"] + (["image_generation"] if image_generator else [])
    with run_record(
        run_dir, task.model_dump(), config.model_dump(), mode, inputs, active
    ) as record:
        reference = (
            copy_image(reference_image, run_dir / "reference_image.png")
            if reference_image
            else None
        )
        initial = run_dir / "initial_image.png"
        task_text = json.dumps({"original_task": task.model_dump()}, ensure_ascii=False)
        reference_inputs = (ImageInput("reference_image", reference),) if reference else ()
        if initial_image is not None:
            copy_image(initial_image, initial)
        else:
            prompt = _design_prompt(
                vlm,
                VLMRequest(
                    purpose="image_prompt",
                    system=IMAGE_PROMPT_SYSTEM,
                    text=task_text,
                    images=reference_inputs,
                ),
                run_dir,
            )
            generated = image_generator.generate(prompt, initial)
            if generated != initial or not initial.is_file():
                raise ProviderError("Image provider did not write the requested output file.")
            # Validate the actual image before using it in the next VLM call.
            copy_image(initial, initial)
        prompt = _design_prompt(
            vlm,
            VLMRequest(
                purpose="video_prompt",
                system=VIDEO_PROMPT_SYSTEM,
                text=task_text,
                images=(ImageInput("initial_image", initial),) + reference_inputs,
            ),
            run_dir,
        )
        video = run_dir / "video.mp4"
        generated_video = video_generator.generate(initial, prompt, video)
        if generated_video != video or not video.is_file():
            raise ProviderError("Video provider did not write the requested output file.")
        record["decision"] = _inspect(task, config, run_dir, vlm, video, initial, reference)
    return run_dir


def judge_video(
    task_path: Path,
    video: Path,
    config: AppConfig | None = None,
    *,
    initial_image: Path | None = None,
    reference_image: Path | None = None,
    output_root: Path | None = None,
    output_dir: Path | None = None,
    allow_paid: bool = False,
) -> Path:
    """QC-only never constructs or calls image/video generation providers."""
    config = config or AppConfig()
    task = load_task(task_path)
    inputs = _require_files(
        task=task_path, video=video, initial_image=initial_image, reference_image=reference_image
    )
    vlm = create_vlm(config.vlm, allow_paid=allow_paid)
    run_dir = new_run_dir(output_root or Path(config.output.root), task.task_id, output_dir)
    with run_record(
        run_dir, task.model_dump(), config.model_dump(), "qc_only", inputs, ["vlm"]
    ) as record:
        initial = (
            copy_image(initial_image, run_dir / "initial_image.png") if initial_image else None
        )
        reference = (
            copy_image(reference_image, run_dir / "reference_image.png")
            if reference_image
            else None
        )
        local_video = run_dir / ("video" + video.suffix.lower())
        shutil.copyfile(video, local_video)
        record["decision"] = _inspect(task, config, run_dir, vlm, local_video, initial, reference)
    return run_dir
