#!/usr/bin/env python3
"""Reconstructed TI2TIS joint inference for Omni-StoryBench.

This is an independently written sampler for snu-aidas/Dynin-Omni_nips.
It is not the original paper inference code and does not establish score parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms
from transformers import AutoConfig, AutoModel, AutoTokenizer


MODEL_REVISION = "04799bddcf17e666976f6894516a4a34b85b12ad"
IMAGE_REVISION = "5a869517cbbda6ac4de8b438fc59b7f053cb4238"
SPEECH_REVISION = "1062a384c4950fcda60be38fc0fc3f169acc7e77"
TASK_ID, SOI_ID, EOI_ID, SOA_ID, EOA_ID = 126107, 126084, 126085, 126097, 126098
IMAGE_CODES, SPEECH_CODES = 8192, 4096
PROMPT_LENGTH, IMAGE_RESOLUTION = 1024, 336
EXPECTED_COUNTS = {"africanstorybook": 54, "digitallibrary": 293, "storybookscanda": 7, "storyweaver": 546}
SENTINELS = {
    0: "africanstorybook/asb10179/page_006__page_007",
    724: "africanstorybook/asb10474/page_006__page_007",
    757: "africanstorybook/asb21992/page_003__page_004",
    758: "africanstorybook/asb31833/page_009__page_010",
    899: "storyweaver/515152-rosie-the-caterpillar/page_007__page_008",
}
PROMPT_COLUMNS = (
    "id", "source", "book", "from_key", "to_key", "current_image", "current_text",
    "genre", "topic", "style", "narrative_tense", "narrative_perspective",
    "book_characters", "condition_characters", "scene", "text_instruction", "ambient_sound",
)
OUTPUTS = {"text": "generated.txt", "image": "generated.png", "speech": "generated.wav"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_revision(path: Path) -> str:
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def repo_load_kwargs(source: str, revision: str, local_only: bool) -> dict:
    options = {"local_files_only": local_only}
    if not Path(source).is_dir():
        options["revision"] = revision
    return options


def load_rows(dataset_root: Path) -> tuple[list[dict], str]:
    parquet_path = dataset_root / "data" / "omni_storybench.parquet"
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)
    parquet = pq.ParquetFile(parquet_path)
    missing = set(PROMPT_COLUMNS) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"missing prompt columns: {sorted(missing)}")
    rows = parquet.read(columns=PROMPT_COLUMNS).to_pylist()
    if len(rows) != 900 or len({row["id"] for row in rows}) != 900:
        raise ValueError("expected 900 unique transitions in canonical Parquet order")
    for index, expected in SENTINELS.items():
        if rows[index]["id"] != expected:
            raise ValueError(f"row-order mismatch at {index}: {rows[index]['id']!r}")
    if dict(Counter(row["source"] for row in rows)) != EXPECTED_COUNTS:
        raise ValueError("unexpected source distribution")
    return rows, sha256(parquet_path)


def current_image_path(dataset_root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError("current_image must be a nonempty relative path")
    root = dataset_root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or unsafe current image: {relative!r}")
    return path


def make_prompt(row: dict) -> str:
    def parse_characters(value):
        return json.loads(value) if isinstance(value, str) else value

    metadata = {key: row[key] for key in ("genre", "topic", "style", "narrative_tense", "narrative_perspective")}
    metadata["characters"] = parse_characters(row["book_characters"])
    condition = {
        "characters": parse_characters(row["condition_characters"]),
        "scene": row["scene"],
        "text content": row["text_instruction"],
        "ambient_sound": row["ambient_sound"],
    }
    return (
        "You are an AI storybook next-page generator. "
        "Given the previous page text and image, and the next-page condition below, "
        "generate the next page text, image, and speech.\n"
        "Requirements:\n"
        "- Next page text should be coherent and consistent with the story, while keeping concise.\n"
        "- Next page image should match the described scene and characters.\n"
        "- Text must contain exactly one dialogue quote.\n"
        "- The provided condition may include scene, characters, actions, emotions, and ambient sound.\n\n"
        f"Story metadata:\n- {json.dumps(metadata, ensure_ascii=False)}\n\n"
        f"Previous page text:\n- {row['current_text']}\n\n"
        f"Next page condition:\n- {json.dumps(condition, ensure_ascii=False)}"
    )


def build_sequence(row, image_path, image_codec, tokenizer, device, text_length, speech_length, mask_id):
    transform = transforms.Compose([
        transforms.Resize(IMAGE_RESOLUTION, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((IMAGE_RESOLUTION, IMAGE_RESOLUTION)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    with Image.open(image_path) as image:
        pixels = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        source = image_codec.get_code(pixels)[0].long() + len(tokenizer)
    if source.numel() != 441:
        raise ValueError(f"expected 441 source image codes at 336 px, found {source.numel()}")
    prompt = [tokenizer.bos_token_id]
    prompt += tokenizer.encode(make_prompt(row), add_special_tokens=False)
    prompt += [tokenizer.eos_token_id]
    if len(prompt) > PROMPT_LENGTH:
        raise ValueError(f"prompt exceeds {PROMPT_LENGTH} training tokens: {len(prompt)}")
    padding = PROMPT_LENGTH - len(prompt)
    ids = [TASK_ID, SOI_ID] + source.tolist() + [EOI_ID]
    prompt_start = len(ids)
    ids += [tokenizer.pad_token_id] * padding + prompt + [SOI_ID]
    image_start = len(ids)
    ids += [mask_id] * source.numel()
    image_end = len(ids)
    ids += [EOI_ID, tokenizer.bos_token_id]
    text_start = len(ids)
    ids += [mask_id] * (text_length - 1)
    text_end = len(ids)
    ids += [SOA_ID]
    speech_start = len(ids)
    ids += [mask_id] * (speech_length - 1)
    speech_end = len(ids)
    spans = {"image": (image_start, image_end), "text": (text_start, text_end), "speech": (speech_start, speech_end)}
    for field in ("emotion", "speed", "pitch", "gender"):
        ids += tokenizer.encode(f"{field}: ", add_special_tokens=False)
        spans[field] = (len(ids), len(ids) + 1)
        ids.append(mask_id)
        ids += tokenizer.encode("\n", add_special_tokens=False)
    if len(ids) > 4096:
        raise ValueError(f"sequence exceeds model context: {len(ids)}")
    attention = torch.ones(len(ids), dtype=torch.bool, device=device)
    attention[prompt_start:prompt_start + padding] = False
    return torch.tensor(ids, dtype=torch.long, device=device)[None], spans, attention[None]


@torch.inference_mode()
def sample_joint(model, sequence, spans, attention, tokenizer, steps, mask_id):
    """Update all three target modalities using the same model forward per step."""
    device = sequence.device
    image_start = len(tokenizer)
    speech_start = image_start + IMAGE_CODES
    choices = {
        "image": torch.arange(image_start, image_start + IMAGE_CODES, device=device),
        "speech": torch.cat((torch.arange(speech_start, speech_start + SPEECH_CODES, device=device), torch.tensor([EOA_ID, tokenizer.eos_token_id], device=device))),
        "text": torch.arange(len(tokenizer), device=device),
    }
    positions = torch.arange(sequence.shape[1], device=device)
    bias = (attention[:, :, None] & attention[:, None, :]).unsqueeze(1)
    for step in range(steps):
        masked = sequence.eq(mask_id)[0]
        if not masked.any():
            break
        logits = model(sequence, attention_bias=bias).logits[0]
        for name, (start, end) in spans.items():
            locations = (masked & (positions >= start) & (positions < end)).nonzero(as_tuple=True)[0]
            if locations.numel() == 0:
                continue
            allowed = choices[name] if name in ("image", "speech") else choices["text"]
            probability = logits[locations][:, allowed].float().softmax(-1)
            if name in ("text", "image"):
                sampled = torch.multinomial(probability, 1).squeeze(-1)
                confidence = probability.gather(-1, sampled[:, None]).squeeze(-1)
            else:
                confidence, sampled = probability.max(-1)
            count = (locations.numel() + steps - step - 1) // (steps - step)
            selected = confidence.topk(count).indices
            sequence[0, locations[selected]] = allowed[sampled[selected]]
        if (step + 1) % 16 == 0 or step + 1 == steps:
            print(f"step={step + 1}/{steps} remaining={int(sequence.eq(mask_id).sum())}", flush=True)
    if sequence.eq(mask_id).any():
        raise RuntimeError("joint sampling left masked tokens")
    return sequence


def resolve_codec_weights(source: str, revision: str, local_only: bool) -> Path:
    local = Path(source)
    if local.is_dir():
        return local / "model.safetensors"
    return Path(hf_hub_download(repo_id=source, filename="model.safetensors", revision=revision, local_files_only=local_only))


def load_speech_codec(source: str, revision: str, runtime_repo: str, device, local_only: bool):
    options = repo_load_kwargs(source, revision, local_only)
    config = AutoConfig.from_pretrained(source, trust_remote_code=True, **options)
    config.runtime_repo_id = runtime_repo
    codec = AutoModel.from_config(config, trust_remote_code=True)
    raw = load_file(str(resolve_codec_weights(source, revision, local_only)), device="cpu")
    state = {
        key.replace(".weight_g", ".parametrizations.weight.original0").replace(".weight_v", ".parametrizations.weight.original1"): value
        for key, value in raw.items()
    }
    missing, unexpected = codec.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"speech decoder weights do not match: missing={missing[:5]}, unexpected={unexpected[:5]}")
    print(f"speech decoder: loaded {len(state)} tensors with no missing/unexpected keys", flush=True)
    del raw, state
    return codec.to(device).eval()


def decode_to_stage(stage: Path, sequence, spans, tokenizer, image_codec, speech_codec, speech_style: str) -> dict:
    stage.mkdir(parents=True, exist_ok=False)
    text_ids = sequence[0, slice(*spans["text"])].tolist()
    if tokenizer.eos_token_id in text_ids:
        text_ids = text_ids[:text_ids.index(tokenizer.eos_token_id)]
    story = tokenizer.decode(text_ids, skip_special_tokens=True).strip()
    if not story:
        raise ValueError("generated text is blank")
    (stage / "generated.txt").write_text(story + "\n", encoding="utf-8")
    image_codes = sequence[0, slice(*spans["image"])] - len(tokenizer)
    pixels = image_codec.decode_code(image_codes[None]).clamp(-1, 1)
    pixels = ((pixels[0] + 1) * 127.5).permute(1, 2, 0).byte().cpu().numpy()
    Image.fromarray(pixels).save(stage / "generated.png")
    speech_base = len(tokenizer) + IMAGE_CODES
    speech_codes = []
    for token_id in sequence[0, slice(*spans["speech"])].tolist():
        if token_id in (EOA_ID, tokenizer.eos_token_id):
            break
        if speech_base <= token_id < speech_base + SPEECH_CODES:
            speech_codes.append(token_id - speech_base)
    if len(speech_codes) < 8:
        raise ValueError(f"speech ended after {len(speech_codes)} codes")
    units = "".join(f"<|speech_{code}|>" for code in speech_codes)
    speech_codec.decode(units, condition=speech_style, output_wav_file=str(stage / "generated.wav"))
    return {"text": story, "speech_codes": len(speech_codes), "image_size": list(pixels.shape[:2])}


def validate_triplet(paths: dict[str, Path]) -> dict:
    for name, path in paths.items():
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing, empty, or symlinked {name} output: {path}")
    text = paths["text"].read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("blank text output")
    with Image.open(paths["image"]) as image:
        if image.format != "PNG" or image.size != (IMAGE_RESOLUTION, IMAGE_RESOLUTION):
            raise ValueError(f"unexpected image format/size: {image.format}, {image.size}")
        image.verify()
    audio, rate = sf.read(paths["speech"])
    if rate <= 0 or len(audio) / rate < 0.1 or not np.isfinite(audio).all():
        raise ValueError("invalid speech waveform")
    return {"text_chars": len(text), "speech_seconds": round(len(audio) / rate, 3), "speech_hz": rate}


def output_paths(root: Path, index: int) -> dict[str, Path]:
    return {name: root / name / str(index) / filename for name, filename in OUTPUTS.items()}


def preserve_partial(root: Path, index: int, paths: dict[str, Path]) -> None:
    present = {name: path for name, path in paths.items() if path.exists() or path.is_symlink()}
    if not present:
        return
    saved = root / "incomplete" / f"{index}.{uuid.uuid4().hex}"
    saved.mkdir(parents=True)
    for name, path in present.items():
        path.replace(saved / f"{name}.{OUTPUTS[name]}")
    print(f"preserved incomplete sample {index}: {saved}", flush=True)


def run(args) -> int:
    dataset = args.dataset_root.resolve()
    output = args.output_root.absolute()
    if output.is_symlink() or output.resolve().is_relative_to(dataset):
        raise ValueError("output root may not be a symlink or inside the dataset")
    if not args.dynin_source.is_dir():
        raise ValueError("--dynin-source must point to a clone of AIDASLab/Dynin-Omni")
    if not torch.cuda.is_available() or not args.device.startswith("cuda:"):
        raise RuntimeError("this 8B runner requires an explicitly selected CUDA device")
    if not (0 <= args.start < args.until <= 900):
        raise ValueError("require 0 <= start < until <= 900")
    if args.steps < 1 or args.text_length < 2 or args.speech_length < 2:
        raise ValueError("steps >= 1 and target lengths >= 2 are required")
    rows, parquet_hash = load_rows(dataset)
    manifest = {
        "checkpoint": args.model, "checkpoint_revision": args.model_revision,
        "dynin_source_revision": source_revision(args.dynin_source),
        "image_codec": args.image_codec, "image_revision": args.image_revision,
        "speech_codec": args.speech_codec, "speech_revision": args.speech_revision,
        "speech_runtime_repo": args.speech_runtime_repo,
        "dataset_parquet_sha256": parquet_hash, "start": args.start, "until": args.until,
        "steps": args.steps, "text_length": args.text_length,
        "speech_length": args.speech_length, "speech_style": args.speech_style,
        "seed": args.seed, "sampler": "joint_confidence_commit_v1", "task_id": TASK_ID,
    }
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"output root is nonempty; select a new directory or use --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    status_path = output / "status.json"
    if args.resume and manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("resume manifest differs from this command or dataset")
        status = json.loads(status_path.read_text(encoding="utf-8"))
    else:
        if args.resume and any(output.iterdir()):
            raise ValueError("cannot resume nonempty output without manifest and status")
        atomic_json(manifest_path, manifest)
        status = {"state": "starting", "samples": {}}
    status["state"] = "running"
    atomic_json(status_path, status)
    torch.cuda.set_device(args.device)
    device = torch.device(args.device)
    sys.path.insert(0, str(args.dynin_source.resolve()))
    from models import DyninOmniModelLM, MAGVITv2  # noqa: E402

    model_options = repo_load_kwargs(args.model, args.model_revision, args.local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, **model_options)
    if len(tokenizer) != 126372 or tokenizer.convert_tokens_to_ids("<|ti2tis|>") is not None:
        raise ValueError("expected the nips tokenizer with reserved TI2TIS ID 126107")
    image_options = repo_load_kwargs(args.image_codec, args.image_revision, args.local_files_only)
    image_codec = MAGVITv2.from_pretrained(args.image_codec, **image_options).to(device).eval()
    speech_codec = load_speech_codec(args.speech_codec, args.speech_revision, args.speech_runtime_repo, device, args.local_files_only)
    model = DyninOmniModelLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, **model_options).to(device).eval()
    if model.config.model_type != "omada" or model.config.vocab_size <= TASK_ID:
        raise ValueError("checkpoint is not the expected nips/OMaDA configuration")
    failures = 0
    for index in range(args.start, args.until):
        paths = output_paths(output, index)
        entry = status["samples"].get(str(index), {})
        if args.resume and entry.get("state") == "complete":
            validate_triplet(paths)
            print(f"skip index={index} verified complete", flush=True)
            continue
        preserve_partial(output, index, paths)
        stage = output / ".staging" / f"{index}.{uuid.uuid4().hex}"
        status["samples"][str(index)] = {"state": "running", "id": rows[index]["id"]}
        atomic_json(status_path, status)
        started = time.time()
        try:
            torch.manual_seed(args.seed + index)
            row = rows[index]
            image_path = current_image_path(dataset, row["current_image"])
            with torch.inference_mode():
                sequence, spans, attention = build_sequence(row, image_path, image_codec, tokenizer, device, args.text_length, args.speech_length, model.config.mask_token_id)
                sequence = sample_joint(model, sequence, spans, attention, tokenizer, args.steps, model.config.mask_token_id)
                decoded = decode_to_stage(stage, sequence, spans, tokenizer, image_codec, speech_codec, args.speech_style)
            staged = {name: stage / filename for name, filename in OUTPUTS.items()}
            check = validate_triplet(staged)
            for name, path in paths.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                staged[name].replace(path)
            if args.write_token_debug:
                atomic_json(output / "debug" / f"{index}.json", {
                    "id": row["id"], "spans": spans,
                    "generated_ids": {name: sequence[0, slice(*span)].tolist() for name, span in spans.items()},
                })
            stage.rmdir()
            status["samples"][str(index)] = {"state": "complete", "id": row["id"], **decoded, **check, "elapsed_seconds": round(time.time() - started, 2)}
            print(f"complete index={index} id={row['id']} seconds={check['speech_seconds']}", flush=True)
        except Exception as error:
            failures += 1
            status["samples"][str(index)] = {"state": "failed", "id": rows[index]["id"], "error": repr(error)}
            traceback.print_exc()
        atomic_json(status_path, status)
    status["state"] = "failed" if failures else "generated_unverified"
    atomic_json(status_path, status)
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynin-source", required=True, type=Path, help="public AIDASLab/Dynin-Omni checkout")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--model", default="snu-aidas/Dynin-Omni_nips")
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--image-codec", default="snu-aidas/magvitv2")
    parser.add_argument("--image-revision", default=IMAGE_REVISION)
    parser.add_argument("--speech-codec", default="snu-aidas/emova_speech_tokenizer_vllm")
    parser.add_argument("--speech-revision", default=SPEECH_REVISION)
    parser.add_argument("--speech-runtime-repo", default="snu-aidas/emova_speech_tokenizer_vllm")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--until", type=int, default=3, help="exclusive Parquet row index")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--text-length", type=int, default=128)
    parser.add_argument("--speech-length", type=int, default=125)
    parser.add_argument("--speech-style", default="gender-female_emotion-neutral_speed-normal_pitch-normal")
    parser.add_argument("--device", required=True, help="explicit GPU, e.g. cuda:0")
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--write-token-debug", action="store_true")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
