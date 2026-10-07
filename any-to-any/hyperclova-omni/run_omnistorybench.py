#!/usr/bin/env python3
"""HyperCLOVA X Omni inference for the public Omni-StoryBench snapshot.

Adapted from OmniServe's benchmark client. The server performs model loading,
encoding, and decoding; this client reads only prompt-safe Parquet columns and
writes candidates in canonical row order.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import time
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote, urlsplit

if TYPE_CHECKING:
    from openai import OpenAI


BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = BENCHMARK_ROOT / "evaluation" / "downloaded_dataset"
DEFAULT_RESULTS_ROOT = BENCHMARK_ROOT / "evaluation" / "baselines" / "HyperClovaX-Omni"
EXPECTED_SAMPLE_COUNT = 900
EXPECTED_SOURCE_COUNTS = {
    "africanstorybook": 54,
    "digitallibrary": 293,
    "storybookscanda": 7,
    "storyweaver": 546,
}
EXPECTED_SENTINEL_IDS = {
    0: "africanstorybook/asb10179/page_006__page_007",
    724: "africanstorybook/asb10474/page_006__page_007",
    757: "africanstorybook/asb21992/page_003__page_004",
    758: "africanstorybook/asb31833/page_009__page_010",
    899: "storyweaver/515152-rosie-the-caterpillar/page_007__page_008",
}
PROMPT_COLUMNS = (
    "id", "source", "book", "from_key", "to_key", "current_image",
    "current_text", "genre", "topic", "style", "narrative_tense",
    "narrative_perspective", "scene", "text_instruction", "ambient_sound",
    "condition_characters", "book_characters",
)
INPUT_PATH_COLUMNS = (
    "current_image", "current_text_path", "sample_path", "metadata_path",
)
ARTIFACT_NAMES = {
    "text": "generated.txt",
    "image": "generated.png",
    "speech": "generated.wav",
}


class ValidationError(RuntimeError):
    """An input or output does not satisfy the benchmark contract."""


def log(message: str) -> None:
    print(f"[LOG] {message}", flush=True)


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def resolve_dataset_file(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{label} must be a non-empty relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValidationError(f"{label} must stay inside the dataset: {value!r}")
    path = (root / Path(*relative.parts)).resolve()
    if not path.is_relative_to(root):
        raise ValidationError(f"{label} resolves outside the dataset: {value!r}")
    if not path.is_file() or path.stat().st_size == 0:
        raise ValidationError(f"{label} is missing or empty: {path}")
    return path


def parse_characters(value: Any, label: str) -> List[Dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"{label} contains malformed JSON") from exc
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValidationError(f"{label} must be a JSON list of objects")
    return [dict(item) for item in value]


def normalize_condition_characters(
    sample_id: str, characters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    # Match the other benchmark adapters: these two absent JSON keys were
    # materialized as empty strings by the public Parquet exporter.
    name = {
        "digitallibrary/the-green-apple/page_006__page_007": "Strawberry",
        "storyweaver/213576-jolly-swing/page_007__page_008": "Mamuni",
    }.get(sample_id)
    if name is not None:
        matching = [
            item for item in characters
            if item.get("name") == name and item.get("speech_intent") == ""
        ]
        if len(matching) != 1:
            raise ValidationError(f"{sample_id}: unexpected speech_intent normalization")
        del matching[0]["speech_intent"]
    return characters


def load_samples(dataset_root: Path) -> List[Dict[str, Any]]:
    """Read the full snapshot in row order without reading reference columns."""
    path = dataset_root / "data" / "omni_storybench.parquet"
    if not path.is_file() or path.stat().st_size == 0:
        raise ValidationError(f"canonical Parquet index is missing or empty: {path}")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValidationError("pyarrow is required; install the client dependencies in README.md") from exc

    columns = list(dict.fromkeys((*PROMPT_COLUMNS, *INPUT_PATH_COLUMNS)))
    parquet = pq.ParquetFile(str(path))
    missing = sorted(set(columns) - set(parquet.schema_arrow.names))
    if missing:
        raise ValidationError(f"Parquet is missing required columns: {missing}")
    rows = parquet.read(columns=columns).to_pylist()
    if len(rows) != EXPECTED_SAMPLE_COUNT:
        raise ValidationError(f"expected 900 rows, found {len(rows)}")
    ids = [row["id"] for row in rows]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValidationError("every row must have a non-empty sample id")
    if len(set(ids)) != EXPECTED_SAMPLE_COUNT:
        raise ValidationError("sample ids must be unique")
    for index, expected in EXPECTED_SENTINEL_IDS.items():
        if ids[index] != expected:
            raise ValidationError(f"canonical row order mismatch at index {index}")
    if dict(Counter(row["source"] for row in rows)) != EXPECTED_SOURCE_COUNTS:
        raise ValidationError("unexpected source distribution")

    samples = []
    for index, row in enumerate(rows):
        expected_id = f"{row['source']}/{row['book']}/{row['from_key']}__{row['to_key']}"
        if row["id"] != expected_id:
            raise ValidationError(f"row {index}: sample id does not match its transition")
        if not isinstance(row["current_text"], str) or not row["current_text"].strip():
            raise ValidationError(f"row {index}: current_text is empty")
        for column in INPUT_PATH_COLUMNS:
            resolve_dataset_file(dataset_root, row[column], f"row {index} {column}")
        metadata = {
            field: row[field]
            for field in ("genre", "topic", "style", "narrative_tense", "narrative_perspective")
        }
        metadata["characters"] = parse_characters(row["book_characters"], f"row {index} book_characters")
        condition = {
            "characters": normalize_condition_characters(
                row["id"], parse_characters(row["condition_characters"], f"row {index} condition_characters"),
            ),
            "scene": row["scene"],
            "text_instruction": row["text_instruction"],
            "ambient_sound": row["ambient_sound"],
        }
        samples.append({
            "index": index, "id": row["id"], "source": row["source"], "book": row["book"],
            "input": {
                "current_page": {"image": row["current_image"], "text": row["current_text"]},
                "book_metadata": metadata,
            },
            "next_page_condition": condition,
        })
    return samples


def require_http_url(value: str, label: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValidationError(f"{label} must be an HTTP(S) URL")
    return value


def resolve_image_url(sample: Dict[str, Any], dataset_root: Path, http_root: str) -> str:
    relative = sample["input"]["current_page"]["image"]
    resolve_dataset_file(dataset_root, relative, "current image")
    return http_root.rstrip("/") + "/" + quote(relative, safe="/")


SYSTEM_PROMPT = """You are an AI assistant that transforms images. When asked to transform, edit, or stylize an image, you MUST use the t2i_model_generation tool to generate the new image. Always respond by calling the tool."""


_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def build_extra_body(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.thinking:
        return {"chat_template_kwargs": {"skip_reasoning": True}}

    # Keep </think> visible to the server reasoning parser.
    extra_body: Dict[str, Any] = {
        "chat_template_kwargs": {"thinking": True},
        "skip_special_tokens": False,
    }
    if args.thinking_token_budget > 0:
        extra_body["thinking_token_budget"] = args.thinking_token_budget
    return extra_body


def strip_think(text: str) -> str:
    """Drop any <think>...</think> that leaked into content.

    Only fires if the server-side reasoning parser did not split it out; a
    correctly configured run leaves nothing to strip.
    """
    cleaned = _THINK_BLOCK.sub("", text or "")
    # Unterminated block (generation hit max_tokens mid-reasoning).
    if "<think>" in cleaned and "</think>" not in cleaned:
        cleaned = cleaned.split("<think>", 1)[0]
    return cleaned.replace("</think>", "").strip()


def get_reasoning(msg: Any) -> str:
    for field in ("reasoning", "reasoning_content"):
        value = getattr(msg, field, None)
        if value:
            return str(value).strip()
    return ""


def format_book_metadata(sample: Dict[str, Any]) -> str:
    return json.dumps(sample["input"]["book_metadata"], ensure_ascii=False, indent=2)


def format_condition(sample: Dict[str, Any]) -> str:
    return json.dumps(sample["next_page_condition"], ensure_ascii=False, indent=2)


def format_speech_condition(sample: Dict[str, Any]) -> str:
    """Voice direction taken ONLY from next_page_condition.

    ground_truth.speech carries gender/emotion/pitch too, but it is the
    evaluation target -- feeding it back would be leakage.
    """
    condition = sample["next_page_condition"]
    lines = []
    for character in condition.get("characters", []) or []:
        parts = [f"- {character.get('name', 'Unknown')}"]
        for key in ("emotion", "speech_intent", "action", "visibility"):
            if character.get(key):
                parts.append(f"{key}: {character[key]}")
        lines.append("; ".join(parts))
    if condition.get("ambient_sound"):
        lines.append(f"- ambient sound: {condition['ambient_sound']}")
    return "\n".join(lines)


def build_text_prompt(sample: Dict[str, Any]) -> str:
    return f"""You are writing the next page of a children's picture book.

Use the previous page image, previous page text, story metadata, and next page condition.
Generate the content for the next page.

Return exactly in this format:
NARRATION: natural story sentences for the next page>
SPEECH: <one short spoken line for the next page>

Rules:
- Keep the same language as the current page text.
- Do not output JSON.
- Do not output explanations.
- Do not output speaker names.
- Keep SPEECH short and natural.

Current page text:
{sample["input"]["current_page"]["text"]}

Story metadata:
{format_book_metadata(sample)}

Next page condition:
{format_condition(sample)}
""".strip()


def build_image_prompt(sample: Dict[str, Any], narration: str, speech_text: str) -> str:
    return f"""Draw the next-page image for the children's picture book.

Use the provided previous-page image as the visual reference for style, character consistency, and scene continuity.

Current page text:
{sample["input"]["current_page"]["text"]}

Next page narration:
{narration}

Story metadata:
{format_book_metadata(sample)}

Next page condition:
{format_condition(sample)}
""".strip()


def build_speech_prompt(sample: Dict[str, Any], narration: str, speech_text: str) -> str:
    direction = format_speech_condition(sample)
    prompt = f"Read the following line aloud:\n{speech_text}\n"
    if direction:
        prompt += f"\nVoice direction:\n{direction}\n"
    return prompt.strip()


_DECORATION = re.compile(r"^[\s>#*\-•–—`_]+|[\s*_`]+$")


_LABEL = re.compile(r"^(NARRATION|SPEECH)\s*:?\s*\**\s*:?\s*", re.IGNORECASE)


_STAGE_DIRECTION = re.compile(r"^[\(\[].*[\)\]]$", re.DOTALL)


def strip_decoration(line: str) -> str:
    return _DECORATION.sub("", line).replace("**", "").strip()


def is_speakable(text: str) -> bool:
    return bool(text) and not _STAGE_DIRECTION.match(text.strip())


def parse_text_output(raw_text: str) -> Tuple[str, str]:
    """Extract the first NARRATION/SPEECH pair.

    The model frequently overruns the requested single page and emits several
    pairs; the first one is the answer, so first-wins rather than last-wins.
    """
    narrations: List[str] = []
    speeches: List[str] = []

    lines = [strip_decoration(line) for line in raw_text.splitlines()]
    lines = [line for line in lines if line]

    # A label may sit alone on its line with the content on the next one:
    #   NARRATION:
    #   She sank into the water...
    # Reading only the label's own line yields nothing, which previously failed
    # the sample outright -- and took the image with it, since image generation
    # needs the narration.
    i = 0
    while i < len(lines):
        match = _LABEL.match(lines[i])
        if not match:
            i += 1
            continue
        value = normalize_text(strip_decoration(lines[i][match.end():]))
        j = i + 1
        while not value and j < len(lines) and not _LABEL.match(lines[j]):
            value = normalize_text(strip_decoration(lines[j]))
            j += 1
        if value:
            (narrations if match.group(1).upper() == "NARRATION" else speeches).append(value)
        i = max(j, i + 1)

    narration = narrations[0] if narrations else ""
    speech = next((s for s in speeches if is_speakable(s)), speeches[0] if speeches else "")

    # Positional fallback ONLY for fully unlabelled output. If the model used
    # labels we trust them: a present NARRATION with no SPEECH means there is no
    # spoken line, and copying the narration into it would fabricate speech.
    if not narrations and not speeches:
        if lines:
            narration = normalize_text(lines[0])
        if len(lines) >= 2:
            speech = normalize_text(lines[1])

    return narration, speech


def generate_text(
    client: OpenAI,
    model_name: str,
    image_url: str,
    prompt: str,
    extra_body: Dict[str, Any],
    max_tokens: int = 2048,
) -> Tuple[str, str]:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=max_tokens,
        extra_body=extra_body,
    )

    msg = response.choices[0].message
    return strip_think(msg.content or ""), get_reasoning(msg)


def generate_speech_audio_url(
    client: OpenAI,
    model_name: str,
    prompt: str,
    extra_body: Dict[str, Any],
    max_tokens: int = 2048,
) -> Optional[str]:
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        extra_body=extra_body,
    )

    msg = response.choices[0].message
    if not getattr(msg, "audio", None):
        return None

    audio_url = base64.urlsafe_b64decode(
        msg.audio.data + "=" * (-len(msg.audio.data) % 4)
    ).decode("utf-8")
    return audio_url.strip()


def generate_image_url_strict(
    client: OpenAI,
    image_url: str,
    text_prompt: str,
    extra_body: Dict[str, Any],
    max_tokens: int = 7000,
    model_name: str = "track_b_model",
) -> Optional[str]:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": text_prompt}
                ]
            }
        ],
        tools=[{
            "type": "function",
            "function": {
                "name": "t2i_model_generation",
                "description": "Generates an RGB image based on the provided discrete image representation.",
                "parameters": {
                    "type": "object",
                    "required": ["discrete_image_token"],
                    "properties": {
                        "discrete_image_token": {
                            "type": "string",
                            "description": "A serialized string of discrete vision tokens, encapsulated by special tokens. The format must be strictly followed: <|discrete_image_start|><|vision_ratio_4:3|><|vision_token|><|visionaaaaa|><|visionbbbbb|>... <|visionzzzzz|><|vision_eol|><|vision_eof|><|discrete_image_end|>.",
                            "minLength": 1
                        }
                    }
                }
            }
        }],
        max_tokens=max_tokens,
        extra_body=extra_body
    )

    if response.choices[0].message.tool_calls:
        args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        return args["discrete_image_token"]

    return None


def check_output_path(path: Path, *, directory: bool = False) -> None:
    """Reject symlinks and unexpected path types before replacing known files."""
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValidationError(f"output path contains a symlink: {current}")
        is_directory = current != path or directory
        if current.exists() and (
            (is_directory and not current.is_dir())
            or (not is_directory and not current.is_file())
        ):
            raise ValidationError(f"unexpected output path type: {current}")


def artifact_path(results_root: Path, modality: str, index: int) -> Path:
    return results_root / modality / str(index) / ARTIFACT_NAMES[modality]


def atomic_write(path: Path, data: Any) -> None:
    check_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = data.encode("utf-8") if isinstance(data, str) else data
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".tmp-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def dump_json(path: Path, payload: Any) -> None:
    atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def download_artifact(url: str, timeout: float) -> bytes:
    # Download the signed URL as returned. The storage endpoint must be
    # reachable from both Docker and this client; do not rewrite its host.
    require_http_url(url, "generated artifact URL")
    import requests

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    if not response.content:
        raise ValueError("generated artifact download was empty")
    return response.content


def process_one_sample(
    sample: Dict[str, Any], client: Any, args: argparse.Namespace,
) -> Dict[str, Any]:
    index = sample["index"]
    paths = {modality: artifact_path(args.results_root, modality, index) for modality in ARTIFACT_NAMES}
    record = {
        "index": index, "id": sample["id"], "errors": {}, "written": [],
        "image_function_call_attempts": 0, "image_function_call_succeeded": False,
    }
    if not args.overwrite and all(path.is_file() and path.stat().st_size > 0 for path in paths.values()):
        record["skipped"] = True
        log(f"[{index}] skipped (all artifacts present)")
        return record

    # Regenerate incomplete samples from the text stage. Remove only this
    # runner's known candidates so an old artifact cannot mask a failed pass.
    for path in paths.values():
        check_output_path(path)
        path.unlink(missing_ok=True)

    extra_body = build_extra_body(args)
    image_url = resolve_image_url(sample, args.dataset_root, args.image_http_root)
    record["current_image_url"] = image_url
    narration, speech_text = "", ""

    try:
        prompt = build_text_prompt(sample)
        record["text_prompt"] = prompt
        raw_text, reasoning = generate_text(
            client, args.model, image_url, prompt, extra_body, args.text_max_tokens,
        )
        record["raw_text"], record["reasoning"] = raw_text, reasoning
        narration, speech_text = parse_text_output(raw_text)
        record["narration"], record["speech_text"] = narration, speech_text
        if not narration:
            raise ValueError("no NARRATION parsed from model output")
        atomic_write(paths["text"], narration + "\n")
        record["written"].append("text")
    except Exception as exc:
        record["errors"]["text"] = f"{type(exc).__name__}: {exc}"
        record["text_traceback"] = traceback.format_exc()

    if not narration:
        record["errors"]["image"] = "image skipped: no narration available"
    else:
        prompt = build_image_prompt(sample, narration, speech_text)
        record["image_prompt"] = prompt
        record["image_attempt_errors"] = []
        for attempt in range(1, args.image_retries + 1):
            record["image_function_call_attempts"] = attempt
            try:
                url = generate_image_url_strict(
                    client, image_url, prompt, extra_body, args.image_max_tokens, args.model,
                )
                if not url:
                    raise ValueError("no image tool call returned by the model")
                data = download_artifact(url, args.download_timeout)
                if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ValueError("image download is not a PNG")
                atomic_write(paths["image"], data)
                record["written"].append("image")
                record["generated_image_url"] = url
                record["image_function_call_succeeded"] = True
                break
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                record["image_attempt_errors"].append(message)
                log(f"[{index}] image attempt {attempt}/{args.image_retries}: {message}")
                if attempt < args.image_retries:
                    time.sleep(args.retry_sleep)
        else:
            record["errors"]["image"] = (
                f"image failed after {args.image_retries} attempts: "
                + record["image_attempt_errors"][-1]
            )

    if not speech_text:
        record["errors"]["speech"] = "speech skipped: no SPEECH line parsed"
    else:
        try:
            prompt = build_speech_prompt(sample, narration, speech_text)
            record["speech_prompt"] = prompt
            url = generate_speech_audio_url(
                client, args.model, prompt, extra_body, args.speech_max_tokens,
            )
            if not url:
                raise ValueError("model returned no audio")
            data = download_artifact(url, args.download_timeout)
            if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
                raise ValueError("speech download is not a WAV")
            atomic_write(paths["speech"], data)
            record["written"].append("speech")
            record["audio_url"] = url
        except Exception as exc:
            record["errors"]["speech"] = f"{type(exc).__name__}: {exc}"
            record["speech_traceback"] = traceback.format_exc()

    log(f"[{index}] written={record['written']} errors={list(record['errors'])}")
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=os.environ.get("HYPERCLOVA_DATASET_ROOT", str(DEFAULT_DATASET_ROOT)))
    parser.add_argument("--results-root", default=os.environ.get("HYPERCLOVA_RESULTS_ROOT", str(DEFAULT_RESULTS_ROOT)))
    parser.add_argument("--base-url", default=os.environ.get("HYPERCLOVA_BASE_URL", "http://localhost:8000/b/v1"))
    parser.add_argument("--api-key", default=os.environ.get("HYPERCLOVA_API_KEY", "not-needed"))
    parser.add_argument("--model", default=os.environ.get("HYPERCLOVA_MODEL", "track_b_model"))
    parser.add_argument("--image-http-root", default=os.environ.get("HYPERCLOVA_IMAGE_HTTP_ROOT"),
                        help="dataset HTTP root reachable from encoder containers")
    parser.add_argument("--start", type=int, default=0, help="first index, inclusive")
    parser.add_argument("--end", type=int, default=None, help="last index, exclusive; default: 900")
    parser.add_argument("--overwrite", action="store_true", help="also regenerate complete samples")
    parser.add_argument("--text-max-tokens", type=int, default=2048)
    parser.add_argument("--speech-max-tokens", type=int, default=2048)
    parser.add_argument("--image-max-tokens", type=int, default=7000)
    parser.add_argument("--thinking", dest="thinking", action="store_true", default=True)
    parser.add_argument("--no-thinking", dest="thinking", action="store_false")
    parser.add_argument("--thinking-token-budget", type=int, default=1024,
                        help="reasoning-token cap; 0 means uncapped")
    parser.add_argument("--image-retries", type=int, default=10,
                        help="maximum total image attempts, including the first")
    parser.add_argument("--retry-sleep", type=float, default=1.5)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--download-timeout", type=float, default=120)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-only", action="store_true",
                       help="validate the dataset; no network calls or output writes")
    modes.add_argument("--dry-run", action="store_true",
                       help="also preview prompts, URLs and output paths; no network calls or writes")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    end = EXPECTED_SAMPLE_COUNT if args.end is None else args.end
    if not 0 <= args.start <= end <= EXPECTED_SAMPLE_COUNT:
        parser.error("--start/--end must select a half-open range within 0..900")
    if args.thinking_token_budget < 0 or args.image_retries < 1 or args.retry_sleep < 0:
        parser.error("budget/sleep must be nonnegative and --image-retries must be >= 1")
    for key in ("text_max_tokens", "speech_max_tokens", "image_max_tokens", "request_timeout", "download_timeout"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    args.end = end

    try:
        args.dataset_root = Path(args.dataset_root).expanduser().resolve()
        samples = load_samples(args.dataset_root)
        log(f"validated {len(samples)} samples; indices [{args.start}, {args.end})")
        if args.validate_only:
            return 0
        args.results_root = Path(os.path.abspath(os.path.expanduser(args.results_root)))
        check_output_path(args.results_root, directory=True)
        if not args.image_http_root:
            raise ValidationError("--image-http-root or HYPERCLOVA_IMAGE_HTTP_ROOT is required")
        require_http_url(args.image_http_root, "image HTTP root")
        if urlsplit(args.image_http_root).query or urlsplit(args.image_http_root).fragment:
            raise ValidationError("image HTTP root must not contain a query or fragment")
        require_http_url(args.base_url, "OmniServe base URL")
        selected = samples[args.start:args.end]
        meta = args.results_root / "_meta"
        for name in ("index_map.json", "run_summary.json"):
            check_output_path(meta / name)
        for sample in selected:
            for modality in ARTIFACT_NAMES:
                check_output_path(artifact_path(args.results_root, modality, sample["index"]))
            check_output_path(meta / "samples" / f"{sample['index']}.json")
        if args.dry_run:
            for sample in selected[:3]:
                print(json.dumps({
                    "index": sample["index"], "id": sample["id"],
                    "image_url": resolve_image_url(sample, args.dataset_root, args.image_http_root),
                    "text_prompt": build_text_prompt(sample),
                    "artifacts": {
                        modality: str(artifact_path(args.results_root, modality, sample["index"]))
                        for modality in ARTIFACT_NAMES
                    },
                }, ensure_ascii=False, indent=2))
            log("dry run complete; no requests or output writes")
            return 0

        try:
            from openai import OpenAI
            from tqdm import tqdm
            import requests  # Check all generation dependencies before writing outputs.
        except ImportError as exc:
            raise ValidationError("install the client dependencies listed in README.md") from exc

        records = []
        dump_json(meta / "index_map.json", {"order": "parquet", "ids": [sample["id"] for sample in samples]})
        # Only the explicit image loop retries requests. Text and speech have
        # one request each; disable the SDK's implicit HTTP retries.
        with OpenAI(base_url=args.base_url, api_key=args.api_key,
                    timeout=args.request_timeout, max_retries=0) as client:
            for sample in tqdm(selected, desc="hyperclova-omni"):
                try:
                    record = process_one_sample(sample, client, args)
                except Exception as exc:
                    message = f"unhandled {type(exc).__name__}: {exc}"
                    record = {
                        "index": sample["index"], "id": sample["id"], "written": [],
                        "errors": {
                            "sample": message,
                            **{
                                modality: message for modality in ARTIFACT_NAMES
                                if not artifact_path(args.results_root, modality, sample["index"]).is_file()
                            },
                        },
                        "traceback": traceback.format_exc(),
                        "image_function_call_attempts": 0,
                        "image_function_call_succeeded": False,
                    }
                records.append(record)
                if not record.get("skipped"):
                    dump_json(meta / "samples" / f"{sample['index']}.json", record)

        attempts = [
            record["image_function_call_attempts"]
            for record in records if record["image_function_call_succeeded"]
        ]
        failures = sum(bool(record["errors"]) for record in records)
        summary = {
            "dataset_root": str(args.dataset_root), "results_root": str(args.results_root),
            "base_url": args.base_url, "model": args.model,
            "image_http_root": args.image_http_root,
            "start": args.start, "end": args.end,
            "thinking": args.thinking, "thinking_token_budget": args.thinking_token_budget,
            "extra_body": build_extra_body(args),
            "text_max_tokens": args.text_max_tokens, "image_max_tokens": args.image_max_tokens,
            "speech_max_tokens": args.speech_max_tokens,
            "image_retries": args.image_retries, "retry_sleep": args.retry_sleep,
            "request_timeout": args.request_timeout, "download_timeout": args.download_timeout,
            "overwrite": args.overwrite, "processed": len(records),
            "skipped": sum(bool(record.get("skipped")) for record in records),
            "failed_samples": failures,
            "counts": {
                modality: {
                    "ok": sum(modality in record["written"] for record in records),
                    "error": sum(modality in record["errors"] for record in records),
                } for modality in ARTIFACT_NAMES
            },
            "image_function_call": {
                "succeeded": len(attempts),
                "failed": sum("image" in record["errors"] for record in records),
                "first_try": sum(attempt == 1 for attempt in attempts),
                "mean_attempts": sum(attempts) / len(attempts) if attempts else None,
                "max_attempts": max(attempts) if attempts else None,
                "attempts_histogram": dict(Counter(str(attempt) for attempt in attempts)),
            },
        }
        dump_json(meta / "run_summary.json", summary)
        log(f"completed: {len(records)} selected samples, {failures} failed; metadata: {meta}")
        return 1 if failures else 0
    except (ValidationError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
