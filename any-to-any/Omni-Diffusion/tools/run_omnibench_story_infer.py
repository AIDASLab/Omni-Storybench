import argparse
import json
import re
import traceback
from pathlib import Path

from PIL import Image
import soundfile as sf
import torch
from tqdm import tqdm

import inference as base_infer


def load_jsonl(jsonl_path):
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def write_text(path, text):
    path.write_text(text, encoding="utf-8")


def dump_json(path, payload):
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2)


def save_wav(path, audio, sample_rate=22050):
    audio_np = audio.detach().cpu().float().numpy()
    sf.write(str(path), audio_np, sample_rate)


def clean_value(text):
    text = text.strip()
    text = text.strip('"').strip("'")
    return " ".join(text.split())


def extract_field(text, field_name):
    pattern = re.compile(
        rf"(?im)^\s*{re.escape(field_name)}\s*:\s*(.*?)(?=^\s*[A-Z_]+\s*:|\Z)",
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return ""
    return clean_value(match.group(1))


def parse_plan_output(raw_text):
    return {
        "narration": extract_field(raw_text, "NARRATION"),
        "speaker": extract_field(raw_text, "SPEAKER"),
        "speech": extract_field(raw_text, "SPEECH"),
        "image_prompt": extract_field(raw_text, "IMAGE_PROMPT"),
    }


def format_metadata(metadata):
    return json.dumps(metadata, ensure_ascii=False, indent=2)


def allowed_input_view(sample):
    return {
        "source": sample.get("source"),
        "book": sample.get("book"),
        "prev_key": sample.get("prev_key"),
        "next_key": sample.get("next_key"),
        "metadata": sample.get("metadata"),
        "current_page": sample.get("current_page"),
        "next_page_condition": sample.get("next_page_condition"),
    }


def build_planning_prompt(sample):
    current_page = sample["current_page"]
    metadata = sample.get("metadata", {})
    condition = sample.get("next_page_condition", {})
    condition_json = condition.get("condition_json", {})

    prompt = f"""You are writing the next page of a children's picture book.

Use the provided current page image, current page text, story metadata, and next page condition.
Keep the next page consistent with the current story.
Use the same language as the current page text.

Current page text:
{current_page.get("text", "")}

Story metadata:
{format_metadata(metadata)}

Next page condition:
{format_metadata(condition_json)}

Return exactly in this format:
NARRATION: <one or two short sentences for the next page narration>
SPEAKER: <the character who speaks, or narrator if no named character fits>
SPEECH: <one short spoken line for the next page>
IMAGE_PROMPT: <one detailed sentence describing the next-page illustration>

Current page image:"""
    return prompt


def build_speech_recovery_prompt(sample, narration):
    current_page = sample["current_page"]
    metadata = sample.get("metadata", {})
    condition = sample.get("next_page_condition", {})
    condition_json = condition.get("condition_json", {})

    prompt = f"""You are generating one spoken line for the next page of a children's picture book.

Current page text:
{current_page.get("text", "")}

Planned next page narration:
{narration}

Story metadata:
{format_metadata(metadata)}

Next page condition:
{format_metadata(condition_json)}

Return exactly in this format:
SPEAKER: <character name>
SPEECH: <one short spoken line>"""
    return prompt


def build_image_prompt(sample, plan):
    metadata = sample.get("metadata", {}).get("metadata", {})
    condition_json = sample.get("next_page_condition", {}).get("condition_json", {})
    characters = condition_json.get("characters", [])

    char_bits = []
    for char in characters:
        name = char.get("name", "").strip()
        action = char.get("action", "").strip()
        emotion = char.get("emotion", "").strip()
        bit = ", ".join(x for x in [name, emotion, action] if x)
        if bit:
            char_bits.append(bit)

    pieces = [
        "A children's picture book illustration.",
        plan.get("image_prompt", ""),
        f"Narration: {plan.get('narration', '')}",
        f"Scene: {condition_json.get('scene', '')}",
        f"Characters: {'; '.join(char_bits)}" if char_bits else "",
        f"Style: {metadata.get('style', '')}",
        f"Topic: {metadata.get('topic', '')}",
        "No text overlay, no caption, no speech bubble.",
    ]
    return " ".join(piece for piece in pieces if piece).strip()


def save_missing_marker(folder, message):
    write_text(folder / "missing.txt", message + "\n")


def clear_stale_markers(*folders):
    for folder in folders:
        for name in ("error.json", "missing.txt"):
            path = folder / name
            if path.exists():
                path.unlink()


def process_one_sample(index, sample, args, s2s_inference):
    output_root = Path(args.output_root)
    text_dir = output_root / "text" / str(index)
    image_dir = output_root / "image" / str(index)
    speech_dir = output_root / "speech" / str(index)

    ensure_dir(text_dir)
    ensure_dir(image_dir)
    ensure_dir(speech_dir)
    clear_stale_markers(text_dir, image_dir, speech_dir)

    text_path = text_dir / "generated.txt"
    image_path = image_dir / "generated.png"
    speech_path = speech_dir / "generated_0.wav"

    if (
        not args.overwrite
        and text_path.exists()
        and image_path.exists()
        and speech_path.exists()
    ):
        return

    current_page = sample["current_page"]
    current_image_path = current_page["image_path"]

    planning_prompt = build_planning_prompt(sample)
    raw_plan_output, _, _ = s2s_inference.run_infer(
        image_path=current_image_path,
        message=planning_prompt,
        max_tokens=args.text_max_tokens,
        steps=args.text_steps,
        task="",
        alg="entropy",
        repeat_penalty=1.05,
    )

    plan = parse_plan_output(raw_plan_output)

    if not plan["narration"]:
        fallback_text = (
            sample.get("next_page_condition", {})
            .get("condition_json", {})
            .get("text content", "")
        )
        plan["narration"] = clean_value(fallback_text) or clean_value(raw_plan_output)

    if not plan["speech"]:
        recovery_prompt = build_speech_recovery_prompt(sample, plan["narration"])
        raw_speech_output, _, _ = s2s_inference.run_infer(
            image_path=current_image_path,
            message=recovery_prompt,
            max_tokens=80,
            steps=80,
            task="",
            alg="entropy",
            repeat_penalty=1.05,
        )
        recovered = parse_plan_output(raw_speech_output)
        plan["speaker"] = plan["speaker"] or recovered.get("speaker", "")
        plan["speech"] = plan["speech"] or recovered.get("speech", "")

    if not plan["speaker"]:
        plan["speaker"] = "character"
    if not plan["speech"]:
        plan["speech"] = "Let's keep going."
    if not plan["image_prompt"]:
        plan["image_prompt"] = build_image_prompt(sample, plan)

    image_generation_prompt = build_image_prompt(sample, plan)
    _, _, generated_image = s2s_inference.run_infer(
        message="Generate an image based on the provided text description.\n" + image_generation_prompt,
        task="T2I",
        max_tokens=args.image_max_tokens,
        steps=args.image_steps,
        alg="entropy-penalty",
        repeat_penalty=1.2,
        max_position_penalty=2.0,
    )

    _, tts_speech, _ = s2s_inference.run_infer(
        message="Convert the text to speech.\n" + plan["speech"],
        task="TTS",
        max_tokens=args.speech_max_tokens,
        steps=args.speech_steps,
        alg="entropy",
        repeat_penalty=1.0,
    )

    write_text(text_path, plan["narration"] + "\n")
    write_text(text_dir / "plan_raw.txt", raw_plan_output + "\n")
    dump_json(
        text_dir / "structured.json",
        {
            "index": index,
            "input": allowed_input_view(sample),
            "output": {
                "narration": plan["narration"],
                "speaker": plan["speaker"],
                "speech": plan["speech"],
                "image_prompt": plan["image_prompt"],
            },
        },
    )

    if generated_image is not None:
        Image.fromarray(generated_image[:, :, ::-1]).save(str(image_path))
    else:
        save_missing_marker(image_dir, "Image was not generated.")

    write_text(speech_dir / "speech_text.txt", plan["speech"] + "\n")
    write_text(speech_dir / "speaker.txt", plan["speaker"] + "\n")
    if tts_speech is not None:
        save_wav(speech_path, tts_speech, 22050)
    else:
        save_missing_marker(speech_dir, "Speech audio was not generated.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--audio_tokenizer_path", type=str, required=True)
    parser.add_argument("--image_tokenizer_path", type=str, required=True)
    parser.add_argument("--flow_path", type=str, required=True)
    parser.add_argument("--audio_tokenizer_type", type=str, default="sensevoice_glm4voice")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--text_max_tokens", type=int, default=192)
    parser.add_argument("--text_steps", type=int, default=192)
    parser.add_argument("--image_max_tokens", type=int, default=260)
    parser.add_argument("--image_steps", type=int, default=260)
    parser.add_argument("--speech_max_tokens", type=int, default=64)
    parser.add_argument("--speech_steps", type=int, default=32)
    args = parser.parse_args()

    base_infer.device_map = args.device
    base_infer.audio_tokenizer_rank = 0
    base_infer.torch_dtype = getattr(torch, args.dtype)
    base_infer.set_seed(args.seed)

    output_root = Path(args.output_root)
    ensure_dir(output_root / "text")
    ensure_dir(output_root / "image")
    ensure_dir(output_root / "speech")

    samples = load_jsonl(args.jsonl)
    end = len(samples) if args.end is None else min(args.end, len(samples))

    s2s_inference = base_infer.S2SInference(
        args.model_name_or_path,
        args.audio_tokenizer_path,
        args.audio_tokenizer_type,
        args.image_tokenizer_path,
        flow_path=args.flow_path,
    )

    failures = []
    for index in tqdm(range(args.start, end), desc="OmniBench story inference"):
        try:
            process_one_sample(index, samples[index], args, s2s_inference)
        except Exception as exc:
            failures.append({"index": index, "error": str(exc)})
            error_payload = {
                "index": index,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            text_dir = output_root / "text" / str(index)
            image_dir = output_root / "image" / str(index)
            speech_dir = output_root / "speech" / str(index)
            ensure_dir(text_dir)
            ensure_dir(image_dir)
            ensure_dir(speech_dir)
            dump_json(text_dir / "error.json", error_payload)
            dump_json(image_dir / "error.json", error_payload)
            dump_json(speech_dir / "error.json", error_payload)

    summary = {
        "jsonl": args.jsonl,
        "start": args.start,
        "end": end,
        "num_samples": end - args.start,
        "num_failures": len(failures),
        "failures": failures,
    }
    dump_json(output_root / "run_summary.json", summary)


if __name__ == "__main__":
    main()
