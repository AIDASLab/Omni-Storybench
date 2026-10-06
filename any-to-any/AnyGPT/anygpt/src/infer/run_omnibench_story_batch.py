import sys
sys.path.append("./")
sys.path.append("./anygpt/src")

import argparse
import json
import re
import traceback
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm
from transformers import GenerationConfig

from infer.cli_infer_chat_model import AnyGPTChatInference
from m_utils.anything2token import modal_special_str, modality_tokens_to_string
from m_utils.conversation import get_conv_template


MMGPT_RESPONSE_PATTERN = re.compile(
    r"\[MMGPT\]\s*:\s*(.*?)(?:<eos>|<eom>)\s*$",
    flags=re.DOTALL,
)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            yield idx, json.loads(line)


def ensure_result_dirs(results_root, sample_idx):
    paths = {}
    for modality in ("text", "image", "speech"):
        path = Path(results_root) / modality / str(sample_idx)
        path.mkdir(parents=True, exist_ok=True)
        paths[modality] = path
    return paths


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text or "").strip()


def strip_wrapping_quotes(text):
    text = (text or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1].strip()
    return text


def strip_meta_lead_sentence(text):
    text = normalize_whitespace(text)
    if not text:
        return text

    parts = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
    if len(parts) != 2:
        return text

    first_sentence = parts[0].strip().lower()
    meta_prefixes = (
        "write a sentence",
        "write one sentence",
        "describe the next page",
        "describe the scene",
        "generate the next page",
        "here is the narration",
    )
    if first_sentence.startswith(meta_prefixes):
        return parts[1].strip()
    return text


def extract_assistant_payload(decoded):
    matches = list(MMGPT_RESPONSE_PATTERN.finditer(decoded))
    if matches:
        return matches[-1].group(1).strip()
    if "[MMGPT]" in decoded:
        decoded = decoded.rsplit("[MMGPT]", 1)[-1]
    return re.sub(r"^\s*:\s*", "", decoded, count=1).strip()


def extract_modality_segments(text, modality):
    special = modal_special_str[modality]
    pattern = re.compile(
        re.escape(special["sos"]) + r"(.*?)" + re.escape(special["eos"]),
        flags=re.DOTALL,
    )
    return [match.group(1) for match in pattern.finditer(text)]


def remove_modality_segments(text):
    cleaned = text
    for modality in modal_special_str:
        special = modal_special_str[modality]
        cleaned = re.sub(
            re.escape(special["sos"]) + r".*?" + re.escape(special["eos"]),
            " ",
            cleaned,
            flags=re.DOTALL,
        )
    cleaned = re.sub(r"^\s*:\s*", "", cleaned, count=1)
    return normalize_whitespace(cleaned)


def parse_text_pass_response(assistant_response):
    clean_text = remove_modality_segments(assistant_response)
    narration = ""
    speech_text = ""
    speech_fallback_used = False

    json_match = re.search(r"\{.*\}", clean_text, flags=re.DOTALL)
    if json_match:
        try:
            parsed_json = json.loads(json_match.group(0))
            narration = normalize_whitespace(parsed_json.get("narration", ""))
            speech_text = normalize_whitespace(
                parsed_json.get("speech", parsed_json.get("speech_text", ""))
            )
        except json.JSONDecodeError:
            pass

    narration_match = re.search(
        r"(?is)(?:^|\n)\s*narration\s*:\s*(.+?)(?=(?:\n\s*speech\s*:)|\Z)",
        clean_text,
    )
    speech_match = re.search(
        r"(?is)(?:^|\n)\s*speech\s*:\s*(.+?)\s*$",
        clean_text,
    )

    if narration_match:
        narration = normalize_whitespace(narration_match.group(1))
    if speech_match:
        speech_text = normalize_whitespace(speech_match.group(1))

    quoted_spans = re.findall(r'"([^"]+)"|“([^”]+)”|\'([^\']+)\'', clean_text)
    quoted_spans = [normalize_whitespace("".join(span)) for span in quoted_spans if "".join(span).strip()]

    if not speech_text and quoted_spans:
        speech_text = quoted_spans[-1]
        speech_fallback_used = True

    if not narration:
        lines = []
        for line in clean_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            stripped = re.sub(r"^\s*(narration|speech)\s*:\s*", "", stripped, flags=re.IGNORECASE)
            stripped = normalize_whitespace(stripped)
            if stripped:
                lines.append(stripped)
        if lines:
            narration = lines[0]
        if len(lines) > 1 and not speech_text:
            speech_text = lines[1]
            speech_fallback_used = True

    narration = strip_meta_lead_sentence(strip_wrapping_quotes(narration))
    speech_text = strip_wrapping_quotes(speech_text)

    if speech_text and narration:
        narration = normalize_whitespace(
            narration.replace(f'"{speech_text}"', "").replace(f"“{speech_text}”", "")
        )

    return {
        "clean_text": clean_text,
        "narration": narration,
        "speech_text": speech_text,
        "speech_fallback_used": speech_fallback_used,
    }


def get_story_fields(sample):
    metadata = sample.get("metadata", {}).get("metadata", {})
    next_page_condition = sample.get("next_page_condition", {})
    condition_text = next_page_condition.get("condition_text")
    if not condition_text:
        condition_text = json.dumps(next_page_condition.get("condition_json", {}), ensure_ascii=False)
    return {
        "current_page_text": sample["current_page"]["text"],
        "metadata_text": json.dumps(metadata, ensure_ascii=False),
        "condition_text": condition_text,
    }


def encode_current_image_string(inferencer, sample):
    image_tokens = inferencer.encode_image(image_path=sample["current_page"]["image_path"])[0]
    return modality_tokens_to_string(tokens=image_tokens, modality="image")


def build_text_pass_instruction(fields):
    return (
        "You are writing the next page of a children's picture book.\n"
        "The current page image is provided before this instruction.\n"
        "Use the current page image, current page text, book metadata, and the next page condition.\n"
        "Output exactly two lines and nothing else.\n"
        "Line 1 must start with 'Narration:' followed by the actual next-page narration in story style.\n"
        "Line 2 must start with 'Speech:' followed by one short spoken line for an appropriate character.\n"
        "The narration must read like a finished book sentence, not like an instruction, explanation, or scene label.\n"
        "Keep Narration to 1-2 short sentences. Keep Speech to 3-12 words.\n"
        "Put only the spoken words after 'Speech:'. Do not add speaker names, bullets, JSON, image tokens, or speech tokens.\n"
        "Do not use meta phrases such as 'Write a sentence', 'Describe the scene', 'The next scene depicts', or 'Here is'.\n\n"
        "Example output:\n"
        "Narration: The little fox tiptoes into the moonlit garden and smiles at the silver flowers.\n"
        "Speech: They're glowing tonight!\n\n"
        f"Current page text:\n{fields['current_page_text']}\n\n"
        f"Book metadata:\n{fields['metadata_text']}\n\n"
        f"Next page condition:\n{fields['condition_text']}"
    )


def build_image_pass_instruction(fields, narration, speech_text):
    parts = [
        "Generate the next-page illustration for the same children's picture book.",
        "The current page image is provided before this instruction as a style and continuity reference.",
        "Return only the image.",
        "",
        f"Current page text:\n{fields['current_page_text']}",
        "",
        f"Generated next page narration:\n{narration or fields['condition_text']}",
    ]
    if speech_text:
        parts.extend(["", f"Suggested spoken line in the scene:\n{speech_text}"])
    parts.extend(
        [
            "",
            f"Book metadata:\n{fields['metadata_text']}",
            "",
            f"Next page condition:\n{fields['condition_text']}",
        ]
    )
    return "\n".join(parts)


def build_speech_pass_instruction(fields, narration, speech_text):
    parts = [
        "Generate speech audio for one short character line from the next scene of a children's picture book.",
        "Return only speech audio. Do not return text.",
        "",
        f"Current page text:\n{fields['current_page_text']}",
        "",
        f"Next page narration:\n{narration or fields['condition_text']}",
    ]
    if speech_text:
        parts.extend(["", f"Read this line aloud exactly:\n{speech_text}"])
    else:
        parts.extend(
            [
                "",
                "Infer one short spoken line for an appropriate character from the context below and speak it naturally.",
            ]
        )
    parts.extend(
        [
            "",
            f"Book metadata:\n{fields['metadata_text']}",
            "",
            f"Next page condition:\n{fields['condition_text']}",
        ]
    )
    return "\n".join(parts)


def build_prompt(user_content, force_modality=None, force_response_prefix=None):
    conv = get_conv_template("MMGPT")
    conv.append_message(conv.roles[0], user_content.strip())
    if force_modality == "image":
        return conv.get_prompt(force_image_generation=True)
    if force_modality == "speech":
        return conv.get_prompt(force_speech_generation=True)
    return conv.get_prompt(force_res_prefix=force_response_prefix)


def generate_response(inferencer, prompt, config_path):
    config_dict = json.load(open(config_path, "r", encoding="utf-8"))
    generation_config = GenerationConfig(**config_dict)
    input_ids = inferencer.tokenizer(prompt, return_tensors="pt", padding=True).input_ids
    input_ids = input_ids.to(inferencer.device)

    with torch.no_grad():
        generated_ids = inferencer.model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
        )
    sequences = generated_ids.sequences
    decoded = inferencer.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=True)[0]
    assistant_response = extract_assistant_payload(decoded)
    return decoded, assistant_response


def write_debug_outputs(output_dir, prefix, prompt, raw_response, assistant_response):
    with open(output_dir / f"{prefix}_prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)
    with open(output_dir / f"{prefix}_raw_response.txt", "w", encoding="utf-8") as f:
        f.write(raw_response)
    with open(output_dir / f"{prefix}_assistant_response.txt", "w", encoding="utf-8") as f:
        f.write(assistant_response)


def write_text_outputs(text_dir, sample, prompt, raw_response, assistant_response, parsed):
    generated_text = parsed["narration"] or parsed["clean_text"]

    with open(text_dir / "input_record.json", "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    write_debug_outputs(text_dir, "text_pass", prompt, raw_response, assistant_response)
    with open(text_dir / "generated_text.txt", "w", encoding="utf-8") as f:
        f.write(generated_text)
    with open(text_dir / "generated_speech_text.txt", "w", encoding="utf-8") as f:
        f.write(parsed["speech_text"])
    with open(text_dir / "structured_generation.json", "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)


def save_image_outputs(inferencer, image_dir, assistant_response):
    image_segments = extract_modality_segments(assistant_response, "image")
    for idx, image_content in enumerate(image_segments):
        image = inferencer.decode_image(image_content)
        image.save(image_dir / f"image_{idx:02d}.jpg")
    return len(image_segments)


def save_speech_outputs(inferencer, speech_dir, assistant_response, voice_prompt_path=None):
    speech_segments = extract_modality_segments(assistant_response, "speech")
    for idx, speech_content in enumerate(speech_segments):
        wav = inferencer.decode_speech(speech_content, prompt_path=voice_prompt_path)
        torchaudio.save(
            str(speech_dir / f"speech_{idx:02d}.wav"),
            wav,
            inferencer.speech_tokenizer.sample_rate,
        )
    return len(speech_segments)


def write_summary(text_dir, parsed, image_count, speech_count, speech_generation_mode):
    generated_text = parsed["narration"] or parsed["clean_text"]
    summary = {
        "text_present": bool(generated_text),
        "narration_present": bool(parsed["narration"]),
        "speech_text_present": bool(parsed["speech_text"]),
        "speech_text_fallback_used": parsed["speech_fallback_used"],
        "speech_generation_mode": speech_generation_mode,
        "image_count": image_count,
        "speech_count": speech_count,
    }
    with open(text_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the legacy story JSONL dataset",
    )
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--image-tokenizer-path", type=str, required=True)
    parser.add_argument("--speech-tokenizer-path", type=str, required=True)
    parser.add_argument("--speech-tokenizer-config", type=str, required=True)
    parser.add_argument("--soundstorm-path", type=str, required=True)
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--text-generation-config", type=str, default="config/text_generate_config.json")
    parser.add_argument("--image-generation-config", type=str, default="config/image_generate_config.json")
    parser.add_argument("--speech-generation-config", type=str, default="config/speech_generate_config.json")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1)
    parser.add_argument("--voice-prompt-path", type=str, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    inferencer = AnyGPTChatInference(
        model_name_or_path=args.model_name_or_path,
        image_tokenizer_path=args.image_tokenizer_path,
        output_dir=args.results_root,
        speech_tokenizer_path=args.speech_tokenizer_path,
        speech_tokenizer_config=args.speech_tokenizer_config,
        soundstorm_path=args.soundstorm_path,
    )

    sample_count = 0
    for sample_idx, sample in tqdm(load_jsonl(args.dataset_path), desc="omnibench_story_batch"):
        if sample_idx < args.start_index:
            continue
        if args.end_index >= 0 and sample_idx > args.end_index:
            break

        result_dirs = ensure_result_dirs(args.results_root, sample_idx)
        if args.skip_existing and (result_dirs["text"] / "summary.json").exists():
            continue

        try:
            fields = get_story_fields(sample)
            image_string = encode_current_image_string(inferencer, sample)

            text_instruction = build_text_pass_instruction(fields)
            text_prompt = build_prompt(
                f"{image_string}\n{text_instruction}",
                force_response_prefix="Narration:",
            )
            text_raw_response, text_assistant_response = generate_response(
                inferencer=inferencer,
                prompt=text_prompt,
                config_path=args.text_generation_config,
            )
            parsed_text = parse_text_pass_response(text_assistant_response)
            write_text_outputs(
                text_dir=result_dirs["text"],
                sample=sample,
                prompt=text_prompt,
                raw_response=text_raw_response,
                assistant_response=text_assistant_response,
                parsed=parsed_text,
            )

            narration_text = parsed_text["narration"] or parsed_text["clean_text"]
            speech_text = parsed_text["speech_text"]

            image_instruction = build_image_pass_instruction(fields, narration_text, speech_text)
            image_prompt = build_prompt(f"{image_string}\n{image_instruction}", force_modality="image")
            image_raw_response, image_assistant_response = generate_response(
                inferencer=inferencer,
                prompt=image_prompt,
                config_path=args.image_generation_config,
            )
            write_debug_outputs(
                result_dirs["image"],
                "image_pass",
                image_prompt,
                image_raw_response,
                image_assistant_response,
            )
            image_count = save_image_outputs(
                inferencer=inferencer,
                image_dir=result_dirs["image"],
                assistant_response=image_assistant_response,
            )

            speech_instruction = build_speech_pass_instruction(fields, narration_text, speech_text)
            speech_prompt = build_prompt(speech_instruction, force_modality="speech")
            speech_raw_response, speech_assistant_response = generate_response(
                inferencer=inferencer,
                prompt=speech_prompt,
                config_path=args.speech_generation_config,
            )
            write_debug_outputs(
                result_dirs["speech"],
                "speech_pass",
                speech_prompt,
                speech_raw_response,
                speech_assistant_response,
            )
            speech_count = save_speech_outputs(
                inferencer=inferencer,
                speech_dir=result_dirs["speech"],
                assistant_response=speech_assistant_response,
                voice_prompt_path=args.voice_prompt_path,
            )

            speech_generation_mode = "read_generated_line" if speech_text else "infer_from_context"
            write_summary(
                text_dir=result_dirs["text"],
                parsed=parsed_text,
                image_count=image_count,
                speech_count=speech_count,
                speech_generation_mode=speech_generation_mode,
            )
            sample_count += 1
        except Exception:
            with open(result_dirs["text"] / "error.txt", "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())

    print(f"processed_samples={sample_count}")


if __name__ == "__main__":
    main()
