"""Build the dataset viewer's data bundle.

Reads the curated example list (tools/viewer_selection.json) and baseline list
(tools/viewer_baselines.json). For each example it cross-checks the sample JSON
against the parquet index and copies downscaled page images; for each baseline
it resolves the generated text, image, and speech exactly as the evaluation
harness does, converts media for the web (JPEG, MP3), records missing outputs
with their logged reason, and attaches the per-sample LLM-judge scores from the
evaluation results. Writes viewer/data/examples.json.

Usage (from the project_page directory):
    uv run --no-project --with pandas --with pyarrow --with pillow --with imageio-ffmpeg \
        python tools/build_viewer.py --dataset /path/to/Omni-StoryBench \
        --outputs /path/to/evaluation_harness/baselines --results /path/to/evaluation_harness/results
"""

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MAX_IMAGE_WIDTH = 800
MAX_OUTPUT_IMAGE_WIDTH = 640
JPEG_QUALITY = 85

# Collections allowed in the public viewer. Storybooks Canada is excluded (CC BY-NC material).
SOURCES = {
    "africanstorybook": {"name": "African Storybook", "license": "CC BY 4.0",
                         "url": "https://www.africanstorybook.org/"},
    "digitallibrary": {"name": "Global Digital Library", "license": "license varies by book",
                       "url": "https://digitallibrary.io/"},
    "storyweaver": {"name": "Storyweaver", "license": "CC BY 4.0",
                    "url": "https://storyweaver.org.in/"},
}

ATTRIBUTION_KEYS = {"title", "authors", "illustrators", "license", "url", "verified"}

PARADIGMS = {
    "orch-large": "Orchestration (Large)",
    "orch-small": "Orchestration (Small)",
    "semi-img": "Semi-orchestration (w/ Image Gen.)",
    "semi-tts": "Semi-orchestration (w/ TTS)",
    "any": "Any-to-any",
}
BASELINE_KEYS = {"dir", "paradigm", "backbone", "image_expert", "speech_expert", "total"}

# LLM-judge results: results/<run>/stage_<n>/<type>/all_results.json. Stages 1-3 are the three judges of each
# evaluation type. Criteria follow the harness order (src/common.py); labels are for display.
JUDGE_STAGES = ("1", "2", "3")
JUDGE_TYPES = {"text": "Narration", "image": "Image", "speech": "Speech", "joint": "Integrated"}
JUDGE_CRITERIA = {
    "text": [("alignment_with_metadata", "Metadata"),
             ("natural_flow_from_current_narration", "Continuity"),
             ("satisfaction_of_generation_conditions", "Conditions"),
             ("semantic_consistency_with_ground_truth", "GT consistency")],
    "image": [("alignment_with_metadata", "Metadata"),
              ("visual_narrative_continuity_from_current_scene", "Continuity"),
              ("satisfaction_of_generation_conditions", "Conditions"),
              ("semantic_consistency_with_ground_truth", "GT consistency")],
    "speech": [("alignment_with_metadata", "Metadata"),
               ("naturalness_and_conversational_relevance", "Naturalness"),
               ("satisfaction_of_generation_conditions", "Conditions"),
               ("semantic_consistency_and_persona_match_with_ground_truth", "GT consistency")],
    "joint": [("alignment_with_metadata", "Metadata"),
              ("natural_multimodal_continuity_from_current_page", "Continuity"),
              ("satisfaction_of_generation_conditions", "Conditions"),
              ("multimodal_semantic_consistency_with_ground_truth", "GT consistency")],
}
JUDGE_STATUSES = {"ok", "skipped", "parse_error"}  # skipped: required outputs missing; parse_error: unscorable

# Candidate-file rules mirrored from the evaluation harness (src/common.py, resolve_*_candidate),
# so the viewer shows the same artifact that the judges scored.
CANDIDATE_RULES = {
    "text": {
        "preferred": ["generated.txt", "generated_text.txt", "text.txt", "candidate.txt", "output.txt"],
        "patterns": ["page_*.txt", "generated_text_*.txt", "generated_[0-9]*.txt", "text_[0-9]*.txt",
                     "candidate*.txt", "output*.txt", "*.txt"],
        "excluded_names": ["missing.txt", "generated_speech_text.txt", "raw_text_response.txt",
                           "text_prompt.txt", "prompt.txt"],
        "excluded_tokens": ["prompt", "raw", "response", "speech"],
    },
    "image": {
        "preferred": ["generated.png"],
        "patterns": ["page_*.png", "page_*.jpg", "page_*.jpeg", "generated_*.png", "generated_*.jpg",
                     "generated_*.jpeg", "generated_*.webp", "image_*.png", "image_*.jpg", "image_*.jpeg",
                     "image_*.webp", "*.png", "*.jpg", "*.jpeg", "*.webp"],
        "excluded_names": [],
        "excluded_tokens": [],
    },
    "speech": {
        "preferred": ["generated.wav"],
        "patterns": ["page_*.wav", "generated_*.wav", "generated_*.flac", "generated_*.mp3", "generated_*.ogg",
                     "generated_*.m4a", "speech_*.wav", "speech_*.flac", "speech_*.mp3", "speech_*.ogg",
                     "speech_*.m4a", "*.wav", "*.flac", "*.mp3", "*.ogg", "*.m4a"],
        "excluded_names": [],
        "excluded_tokens": [],
    },
}


def check_equal(what, sample_id, a, b):
    if a != b:
        raise ValueError(f"{sample_id}: {what} mismatch:\n  sample JSON: {a!r}\n  other:       {b!r}")


def save_jpeg(src, dst, max_width):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        if im.width > max_width:
            im = im.resize((max_width, round(im.height * max_width / im.width)), Image.LANCZOS)
        im.save(dst, "JPEG", quality=JPEG_QUALITY, optimize=True)


def copy_image(dataset, rel_path, out_dir):
    src = dataset / rel_path
    if not src.is_file():
        raise FileNotFoundError(f"missing image {src}")
    dst_rel = Path("images") / Path(rel_path).relative_to("images").with_suffix(".jpg")
    save_jpeg(src, out_dir / dst_rel, MAX_IMAGE_WIDTH)
    return dst_rel.as_posix()


# ---- Baseline outputs ----

def natural_sort_key(path):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", path.name)]


def list_candidates(sample_dir, rules):
    excluded = set(rules["excluded_names"])
    found, seen = [], set()

    def add(path):
        if not path.is_file() or path.name in excluded or path in seen:
            return
        if any(token in path.name.lower() for token in rules["excluded_tokens"]):
            return
        found.append(path)
        seen.add(path)

    for name in rules["preferred"]:
        add(sample_dir / name)
    for pattern in rules["patterns"]:
        for path in sorted(sample_dir.glob(pattern), key=natural_sort_key):
            add(path)
    return found


def missing_reason(sample_dir):
    """Reason recorded by the generation run: missing.txt (harness convention) or the error line of error.log."""
    missing = sample_dir / "missing.txt"
    if missing.is_file():
        return missing.read_text().strip() or None
    log = sample_dir / "error.log"
    if log.is_file():
        for line in log.read_text().splitlines():
            key, sep, value = line.partition(":")
            if sep and key.strip() == "error":
                return value.strip()
    return None


def convert_audio(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-y", "-i", str(src),
           "-ac", "1", "-ar", "24000", "-c:a", "libmp3lame", "-b:a", "64k", str(dst)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not dst.is_file() or dst.stat().st_size == 0:
        raise RuntimeError(f"audio conversion failed for {src}: {result.stderr.strip()}")


def resolve_output(run_dir, modality, index, sample_id, out_dir):
    modality_root = run_dir / modality
    if not modality_root.is_dir():
        raise FileNotFoundError(f"{modality_root} does not exist (wrong --outputs or baseline dir?)")
    sample_dir = modality_root / str(index)
    if not sample_dir.is_dir():
        return {"status": "missing", "reason": None}
    candidates = list_candidates(sample_dir, CANDIDATE_RULES[modality])
    if not candidates:
        return {"status": "missing", "reason": missing_reason(sample_dir)}
    if len(candidates) > 1:
        raise ValueError(f"{sample_dir}: multiple {modality} candidates {[c.name for c in candidates]}; "
                         "decide how the viewer should show them")
    src = candidates[0]
    stem = Path("outputs") / run_dir.name / sample_id
    if modality == "text":
        text = src.read_text(encoding="utf-8").strip()
        return {"status": "ok", "text": text} if text else {"status": "empty", "reason": None}
    if modality == "image":
        dst_rel = Path(f"{stem}.jpg")
        save_jpeg(src, out_dir / dst_rel, MAX_OUTPUT_IMAGE_WIDTH)
        return {"status": "ok", "path": dst_rel.as_posix()}
    dst_rel = Path(f"{stem}.mp3")
    convert_audio(src, out_dir / dst_rel)
    return {"status": "ok", "path": dst_rel.as_posix()}


def judge_scores(results_root, run, indices):
    """Per-sample judge results of one run for the given dataset indices: {index: {type: summary}}."""
    by_index = {i: {} for i in indices}
    for jtype in JUDGE_TYPES:
        judges = {i: [] for i in indices}
        for stage in JUDGE_STAGES:
            data = json.loads((results_root / run / f"stage_{stage}" / jtype / "all_results.json").read_text())
            model = data["summary"]["model_name"].split("/")[-1]
            records = data["records"]
            for i, sid in indices.items():
                rec = records[i]
                rec_id = f"{rec['source']}/{rec['book']}/{rec['prev_key']}__{rec['next_key']}"
                if rec["dataset_index"] != i or rec_id != sid:
                    raise ValueError(f"{run} stage {stage} {jtype}: record {i} is {rec_id!r}, expected {sid!r}")
                if rec["status"] not in JUDGE_STATUSES:
                    raise ValueError(f"{run} stage {stage} {jtype} {sid}: unexpected status {rec['status']!r}")
                entry = {"model": model, "status": rec["status"]}
                if rec["status"] == "ok":
                    entry["scores"] = [rec["judge_eval"][key]["score"] for key, _ in JUDGE_CRITERIA[jtype]]
                    entry["mean"] = sum(entry["scores"]) / len(entry["scores"])
                judges[i].append(entry)
        for i in indices:
            valid = [j["mean"] for j in judges[i] if j["status"] == "ok"]
            by_index[i][jtype] = {"judges": judges[i], "n_valid": len(valid),
                                  "mean": sum(valid) / len(valid) if valid else None}
    return by_index


def load_baselines(path, outputs_root, manifests_root):
    config = json.loads(path.read_text())["baselines"]
    seen = set()
    for b in config:
        if set(b) != BASELINE_KEYS:
            raise ValueError(f"baseline entry keys {sorted(b)} != {sorted(BASELINE_KEYS)}")
        if b["paradigm"] not in PARADIGMS:
            raise ValueError(f"{b['dir']}: unknown paradigm {b['paradigm']!r}")
        if b["dir"] in seen:
            raise ValueError(f"duplicate baseline dir {b['dir']}")
        seen.add(b["dir"])
        if not (outputs_root / b["dir"]).is_dir():
            raise FileNotFoundError(f"baseline outputs not found: {outputs_root / b['dir']}")
        manifest = json.loads((manifests_root / b["dir"] / "dataset_manifest.json").read_text())
        b["manifest_ids"] = [r["id"] for r in manifest["records"]]
    return config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path, help="local snapshot of snu-aidas/Omni-StoryBench")
    ap.add_argument("--outputs", required=True, type=Path,
                    help="baseline generations root (evaluation_harness/baselines)")
    ap.add_argument("--results", required=True, type=Path,
                    help="evaluation results root holding <run>/dataset_manifest.json (index -> sample id) "
                         "and <run>/stage_<n>/<type>/all_results.json (judge scores)")
    ap.add_argument("--selection", type=Path, default=ROOT / "tools" / "viewer_selection.json")
    ap.add_argument("--baselines", type=Path, default=ROOT / "tools" / "viewer_baselines.json")
    ap.add_argument("--out", type=Path, default=ROOT / "viewer" / "data")
    args = ap.parse_args()

    selection = json.loads(args.selection.read_text())
    entries = selection["examples"]
    ids = [e["id"] for e in entries]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate ids in selection: {ids}")
    baselines = load_baselines(args.baselines, args.outputs, args.results)

    raw = pd.read_parquet(args.dataset / "data" / "omni_storybench.parquet")
    position = {sid: i for i, sid in enumerate(raw["id"])}
    df = raw.set_index("id")
    if not df.index.is_unique:
        raise ValueError("parquet ids are not unique")

    missing_ids = [sid for sid in ids if sid not in position]
    if missing_ids:
        raise KeyError(f"not found in parquet: {missing_ids}")
    selected = {position[sid]: sid for sid in ids}
    scores = {b["dir"]: judge_scores(args.results, b["dir"], selected) for b in baselines}

    for sub in ("images", "outputs"):
        if (args.out / sub).exists():
            shutil.rmtree(args.out / sub)  # generated output only; rebuilt from scratch below
    args.out.mkdir(parents=True, exist_ok=True)

    examples = []
    for entry in entries:
        sid = entry["id"]
        if sid not in df.index:
            raise KeyError(f"{sid} not found in parquet")
        row = df.loc[sid]
        if row["source"] not in SOURCES:
            raise ValueError(f"{sid}: source {row['source']!r} is not allowed in the public viewer")
        attribution = entry.get("attribution")
        if attribution is not None and set(attribution) != ATTRIBUTION_KEYS:
            raise ValueError(f"{sid}: attribution keys {sorted(attribution)} != {sorted(ATTRIBUTION_KEYS)}")
        if attribution is not None and "NC" in (attribution["license"] or "").upper().replace("-", " ").split():
            raise ValueError(f"{sid}: non-commercial license {attribution['license']!r} is not allowed in the public viewer")

        sample = json.loads((args.dataset / row["sample_path"]).read_text())
        check_equal("id", sid, sample["id"], sid)
        cur, nxt = sample["input"]["current_page"], sample["ground_truth"]["next_page"]
        speech = sample["ground_truth"]["speech"]
        cond = sample["next_page_condition"]
        meta = sample["input"]["book_metadata"]

        # Cross-check the per-sample JSON against the parquet index. The texts/*.txt files are not used:
        # they hold the narration plus extracted dialogue lines, and can differ from the benchmark text.
        check_equal("current image", sid, cur["image"], row["current_image"])
        check_equal("next image", sid, nxt["image"], row["next_image"])
        check_equal("current text", sid, cur["text"], row["current_text"])
        check_equal("next text", sid, nxt["text"], row["next_text"])
        for k in ("utterance", "speaker", "emotion", "speed", "pitch", "gender"):
            check_equal(f"speech {k}", sid, speech[k], row[f"speech_{k}"])
        for k in ("scene", "text_instruction", "ambient_sound"):
            check_equal(k, sid, cond[k], row[k])
        for k in ("genre", "topic", "style", "narrative_tense", "narrative_perspective"):
            check_equal(k, sid, meta[k], row[k])
        check_equal("condition characters", sid, cond["characters"], json.loads(row["condition_characters"]))
        check_equal("book characters", sid, meta["characters"], json.loads(row["book_characters"]))

        # Baseline output directories are indexed by the evaluation manifest; it must agree with the parquet.
        index = position[sid]
        outputs = {}
        for b in baselines:
            check_equal(f"{b['dir']} manifest id at index {index}", sid, sid, b["manifest_ids"][index])
            run_dir = args.outputs / b["dir"]
            outputs[b["dir"]] = {m: resolve_output(run_dir, m, index, sid, args.out)
                                 for m in ("text", "image", "speech")}
            outputs[b["dir"]]["judges"] = scores[b["dir"]][index]

        examples.append({
            "id": sid,
            "aspect": entry["aspect"],
            "rationale": entry["rationale"],
            "attribution": attribution,
            "source": {"key": row["source"], **SOURCES[row["source"]]},
            "book": sample["book"],
            "transition": sample["transition"],
            "input": {
                "current_page": {"image": copy_image(args.dataset, cur["image"], args.out), "text": cur["text"]},
                "book_metadata": meta,
            },
            "next_page_condition": cond,
            "ground_truth": {
                "next_page": {"image": copy_image(args.dataset, nxt["image"], args.out), "text": nxt["text"]},
                "speech": speech,
            },
            "outputs": outputs,
        })

    bundle = {
        "dataset": selection["dataset"],
        "dataset_revision": selection.get("dataset_revision"),
        "baselines": [{"key": b["dir"], "paradigm": b["paradigm"], "paradigm_label": PARADIGMS[b["paradigm"]],
                       "backbone": b["backbone"], "image_expert": b["image_expert"],
                       "speech_expert": b["speech_expert"], "total": b["total"]} for b in baselines],
        "judge_types": JUDGE_TYPES,
        "judge_criteria": {t: [label for _, label in crit] for t, crit in JUDGE_CRITERIA.items()},
        "examples": examples,
    }
    (args.out / "examples.json").write_text(json.dumps(bundle, indent=1, ensure_ascii=False) + "\n")

    n_images = sum(1 for _ in (args.out / "images").rglob("*.jpg"))
    statuses = {}
    for ex in examples:
        for per_mod in ex["outputs"].values():
            for mod in ("text", "image", "speech"):
                key = (mod, per_mod[mod]["status"])
                statuses[key] = statuses.get(key, 0) + 1
    print(f"wrote {len(examples)} examples, {n_images} page images, {len(baselines)} baselines to {args.out}")
    print("baseline output statuses:", dict(sorted(statuses.items())))
    judge_status = {}
    for ex in examples:
        for per_mod in ex["outputs"].values():
            for jtype, summary in per_mod["judges"].items():
                for j in summary["judges"]:
                    judge_status[(jtype, j["status"])] = judge_status.get((jtype, j["status"]), 0) + 1
    print("judge result statuses:", dict(sorted(judge_status.items())))


if __name__ == "__main__":
    main()
