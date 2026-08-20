"""Derive a *_voxcpm baseline from a finished *_parler baseline.

The backbone outputs (text/image/raw) are identical between the two TTS
variants, so they are copied (hardlinked when possible) from the parler run.
Only speech is regenerated, through the same T2SGenerator/VoxCPMImpl the
native pipeline uses, so the wavs match what a from-scratch VoxCPM run would
produce.

Usage (from baseline/, in an env with voxcpm, e.g. mmada_clean):
    python -m src.parler_to_voxcpm \
        --src .../baselines/emu3_parler --dst .../baselines/emu3_voxcpm
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from src.models import SpeechMetadata, SpeakerMetadata

T2S_MODEL = "openbmb/VoxCPM2"
WAV_NAME = "generated.wav"
COPY_MODALITIES = ("text", "image", "raw")
# Failed source samples carry error.log in the speech dir too; mirror everything.
ALL_MODALITIES = ("text", "image", "speech", "raw")

# Same fallbacks the configs use when the backbone omitted a field.
DEFAULT_SPEAKER = {
    "name": "Narrator",
    "emotion": "neutral",
    "speed": "normal",
    "pitch": "normal",
    "gender": "female",
}


def load_speech_metadata(raw_dir: Path) -> SpeechMetadata | None:
    """Speech metadata from the backbone's raw output, falling back to the quote.

    Returns None for samples that failed in the source run (no raw json).
    """
    raw_json_path = next(iter(sorted(raw_dir.glob("*.json"))), None)
    if raw_json_path is None:
        return None
    try:
        with raw_json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[Error] Failed to read {raw_json_path}: {e}")
        return None

    meta = data.get("speech_metadata")
    if isinstance(meta, dict):
        try:
            return SpeechMetadata.model_validate(meta)
        except Exception:
            pass

    quote = data.get("quote")
    if isinstance(quote, str) and quote.strip():
        return SpeechMetadata(speaker=SpeakerMetadata(line=quote.strip(), **DEFAULT_SPEAKER))

    return None


def link_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_sample_files(src_root: Path, dst_root: Path, index: str, modalities=COPY_MODALITIES):
    for modality in modalities:
        src_dir = src_root / modality / index
        for f in sorted(src_dir.glob("*")):
            if f.is_file():
                link_or_copy(f, dst_root / modality / index / f.name)


def collect_indices(src_root: Path, start: int, until: int | None) -> list[str]:
    indices = sorted(
        (p.name for p in (src_root / "raw").iterdir() if p.is_dir() and p.name.isdigit()),
        key=int,
    )
    return [i for i in indices if int(i) >= start and (until is None or int(i) < until)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Finished *_parler baseline root")
    parser.add_argument("--dst", required=True, help="*_voxcpm baseline root to populate")
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--until", type=int, default=None, help="End index (exclusive)")
    parser.add_argument("--repair", action="store_true", help="Skip samples whose wav already exists")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda", help="Device for VoxCPM")
    args = parser.parse_args()

    src_root = Path(args.src).expanduser().resolve()
    dst_root = Path(args.dst).expanduser().resolve()
    if not (src_root / "raw").is_dir():
        sys.exit(f"No raw/ under {src_root}; expected a finished harness-format baseline.")
    if src_root == dst_root:
        sys.exit("--src and --dst must differ; this script derives a new baseline.")

    indices = collect_indices(src_root, args.start, args.until)
    print(f"[Plan] {src_root.name} -> {dst_root.name}: {len(indices)} samples "
          f"(range {args.start}..{args.until if args.until is not None else 'end'})")

    jobs = []
    skipped_missing = []
    for index in indices:
        meta = load_speech_metadata(src_root / "raw" / index)
        if meta is None:
            skipped_missing.append(index)
            continue
        jobs.append((index, meta))

    if skipped_missing:
        print(f"[Warn] {len(skipped_missing)} samples failed in the source run "
              f"(no raw json); their error files are copied as-is: "
              f"{', '.join(skipped_missing[:10])}{' ...' if len(skipped_missing) > 10 else ''}")

    if args.dry_run:
        for index, meta in jobs[:20]:
            spk = meta.speaker
            print(f"[DryRun] {index}: ({spk.gender}/{spk.emotion}/{spk.speed}/{spk.pitch}) {spk.line[:100]}")
        return

    for index in skipped_missing:
        copy_sample_files(src_root, dst_root, index, modalities=ALL_MODALITIES)

    from src.t2s_generator import T2SGenerator
    print(f"[Model] Loading {T2S_MODEL}...")
    t2s = T2SGenerator(T2S_MODEL, device=args.device)

    done = failed = skipped_existing = 0
    for index, meta in tqdm(jobs):
        wav_path = dst_root / "speech" / index / WAV_NAME
        try:
            copy_sample_files(src_root, dst_root, index)
            if args.repair and wav_path.exists():
                skipped_existing += 1
                continue
            t2s.forward(meta, str(wav_path))
            done += 1
        except Exception as e:
            failed += 1
            print(f"[Error] index {index}: {type(e).__name__}: {e}")

    print(f"[Done] generated={done} skipped_existing={skipped_existing} "
          f"failed={failed} error_copied={len(skipped_missing)}")


if __name__ == "__main__":
    main()
