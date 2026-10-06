# Omni-StoryBench runner guide

Use [README.md](README.md) to install the environment and download model assets and the latest Omni-StoryBench dataset.

## Path and environment configuration

`run_omnistorybench_reinference.sh` defaults to the active environment's `python`, input `evaluation/downloaded_dataset/`, and output `evaluation/baselines/Omni-Diffusion/`, relative to the benchmark repository. Set paths for your environment:

```bash
OMNIDIFF_PYTHON=/path/to/your/environment/bin/python \
CUDA_VISIBLE_DEVICES=0 \
bash any-to-any/Omni-Diffusion/run_omnistorybench_reinference.sh \
  --dataset-root /path/to/Omni-StoryBench \
  --output-root /path/to/generated/Omni-Diffusion
```

Select one GPU with `CUDA_VISIBLE_DEVICES`, or use `OMNIDIFF_GPU` when `CUDA_VISIBLE_DEVICES` is unset. An explicitly empty `CUDA_VISIBLE_DEVICES` is preserved. Dataset and output paths may be absolute or relative to your launch directory. The output directory must have real directory components rather than symlinks.

## Input checks

Append `--validate-only` to check the dataset and required model assets without loading the model. Append `--dry-run` to also inspect the selected output range. Neither mode generates candidates. For example:

```bash
bash any-to-any/Omni-Diffusion/run_omnistorybench_reinference.sh --validate-only
bash any-to-any/Omni-Diffusion/run_omnistorybench_reinference.sh --dry-run
```

Checks cover 900 unique dataset IDs, source counts and sentinel rows, referenced input files, model/tokenizer/decoder assets, and the required SenseVoice snapshot. The row order of `data/omni_storybench.parquet` defines indices `0..899`.

## Generation protocol

For each transition, the runner makes a planning call conditioned on the current image and story inputs, one image-generation call, and one speech-generation call. It does not retry failed image or speech generation.

The seed is set when the backend loads. Use the default full range in one process for the benchmark run. `--start` and `--end` select a half-open range, for example `--start 0 --end 10`. Running a range again replaces its candidates; it is not an automatic resume operation. Missing modalities remain missing and successful modalities are retained.

## Optional result checks

`--verify-results-only` checks for exactly 900 decodable triplets without loading the model. It requires all three canonical artifacts for every index and rejects symlinks and extra files. Natural missing model outputs can fail this strict completeness check; missing artifacts remain part of the baseline's outcome.

## Legacy JSONL entry point

`run_omni_diffusion.sh` launches `tools/run_omnibench_story_infer.py` for the nested JSONL format used by the earlier story adapter:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash any-to-any/Omni-Diffusion/run_omni_diffusion.sh \
  --jsonl /path/to/story_inputs.jsonl \
  --output_root /path/to/legacy_outputs
```

Each JSONL record contains `current_page`, `metadata`, and `next_page_condition`; its paths must point to available input files. This is a separate input format from the public Parquet dataset. The Parquet runner above provides the public benchmark's row mapping and canonical output layout.

The retained GLM-4-Voice data processor also includes optional training utilities. If using those utilities separately, set `GLM_MAIN_SPEAKER_EMBEDDING` and `GLM_GPT_SPEAKER_EMBEDDING` to your speaker-embedding files, and `GLM_SFT_AUDIO_ROOT` to your training audio directory. They are not required for the story inference commands above.
