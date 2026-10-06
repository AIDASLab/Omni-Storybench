# Omni-StoryBench evaluation

Evaluate generated text, images, and speech with the Omni-StoryBench judge panels, BERTScore, CLIP image similarity, and speech attribute classification. Run the commands below from the repository root unless stated otherwise.

## Judge groups and evaluation stages

The command-line stages correspond to **Group A, B, and C**, called **judge panels A, B, and C** in Appendix F.6 of the paper. They evaluate the same generated candidates using different judges; they are not dataset versions or successive refinements of a candidate.

| CLI stage | Paper judge panel | Text | Image | Speech | Integrated (`joint`) |
| --- | --- | --- | --- | --- | --- |
| `1` | A | `Qwen/Qwen3-30B-A3B-Instruct-2507` | `Qwen/Qwen3-VL-32B-Instruct` | `nvidia/audio-flamingo-next-hf` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` |
| `2` | B | `ByteDance-Seed/Seed-OSS-36B-Instruct` | `OpenGVLab/InternVL3_5-38B-Instruct` | `OpenMOSS-Team/MOSS-Audio-8B-Instruct` | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` |
| `3` | C | `nvidia/Llama-3_3-Nemotron-Super-49B-v1_5` | `LGAI-EXAONE/EXAONE-4.5-33B` | `moonshotai/Kimi-Audio-7B-Instruct` | `google/gemma-4-12B-it` |

`run_evaluation.sh` runs the selected groups in order. Within each group it starts four jobs in parallel, assigning one GPU each to text, image, speech, and integrated judging. Group A also schedules BERTScore, CLIP, and speech attribute classification as GPUs become available. These automatic metrics are shared across the judge groups. JSON normalization and model-assisted repair run at the end of each group, followed by summary generation after all selected groups finish.

## Environment setup

Use Linux, Conda, Bash 5.1 or newer, Git, FFmpeg, and four CUDA GPUs with sufficient memory for the assigned checkpoints. Choose GPU IDs and cache locations for your own machine. Downloaded model weights require additional disk space; access-controlled models require permission from their providers.

| Environment specification | Environment name | Workload |
| --- | --- | --- |
| [`environment-judge-group-a.yml`](environment-judge-group-a.yml) | `omnistory-judge-a` | Dataset manifest; Group A text/image/speech; BERTScore, CLIP, and JSON repair |
| [`environment-judge-group-a-omni.yml`](environment-judge-group-a-omni.yml) | `omnistory-judge-a-omni` | Group A integrated Qwen3-Omni |
| [`environment-judge-groups-bc.yml`](environment-judge-groups-bc.yml) | `omnistory-judge-bc` | Group B and C text/image/speech/integrated judges |
| [`environment-speech-metadata.yml`](environment-speech-metadata.yml) | `omnistory-speech` | Speech attribute classification |

```bash
conda env create -f evaluation/environment-judge-group-a.yml
conda env create -f evaluation/environment-judge-group-a-omni.yml
conda env create -f evaluation/environment-judge-groups-bc.yml
conda env create -f evaluation/environment-speech-metadata.yml
conda activate omnistory-judge-a
```

Keep the environment stacks separate: their model backends require different dependency versions. The Group B/C specification uses CUDA 12.9 wheels, so its driver must support that runtime. The Group A and speech specifications include the Transformers source checkout needed by their audio backends; Git is required during installation. Refer to the versions and wheel URLs in each YAML when provisioning your machine.

Models use the Hugging Face cache by default. To choose a shared cache location, set `HF_HOME` before running; for example, `export HF_HOME="/path/to/your/model-cache"`. Set `OMNISTORYBENCH_TMPDIR` to choose a temporary directory; its default is `evaluation/.tmp`. Additional environment overrides and import checks are documented in [the environment guide](references/ENVIRONMENT_GUIDE.md).

## Download the dataset

Download the latest [Omni-StoryBench dataset](https://huggingface.co/datasets/snu-aidas/Omni-StoryBench):

```bash
python evaluation/download_dataset.py
```

The downloader requires Python 3.9 or newer and uses only the standard library. Its default destination is `evaluation/downloaded_dataset/`. To choose a different location:

```bash
python evaluation/download_dataset.py --local-dir "/path/to/your/Omni-StoryBench"
```

Use the same location as `--dataset-root` for inference and evaluation. The helper verifies downloaded files and skips matching files when rerun. If an existing file differs from the latest dataset, choose a new destination to keep the older copy. Use `--list-only` to inspect the download size without downloading assets.

The dataset keeps the Hugging Face repository layout:

```text
downloaded_dataset/
├── data/omni_storybench.parquet
├── samples/<source>/<book>/<from>__<to>.json
├── metadata/<source>/<book>.json
├── images/<source>/<book>/<page>.<ext>
└── texts/<source>/<book>/<page>.txt
```

The evaluator reads 900 transitions in parquet row order, giving them indices `0..899`. Candidate generation must use that same order. The parquet contains current/next image and narration paths, story metadata, generation conditions, and target speech text and attributes. `src/dataset.py` validates the fields and referenced files, including their sample JSON records. There is no reference audio waveform.

The runner validates the dataset once and writes `dataset_manifest.json`. Other environments reuse that manifest without needing to decode parquet. Regenerate the manifest if you move or change the dataset; it contains resolved dataset paths.

## Prepare generated candidates

Pass a baseline output directory containing all three modality directories:

```text
evaluation/baselines/<model-name>/
├── text/0/generated.txt
├── image/0/generated.png
├── speech/0/generated.wav
├── text/1/generated.txt
├── image/1/generated.png
├── speech/1/generated.wav
└── ...
```

Use nonempty UTF-8 text, decodable images, and WAV speech files. Each index directory should contain only its intended candidate and, optionally, `missing.txt` explaining a missing output. An absent index directory also represents a missing candidate. The evaluator supports additional filenames through `src/common.py`, but keeping reference images, prompts, or multiple candidate files in these directories can change which file is selected. The speech attribute classifier selects the first sorted WAV file for each index.

## Run all three judge groups

```bash
conda activate omnistory-judge-a
export CUDA_DEVICE_ORDER=PCI_BUS_ID

bash evaluation/run_evaluation.sh \
  --results-root evaluation/baselines/AnyGPT \
  --dataset-root evaluation/downloaded_dataset \
  --gpus 0,1,2,3 \
  --stages 1,2,3 \
  --run-name AnyGPT
```

Replace the input paths, run name, and four GPU IDs with your own. The GPU list assigns devices to text, image, speech, and integrated jobs in that order. The runner sets each job's `CUDA_VISIBLE_DEVICES`; select the devices through `--gpus`. Stage-end JSON repair uses the first listed GPU. Each judge uses tensor parallelism 1.

Use a distinct plain directory name for `--run-name` to keep earlier results. By default, the runner uses the basename of `--results-root`. Outputs are stored under `evaluation/results/<run-name>/`; existing outputs with the same name can be overwritten.

For a small evaluation, add `--limit 1`. For a scheduling and input check without loading models, add `--dry-run --limit 1` and use a separate run name. A dry run still validates the complete dataset and writes placeholder outputs. `--start-index` is inclusive and `--end-index` is exclusive in the shell runner.

To evaluate a single group, use `--stages 1`, `--stages 2`, or `--stages 3`. The default is Group A (`1`). Automatic metrics are computed only when Group A is included. To use custom Conda environment names, supply the options listed in the [environment guide](references/ENVIRONMENT_GUIDE.md). All options are available with:

```bash
bash evaluation/run_evaluation.sh --help
```

## Direct entrypoints

For individual jobs, prepare the dataset manifest first:

```bash
conda run --no-capture-output -n omnistory-judge-a \
  python evaluation/scripts/prepare_dataset_manifest.py \
  --dataset-root evaluation/downloaded_dataset \
  --output-manifest evaluation/results/manual/dataset_manifest.json

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n omnistory-judge-bc \
  python evaluation/scripts/evaluate_modality.py \
  --stage 2 --modality text \
  --dataset-root evaluation/downloaded_dataset \
  --dataset-manifest-path evaluation/results/manual/dataset_manifest.json \
  --results-root evaluation/baselines/AnyGPT \
  --output-json evaluation/results/manual/stage_2/text/all_results.json \
  --summary-json evaluation/results/manual/stage_2/text/summary.json \
  --limit 1
```

Use `scripts/evaluate_joint.py --stage <1|2|3>` for integrated evaluation; it has no `--modality` argument. `scripts/evaluate_aux_metric.py` accepts `--metric bertscore` or `--metric clip`. `scripts/evaluate_speech_metadata.py` runs speech attribute classification. Each script provides its complete arguments through `--help`. Output and manifest paths must be inside `evaluation/`; dataset and candidate inputs can be elsewhere.

## Outputs and scores

```text
evaluation/results/<run-name>/
├── dataset_manifest.json
├── stage_<n>/<text|image|speech|joint>/all_results.json
├── stage_<n>/<text|image|speech|joint>/summary.json
├── stage_<n>/<text|image|speech|joint>/log.txt
├── stage_<n>/repair_manifest.json
├── stage_1/<bertscore|clip>/all_results.json
├── stage_1/<bertscore|clip>/summary.json
├── stage_1/speech_classifier/metadata_comparison.json
├── stage_1/speech_classifier/class_frequency_summary.json
├── final_summary.json
├── final_table.csv
└── final_table.md
```

- **Judge scores:** each judge returns four integer rubric scores from 1 to 10 with rationales, covering metadata alignment, continuity, generation conditions, and ground-truth consistency. Integrated judging requires all three candidate modalities.
- **BERTScore:** text F1 from `BERTScorer(lang="en", rescale_with_baseline=False)`.
- **CLIP:** cosine similarity between normalized RGB image features from `openai/clip-vit-base-patch32`.
- **Speech attributes:** exact-match accuracy for normalized emotion, speed, pitch, and apparent gender labels, averaged over the four attributes. Exact match across all four attributes is also reported.

`final_table.csv` and `final_table.md` contain one row per judge group. The group summaries use zero-filled averages: unavailable or unscorable criterion values contribute zero, and the denominator is the number of processed records (900 for a complete run). The paper's main results use valid-only averages (Appendix E.5), while Appendix F.4 reports a separate zero-filled analysis. The tables include the automatic metrics in the Group A row. They do not combine the three group rows into the paper's Total Average. Keep the detailed per-record scores and validity statuses when computing a different aggregation policy.

To rebuild the group tables from existing results:

```bash
python evaluation/scripts/aggregate_results.py \
  --run-dir evaluation/results/AnyGPT --stages 1,2,3 \
  --output-json evaluation/results/AnyGPT/final_summary.json \
  --output-csv evaluation/results/AnyGPT/final_table.csv \
  --output-md evaluation/results/AnyGPT/final_table.md
```

For optional DreamSim image-distance evaluation, follow the [DreamSim project page](https://dreamsim-nights.github.io/) and [official repository](https://github.com/ssundaram21/dreamsim). Compare each generated image with its corresponding next-page ground-truth image using the provided preprocessing and model; lower distance indicates greater similarity.

## Troubleshooting

- A missing `pyarrow` error during dataset preparation means the manifest command needs the `omnistory-judge-a` environment. Subsequent jobs should receive the prepared manifest.
- For model-load or GPU-memory errors, check the relevant job's `log.txt`, checkpoint access, available device memory, and environment selection. `--vllm-max-model-len` and `--vllm-gpu-memory-utilization` expose the runner's memory settings.
- Judge responses that cannot be parsed remain unscorable. The repair manifest records repair outcomes, and the stage gate stops a run when systemic inference errors meet its threshold. Inspect these statuses separately from missing generated candidates.
- Invalid text encoding or undecodable images/audio should be corrected before running. BERTScore and CLIP process batches; a failed batch can mark multiple records as errors. The speech classifier stops at its first inference exception by default.
- If a child job fails, inspect and stop any remaining jobs from that run before restarting. Use a new run name to preserve the earlier diagnostics.
