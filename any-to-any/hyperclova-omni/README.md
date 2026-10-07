# HyperCLOVA X Omni inference for Omni-StoryBench

Generate the next-page narration, illustration, and spoken character line with
[HyperCLOVAX-SEED-Omni-8B](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B)
served by [NAVER's OmniServe](https://github.com/NAVER-Cloud-HyperCLOVA-X/OmniServe).
This directory contains the benchmark HTTP client and launcher. Model loading,
vision/audio processing, and Docker deployment are handled by OmniServe.

## Prepare the OmniServe server

Follow the [official OmniServe Quick Start](https://github.com/NAVER-Cloud-HyperCLOVA-X/OmniServe#quick-start)
to install Docker prerequisites, download and convert the Omni checkpoint, and
configure model paths, GPU assignments, and S3-compatible storage. Start the
**Track B** services from the separate OmniServe checkout:

```bash
docker compose --profile track-b build
docker compose --profile track-b up -d
docker compose --profile track-b ps
```

Use its chainer endpoint, normally `http://localhost:8000/b/v1`, with served model
name `track_b_model`. If your deployment publishes another port, supply its URL
with `--base-url`; the local benchmark deployment used port `8100`.

Object storage must be reachable by both the containers and this client. The
chainer returns signed URLs for generated images and audio; the client downloads
those URLs unchanged. A working external S3-compatible service or separately
configured MinIO deployment is required. The upstream Compose file does not
include the local benchmark deployment's MinIO additions.

## Prepare the client and dataset

Use Python 3.10 or newer in an environment separate from the model server:

```bash
python -m pip install openai requests tqdm pyarrow
```

From the Omni-StoryBench repository root, download the public dataset:

```bash
python evaluation/download_dataset.py
```

The runner defaults to `evaluation/downloaded_dataset/`. Its
`data/omni_storybench.parquet` row order defines output indices `0..899`.
Validation checks 900 unique IDs, source counts, sentinel rows, and referenced
input files. It reads only current-page inputs, book metadata, next-page
conditions, and input-path columns; reference columns and sample JSON contents
are never loaded. The two known Parquet-materialized empty `speech_intent` fields
are normalized consistently with the other any-to-any adapters.

Expose the dataset root through an HTTP server so the encoder containers can
fetch current-page images. Run this in another terminal from the repository root:

```bash
python -m http.server 8008 \
  --directory evaluation/downloaded_dataset \
  --bind 0.0.0.0
```

Set `HYPERCLOVA_IMAGE_HTTP_ROOT` to `http://<host-address>:8008`, replacing
`<host-address>` with an address reachable from Docker. The URL must serve the
same dataset root supplied to inference. Container `localhost` points to the
container itself. The current chainer's encoder route requires HTTP(S) image
URLs, so this client uses the dataset HTTP server.

## Run inference

Run these commands from the Omni-StoryBench repository root:

```bash
# Inspect options and validate the full dataset without contacting a server.
bash any-to-any/hyperclova-omni/run_omnistorybench.sh --help
bash any-to-any/hyperclova-omni/run_omnistorybench.sh --validate-only

# Replace the address with your Docker-reachable host address.
export HYPERCLOVA_IMAGE_HTTP_ROOT="http://<host-address>:8008"

# Preview the first sample's prompt, image URL, and output paths.
bash any-to-any/hyperclova-omni/run_omnistorybench.sh --dry-run --end 1

# Generate all 900 samples using the upstream default endpoint.
bash any-to-any/hyperclova-omni/run_omnistorybench.sh --start 0 --end 900
```

Both `--validate-only` and `--dry-run` make no network calls and write no outputs.
They require `pyarrow`; the other client dependencies are needed for generation.
For a one-sample generation check, use `--start 0 --end 1` with a separate result
directory. `--start` is inclusive and `--end` is **exclusive**.

Override paths and endpoints as needed:

```bash
bash any-to-any/hyperclova-omni/run_omnistorybench.sh \
  --dataset-root /path/to/Omni-StoryBench \
  --results-root /path/to/HyperClovaX-Omni-results \
  --base-url http://localhost:8100/b/v1 \
  --image-http-root "http://<host-address>:8008"
```

Paths may be absolute or relative to your launch directory. The Python entrypoint
also works directly: `python any-to-any/hyperclova-omni/run_omnistorybench.py`.
The launcher honors `HYPERCLOVA_PYTHON` to select an interpreter. Persistent runner
defaults are available through `HYPERCLOVA_DATASET_ROOT`, `HYPERCLOVA_RESULTS_ROOT`,
`HYPERCLOVA_BASE_URL`, `HYPERCLOVA_MODEL`, `HYPERCLOVA_IMAGE_HTTP_ROOT`, and
`HYPERCLOVA_API_KEY`. Explicit command-line arguments take precedence. The default
API key is `not-needed`; set one only if your deployment requires authentication.
Client execution does not require a GPU.

## Generation protocol and outputs

The prompts and model requests are adapted from the benchmark client
`generate_omnistory.py` in the existing OmniServe deployment. Each sample uses:

1. A current-image-conditioned text request producing `NARRATION` and `SPEECH`.
2. An illustration request using the current image, generated narration, and
   story conditions, with the `t2i_model_generation` tool.
3. A speech request reading the generated spoken line, with voice directions
   taken from next-page conditions. The response's `audio.data` is a URL-safe
   base64-encoded download URL, rather than waveform bytes.

| Setting | Default | Override |
| --- | --- | --- |
| Reasoning | Enabled | `--no-thinking` |
| Reasoning token budget | 1024 | `--thinking-token-budget`; `0` means uncapped |
| Text / image / speech maximum tokens | 2048 / 7000 / 2048 | `--text-max-tokens`, `--image-max-tokens`, `--speech-max-tokens` |
| Total image attempts | 10, including the first | `--image-retries` |
| Delay between image attempts | 1.5 seconds | `--retry-sleep` |
| Model-request / artifact-download timeout | 600 / 120 seconds | `--request-timeout`, `--download-timeout` |

Reasoning requests retain special tokens for the server's reasoning parser.
Text candidates exclude reasoning. The text parser retains the existing handling
of decorated labels, labels on their own lines, and multiple narration/speech
pairs. A missing spoken line is not replaced with narration.

Text and speech each use one request. SDK HTTP retries are disabled. The image
loop can retry missing tool calls, model/decoder failures, and failed or invalid
downloads; each attempt is recorded. Set `--image-retries 1` for one image
attempt. Temperature, top-p, and random seed are not overridden by the client;
they use the server defaults. Restarted ranges and retries can change results.

The default result directory is `evaluation/baselines/HyperClovaX-Omni/`:

```text
<results-root>/
├── text/<index>/generated.txt
├── image/<index>/generated.png
├── speech/<index>/generated.wav
└── _meta/
    ├── index_map.json
    ├── run_summary.json
    └── samples/<index>.json
```

Each successful modality is saved immediately. If a later modality fails,
successful candidates remain and the next sample is attempted. A failed pass
leaves its candidate absent; errors, prompts, raw text, reasoning, and image
attempts are stored in `_meta/`, outside the modality folders. PNG and WAV
headers are checked before saving downloads. A run exits with status `1` if any
selected sample fails; setup/validation errors exit with status `2`.

Complete samples are skipped unless `--overwrite` is supplied. Incomplete
samples restart from the text stage and replace only this runner's three known
candidate files. Skipped samples retain their existing per-sample diagnostics;
the run summary describes the current invocation. Output paths must not contain
symlinks. Use a fresh result directory for a separate experiment.

Pass the directory containing `text/`, `image/`, and `speech/` to the
[evaluation workflow](../../evaluation/README.md) as `--results-root`, using the
same dataset snapshot. No root-runner configuration is needed for this baseline.
