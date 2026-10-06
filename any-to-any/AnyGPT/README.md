# AnyGPT inference for Omni-StoryBench

Run the AnyGPT baseline to generate the next story page as text, image, and speech. The benchmark adapter generates these modalities in three successive passes for each sample and writes the files expected by the evaluation code.

## Original work and code provenance

- Paper: [AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling](https://arxiv.org/abs/2402.12226).
- Official implementation: [OpenMOSS/AnyGPT](https://github.com/OpenMOSS/AnyGPT).
- Project page: [AnyGPT](https://junzhan2000.github.io/AnyGPT.github.io/).

Please cite the original AnyGPT paper when using this model. The official repository provides the authors' citation and license information and acknowledges its SpeechGPT, Vicuna, SpeechTokenizer, SoundStorm, and SEED dependencies.

| Component | Origin and role |
| --- | --- |
| `anygpt/src/infer/cli_infer_chat_model.py`, `voice_clone.py`, `pre_post_process.py`, and `anygpt/src/m_utils/` | Adapted from the official AnyGPT inference and prompt utilities. The chat helper supports a configurable diffusion checkpoint and forwards the speech tokenizer settings used by the benchmark. |
| `seed2/`, `soundstorm_speechtokenizer/`, `config/`, `requirements.txt` | Upstream AnyGPT tokenizer, decoder, generation configuration, and dependency components used by the baseline. |
| `models/` configuration and tokenizer files | Runtime assets associated with the AnyGPT chat, SpeechTokenizer, and diffusion checkpoints. Download model weights separately. |
| `anygpt/src/infer/run_omnistorybench_hf.py`, `run_omnistorybench_hf.sh` | Our Omni-StoryBench adapter: public dataset loading, canonical sample indices, story prompts, three-pass generation, and evaluation-compatible output files. |
| `anygpt/src/infer/run_omnibench_story_batch.py`, `full_inference_anygpt_899.sh` | Our earlier JSONL batch adapter; see the legacy format notes below. |
| This README and `OMNISTORYBENCH_REINFERENCE.md` | Our benchmark usage and generation protocol documentation. |

## Environment and checkpoints

Follow the [official AnyGPT installation instructions](https://github.com/OpenMOSS/AnyGPT#installation) to prepare and activate the model environment. Install `pyarrow` in that environment for the public dataset's Parquet reader:

```bash
conda activate AnyGPT
python -m pip install pyarrow
```

Download the model weights following the official instructions. The launcher uses the following layout inside this directory:

```text
models/
├── anygpt/chat/                         # fnlp/AnyGPT-chat
│   ├── config.json
│   ├── generation_config.json
│   ├── added_tokens.json
│   ├── special_tokens_map.json
│   ├── tokenizer.model
│   ├── tokenizer_config.json
│   └── pytorch_model.bin
├── seed-tokenizer-2/seed_quantizer.pt
├── speechtokenizer/
│   ├── config.json
│   └── ckpt.dev
├── soundstorm/speechtokenizer_soundstorm_mls.pt
└── stable-diffusion-2-1-unclip/          # component configurations and weights
```

Model sources: [fnlp/AnyGPT-chat](https://huggingface.co/fnlp/AnyGPT-chat), [fnlp/AnyGPT-speech-modules](https://huggingface.co/fnlp/AnyGPT-speech-modules), [AILab-CVC/seed-tokenizer-2](https://huggingface.co/AILab-CVC/seed-tokenizer-2), and [sd2-community/stable-diffusion-2-1-unclip](https://huggingface.co/sd2-community/stable-diffusion-2-1-unclip). Keep each checkpoint's configuration, tokenizer, and weights together. The AnyGPT model preflight expects `pytorch_model.bin`.

Initialization also loads `facebook/encodec_32khz`, `facebook/encodec_24khz`, `bert-base-uncased`, and the EVA vision checkpoint used by SEED. Allow these downloads on the first run, or populate the selected caches beforehand. The launcher defaults to `.cache/` inside this directory; set `ANYGPT_CACHE_ROOT` to a writable cache directory in your environment. Existing Hugging Face cache overrides can take precedence over `HF_HOME`. Set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` only after the required assets are cached.

You can keep checkpoints elsewhere by supplying `--model-name-or-path`, `--image-tokenizer-path`, `--speech-tokenizer-path`, `--speech-tokenizer-config`, `--soundstorm-path`, and `--diffusion-model-path`. Absolute paths are accepted; relative checkpoint and generation configuration paths are resolved from this AnyGPT directory.

## Download the public dataset

From the repository root, download the latest [Omni-StoryBench dataset](https://huggingface.co/datasets/snu-aidas/Omni-StoryBench):

```bash
python evaluation/download_dataset.py
```

The shared default destination is `evaluation/downloaded_dataset/`. To choose another location:

```bash
python evaluation/download_dataset.py --local-dir /path/to/Omni-StoryBench
```

Replace `/path/to/Omni-StoryBench` with your dataset directory and supply the same path to inference with `--dataset-root`. Keep the downloaded relative layout intact:

```text
<dataset-root>/
├── data/omni_storybench.parquet
└── ... image, text, sample, and metadata files referenced by the rows
```

The canonical sample index is the Parquet row number, `0..899`. The adapter checks the benchmark's 900 unique samples, source counts, and canonical sentinel IDs before generation. It reads current-page inputs, book metadata, and next-page instructions; it does not read ground-truth columns or per-sample JSON contents.

Required columns are `id`, `source`, `book`, `from_key`, `to_key`, `current_image`, `current_text`, `current_text_path`, `sample_path`, `metadata_path`, `genre`, `topic`, `style`, `narrative_tense`, `narrative_perspective`, `book_characters`, `scene`, `text_instruction`, `ambient_sound`, and `condition_characters`. File references must be relative to the dataset root and remain within it. Character fields can be lists of objects or JSON strings encoding those lists.

## Run inference

Activate the AnyGPT environment and run these commands from `any-to-any/AnyGPT/`:

```bash
# Inspect the command-line options.
python anygpt/src/infer/run_omnistorybench_hf.py --help

# Check the dataset and required checkpoint/configuration files.
bash run_omnistorybench_hf.sh --validate-only

# Choose an available GPU and process all 900 samples.
CUDA_VISIBLE_DEVICES=0 bash run_omnistorybench_hf.sh \
  --start-index 0 --end-index 899
```

Set `CUDA_VISIBLE_DEVICES` to the GPU you want to use. The launcher honors an existing value; otherwise it uses `ANYGPT_GPU`, defaulting to `0`. The underlying model uses `device_map="auto"`, so expose only the devices you intend to allocate. The launcher uses `python` from the active environment; set `ANYGPT_PYTHON=/path/to/environment/bin/python` to select another interpreter.

The default input is `evaluation/downloaded_dataset/` and the default output is `evaluation/baselines/AnyGPT/`, both relative to the repository root. For custom locations:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_omnistorybench_hf.sh \
  --dataset-root /path/to/Omni-StoryBench \
  --results-root /path/to/AnyGPT-results
```

You can also set `ANYGPT_DATASET_ROOT` and `ANYGPT_RESULTS_ROOT` as persistent launcher defaults. Explicit command-line arguments take precedence. The output path and artifact tree must not contain symlinks. Use a dedicated result directory with only the expected modality/sample files.

Outputs follow this layout:

```text
<results-root>/
├── text/<index>/generated.txt
├── image/<index>/generated.png
└── speech/<index>/generated.wav
```

Pass this result root to the [evaluation workflow](../../evaluation/README.md). An optional strict audit checks for all 900 complete triplets:

```bash
bash run_omnistorybench_hf.sh --verify-results-only
# For a custom run, include its --dataset-root and --results-root arguments.
```

Both validation modes require the dataset and checkpoint/configuration files. They do not initialize the model or generate outputs.

## Generation protocol and output handling

For each sample, the adapter encodes the current image, generates narration and a candidate spoken line, generates the next illustration, then generates speech. Text and image passes use the encoded current image; speech uses the textual story context.

| Pass | Configuration | Sampling | Maximum new tokens |
| --- | --- | --- | --- |
| Text | `config/text_generate_config.json` | temperature 1, top-p 1 | 100 |
| Image | `config/image_generate_config.json` | temperature 1, top-p 1 | 200 |
| Speech | `config/speech_generate_config.json` | temperature 1, top-p 1, repetition penalty 1.05 | 1500 |

All three use sampling with a minimum of 10 new tokens. The speech configuration includes `vc_steps=4`. The adapter does not set a random seed, retry missing modalities, or automatically resume. Use one continuous run over `0..899` to follow the benchmark protocol; independently restarted ranges change the sequence of random draws.

Each modality is saved immediately. If a later pass fails, earlier files remain and generation continues with the next sample. A response without image or speech tokens leaves that file missing. When multiple segments are produced, all are decoded in order and only the first is saved. Check the printed failure/missing-modality counts and the artifact tree after the run; the process can finish with incomplete samples. Use a fresh output directory for a new experiment to avoid mixing old and new outputs.

## Legacy JSONL adapter

`full_inference_anygpt_899.sh` runs the earlier JSONL adapter. Supply `--dataset-path /path/to/stories.jsonl`; records must contain `current_page.text`, `current_page.image_path`, `metadata.metadata`, and `next_page_condition.condition_text` or `.condition_json`. It accepts `--results-root` and the model/tokenizer checkpoint flags listed by its `--help`. For this adapter, select the diffusion checkpoint with `ANYGPT_DIFFUSION_MODEL_PATH=/path/to/diffusion-checkpoint`; it does not accept `--diffusion-model-path`. Its default output directory is `evaluation/baselines/AnyGPT_legacy/`.

This adapter writes legacy filenames (`generated_text.txt`, `image_00.jpg`, `speech_00.wav`) and diagnostic files. Use `run_omnistorybench_hf.sh` for the public dataset and canonical evaluation artifact format.
