# Omni-Diffusion on Omni-StoryBench

Generate a story continuation as text, an image, and speech from the current page and the next-page conditions. The benchmark runner uses a planning call followed by separate image and speech generation calls.

## Original work and code provenance

- Paper: [Omni-Diffusion: Unified Multimodal Understanding and Generation with Masked Discrete Diffusion](https://arxiv.org/abs/2603.06577).
- Official repository: [VITA-MLLM/Omni-Diffusion](https://github.com/VITA-MLLM/Omni-Diffusion).
- Project: [omni-diffusion.github.io](https://omni-diffusion.github.io/).

Please cite the original Omni-Diffusion paper when using the model, following the official repository's citation guidance.

| Component | Origin and role |
| --- | --- |
| `setup.py`, `requirements_ds_gpu.txt`, `omni_diffusion/constants.py`, `omni_diffusion/tokenizer.py`, and `omni_diffusion/data/{build,data_collator,dataset_base,dataset_qwen2,utils}.py` | Unchanged files from the official Omni-Diffusion repository. |
| Dream configuration, generation and tokenization modules; MAGVIT `modeling_magvitv2.py`, `common_modules.py`, `misc.py` | Unchanged upstream model components under `omni_diffusion/models/`. |
| `tools/inference.py`, the two modality tokenizer wrappers, image processor, Dream model/SenseVoice loaders, MAGVIT `modeling_utils.py`, and package import initializers | Based on upstream code with compatibility adaptations used by the benchmark, including local imports, attention-backend selection and model loading. |
| `models/` | Configuration, tokenizer and local model-code assets accompanying the pretrained models. The local SenseVoice loader pins its model dependency. Download the weight files separately as below. |
| `third_party/GLM-4-Voice/` and nested `Matcha-TTS/` | Third-party speech components and their licenses. Optional training paths in the GLM data processor are configurable through environment variables. |
| `tools/run_omnibench_story_infer.py`, `tools/run_omnistorybench_reinference.py`, both shell launchers, and the benchmark guides | Omni-StoryBench additions: story prompts, dataset mapping, the separate-modality inference loop, artifact export, and usage instructions. The Parquet runner supports configurable dataset and output directories. |

## Environment setup

Prepare a CUDA-enabled environment using the [official installation instructions](https://github.com/VITA-MLLM/Omni-Diffusion#requirements-and-installation). Install this directory's package and requirements in that environment:

```bash
cd any-to-any/Omni-Diffusion
python -m pip install -r requirements_ds_gpu.txt
python -m pip install -e .
```

`requirements_ds_gpu.txt` specifies Transformers 4.51.3. The required GLM-4-Voice and Matcha-TTS source dependencies are included under `third_party/`. Run the following commands from the benchmark repository root unless otherwise indicated.

## Download model assets

Download the weights into the directories below. Keep the provided configuration and Python files when adding the weights.

| Asset | Source | Local directory |
| --- | --- | --- |
| Omni-Diffusion | [lijiang/Omni-Diffusion](https://huggingface.co/lijiang/Omni-Diffusion) | `any-to-any/Omni-Diffusion/models/Omni-Diffusion/` |
| Speech tokenizer | [THUDM/glm-4-voice-tokenizer](https://huggingface.co/THUDM/glm-4-voice-tokenizer) | `any-to-any/Omni-Diffusion/models/THUDM/glm-4-voice-tokenizer/` |
| Speech decoder | [THUDM/glm-4-voice-decoder](https://huggingface.co/THUDM/glm-4-voice-decoder) | `any-to-any/Omni-Diffusion/models/THUDM/glm-4-voice-decoder/` |
| Image tokenizer | [showlab/magvitv2](https://huggingface.co/showlab/magvitv2) | `any-to-any/Omni-Diffusion/models/showlab/magvitv2/` |
| SenseVoice audio encoder | [FunAudioLLM/SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) | Repository-local Hugging Face cache |

With the Hugging Face CLI installed in your environment:

```bash
OMNIDIFF_ROOT="$PWD/any-to-any/Omni-Diffusion"
hf download lijiang/Omni-Diffusion \
  --include '*.safetensors' --local-dir "$OMNIDIFF_ROOT/models/Omni-Diffusion"
hf download THUDM/glm-4-voice-tokenizer \
  --include 'model.safetensors' --local-dir "$OMNIDIFF_ROOT/models/THUDM/glm-4-voice-tokenizer"
hf download THUDM/glm-4-voice-decoder \
  --include 'flow.pt' 'hift.pt' --local-dir "$OMNIDIFF_ROOT/models/THUDM/glm-4-voice-decoder"
hf download showlab/magvitv2 \
  --include 'pytorch_model.safetensors' --local-dir "$OMNIDIFF_ROOT/models/showlab/magvitv2"
HF_HUB_OFFLINE=0 hf download FunAudioLLM/SenseVoiceSmall \
  --revision 3847d57b6bdf2dd8875cb1508d2af43d80a16bf7 \
  --cache-dir "$OMNIDIFF_ROOT/.cache/huggingface/hub"
```

The SenseVoice revision is a model dependency required by the loader. Its snapshot must include `config.yaml`, `configuration.json`, `chn_jpn_yue_eng_ko_spectok.bpe.model`, `am.mvn`, and `model.pt`. Inference uses the downloaded assets offline.

## Download the dataset

Download the latest [Omni-StoryBench dataset](https://huggingface.co/datasets/snu-aidas/Omni-StoryBench):

```bash
python evaluation/download_dataset.py
```

The default location is `evaluation/downloaded_dataset/`. To use another location, run `python evaluation/download_dataset.py --local-dir /path/to/Omni-StoryBench` and pass the same directory as `--dataset-root` below.

The runner reads `data/omni_storybench.parquet` and the referenced input files from that directory. Parquet row `i` maps to output index `i`; retain the supplied row order. The benchmark contains 900 transitions. Its inputs include the current image and text, book metadata, and next-page conditions. Reference next-page images, text, and speech are excluded from inference prompts.

## Run inference

Activate your inference environment, choose one GPU available on your machine, and run:

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash any-to-any/Omni-Diffusion/run_omnistorybench_reinference.sh
```

Replace `0` with your chosen GPU ID. The backend uses logical `cuda:0` within the selected visible device. To choose an interpreter, input location, or output location explicitly:

```bash
OMNIDIFF_PYTHON=/path/to/your/environment/bin/python \
CUDA_VISIBLE_DEVICES=0 \
  bash any-to-any/Omni-Diffusion/run_omnistorybench_reinference.sh \
    --dataset-root /path/to/Omni-StoryBench \
    --output-root /path/to/generated/Omni-Diffusion
```

Relative paths are resolved from the directory where you launch the command. `OMNIDIFF_DATASET_ROOT` and `OMNIDIFF_OUTPUT_ROOT` can also set defaults; command-line arguments take precedence. The launcher uses `python` from the active environment unless `OMNIDIFF_PYTHON` is set. Runtime caches and temporary generation files live under `any-to-any/Omni-Diffusion/.cache/`.

The default range is `[0, 900)`, seed 42, and dtype `bfloat16`. Text uses 192 maximum tokens and 192 steps; image generation uses 260 tokens and 260 steps; speech uses 64 tokens and 32 steps. Use `--help` to list the options. Run the full range in one process to preserve the benchmark's sequential random-number stream. Rerunning a range generates it again.

## Outputs and evaluation

The default output directory is `evaluation/baselines/Omni-Diffusion/`:

```text
Omni-Diffusion/
├── image/{index}/generated.png
├── speech/{index}/generated.wav
└── text/{index}/generated.txt
```

Each successful modality is saved independently. If a model call produces no image or speech, that artifact remains absent; the runner does not fill it with a placeholder or retry it. Keep console logs outside the artifact directory.

Use this directory as `--results-root` and the downloaded dataset directory as `--dataset-root` in the [evaluation instructions](../../evaluation/README.md). See [the runner guide](OMNISTORYBENCH_REINFERENCE.md) for input checks, generation behavior, and the legacy JSONL entry point.
