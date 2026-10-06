# Dynin-Omni_nips on Omni-StoryBench

`run_storybench.py` generates text, image, and speech **jointly in one masked-diffusion sequence** for each Omni-StoryBench transition. It uses the `snu-aidas/Dynin-Omni_nips` weights and the reserved TI2TIS task token ID `126107`. The checkpoint's config reports `model_type: omada`; the public Dynin-Omni model source loads its weights and gave identical logits to the local OMaDA model in a forward-pass comparison.

This is a **reconstructed inference implementation**, not the original paper script. Its sampling rule, target-length policy, fixed speech style, and evaluation protocol have not been verified against the paper run. Do not report these outputs as a reproduction of Table 3 (`Overall 7.39`) or Appendix Table 14 (`non-block Joint, 128 NFE, Overall 7.93`). Those are different table entries. The checkpoint's training overlap with this benchmark is also unverified.

## Setup

Use Python 3.10, a CUDA GPU with enough free memory for the 8B model and the image/speech decoders, and a clean output directory. From a parent directory, clone this benchmark repository and the [public Dynin-Omni source](https://github.com/AIDASLab/Dynin-Omni) at the version used for the three-sample check:

```bash
git clone https://github.com/AIDASLab/Omni-Storybench.git
git clone https://github.com/AIDASLab/Dynin-Omni.git
git -C Dynin-Omni checkout d6e3ca874f399a976c42bd3c8a58eb469eb7e3a3
python -m pip install -r Dynin-Omni/requirements.txt
python -m pip install -r Omni-Storybench/any-to-any/Dynin-omni/requirements.txt
```

Install a CUDA-enabled PyTorch build appropriate for your machine if your environment does not already have one. The checked local environment used PyTorch 2.7.1, torchvision 0.22.1, Transformers 4.47.1, and diffusers 0.32.2. The speech decoder requires `snu-aidas/emova_speech_tokenizer_vllm`; the runner verifies that all of its saved weights load into the active PyTorch parametrization. Access to [`snu-aidas/Dynin-Omni_nips`](https://huggingface.co/snu-aidas/Dynin-Omni_nips) is required. Authenticate with Hugging Face using your normal account setup; do not put tokens in scripts or output files.

Download [the benchmark dataset](https://huggingface.co/datasets/snu-aidas/Omni-StoryBench) with the repository helper:

```bash
python Omni-Storybench/evaluation/download_dataset.py \
  --local-dir /path/to/Omni-StoryBench-data
```

`--dynin-source` points to the public Dynin-Omni clone. The runner reads **only prompt-safe Parquet columns** and the current-page image. It checks the 900-row order and source distribution. It never opens the reference next-page text, image, speech, or per-sample JSON.

## Three-sample check

From the parent directory containing `Omni-Storybench/` and `Dynin-Omni/`:

```bash
python Omni-Storybench/any-to-any/Dynin-omni/run_storybench.py \
  --dynin-source Dynin-Omni \
  --dataset-root /path/to/Omni-StoryBench-data \
  --output-root /path/to/new/dynin-nips-joint128 \
  --device cuda:0 \
  --start 0 --until 3 \
  --steps 128 --seed 20261006
```

`--until` is exclusive. The checkpoint, image codec, and speech codec use pinned Hugging Face revisions by default. Each output index is the Parquet row number:

```text
<output-root>/
├── manifest.json
├── status.json
├── text/<index>/generated.txt
├── image/<index>/generated.png
└── speech/<index>/generated.wav
```

The three-row local check produced parseable files and recognizable scenes. Speech was transcribed with a separate local ASR model as a diagnostic; one word in sample 0 differed in ASR. These checks do not establish quantitative score parity or benchmark independence. See `manifest.json` for the exact dataset hash and model/settings identity of a run. Set `--write-token-debug` to retain generated token IDs and spans.

To use already downloaded snapshots without network requests, pass local directories for `--model`, `--image-codec`, and `--speech-codec`, plus `--local-files-only`. The speech decoder may also need its cached runtime assets; set `HF_HUB_OFFLINE=1` for a fully offline run. The sample checkpoint config may emit a `model_type omada` versus `dynin_omni` warning while loading via the public class; the runner checks the loaded checkpoint configuration before generation.

## Full batch and resuming

The full 900-row run can take hours. Launch it under a connection-independent supervisor such as `tmux`, with a new output directory and an explicit GPU:

```bash
tmux new-session -d -s dynin-storybench \
  'python Omni-Storybench/any-to-any/Dynin-omni/run_storybench.py \
    --dynin-source Dynin-Omni \
    --dataset-root /path/to/Omni-StoryBench-data \
    --output-root /path/to/new/dynin-nips-joint128 \
    --device cuda:0 --start 0 --until 900 --steps 128 \
    > /path/to/new/dynin-nips-joint128.log 2>&1'
```

Check the `tmux` session, real worker PID, log freshness, `status.json`, GPU use, and disk space after launch. If interrupted, rerun the **same** command with `--resume`. The runner checks the saved manifest and complete output triplets, skips finished indices, and preserves any incomplete published files under `incomplete/` before retrying. If a recorded complete triplet is damaged or missing, resume stops for inspection. A failed index is recorded and the process exits nonzero. A successful process ends with `status.json` state `generated_unverified`: file validity is checked, while visual and speech quality still require inspection before evaluation.

## Method and limits

The sequence follows the training prompt layout: TI2TIS task ID; source image tokens; a 1024-token padded text prompt made from current-page text, book metadata, and next-page conditions; masked target image tokens; masked target text; masked speech codes; and a speech metadata tail. The 336 px input gives 441 MAGVITv2 codes. The target uses 128 text slots and 125 speech slots, matching the training config dimensions. At each of 128 steps, **one model forward** predicts all currently masked modalities; selected tokens are committed according to confidence. This is joint inference, not three separate generation calls.

The speech waveform uses a fixed female-neutral decoder style. Generated metadata tail tokens are retained in optional token debug output but are not yet mapped to decoder style. The sampler has no classifier-free guidance. These are implementation choices requiring protocol comparison before this code or its outputs can support a paper reproduction claim.
