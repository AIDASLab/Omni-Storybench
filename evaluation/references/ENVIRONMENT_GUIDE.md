# Environment guide

Create the four environments listed in the [evaluation README](../README.md#environment-setup). The runner selects them automatically according to the judge group and workload.

| Workload | Default environment |
| --- | --- |
| Dataset manifest | `omnistory-judge-a` |
| Group A / stage 1 text, image, speech | `omnistory-judge-a` |
| Group A / stage 1 integrated | `omnistory-judge-a-omni` |
| BERTScore, CLIP, and stage-end JSON repair | `omnistory-judge-a` |
| Speech attributes | `omnistory-speech` |
| Groups B/C / stages 2/3, all judge tracks | `omnistory-judge-bc` |

Keep the package versions from each environment specification together. Configure checkpoint credentials, model cache locations, and the CUDA driver for your machine. The Group B/C environment uses CUDA 12.9 wheels; the Group A and speech environments use the Transformers source revision named in their specifications.

## Custom environment names

If you create the specifications with custom names, pass those names to the runner. From the repository root:

```bash
bash evaluation/run_evaluation.sh \
  --results-root "/path/to/your/baseline-outputs" \
  --dataset-root "/path/to/your/Omni-StoryBench" \
  --gpus 0,1,2,3 --stages 1,2,3 \
  --dataset-conda-env omnistory-judge-a \
  --text-conda-env-stage1 omnistory-judge-a \
  --image-conda-env-stage1 omnistory-judge-a \
  --speech-conda-env-stage1 omnistory-judge-a \
  --joint-conda-env-stage1 omnistory-judge-a-omni \
  --text-conda-env-other omnistory-judge-bc \
  --image-conda-env-other omnistory-judge-bc \
  --speech-conda-env-other omnistory-judge-bc \
  --joint-conda-env-other omnistory-judge-bc \
  --aux-conda-env omnistory-judge-a \
  --json-repair-conda-env omnistory-judge-a \
  --speech-classifier-conda-env omnistory-speech
```

Replace the example paths, device IDs, and environment names with your settings. An environment value of `current` runs the active Python interpreter. The `--text-conda-env`, `--image-conda-env`, and `--speech-conda-env` options override their respective track across all three groups.

## Check imports

Run the following from `evaluation/` after creating the environments:

```bash
PYTHONPATH="$PWD" conda run -n omnistory-judge-a python tests/import_smoke.py dataset
PYTHONPATH="$PWD" conda run -n omnistory-judge-a python tests/import_smoke.py stage1
PYTHONPATH="$PWD" conda run -n omnistory-judge-a-omni python tests/import_smoke.py omni
PYTHONPATH="$PWD" conda run -n omnistory-judge-bc python tests/import_smoke.py vllm
PYTHONPATH="$PWD" conda run -n omnistory-judge-a python tests/import_smoke.py repair
PYTHONPATH="$PWD" conda run -n omnistory-speech python tests/import_smoke.py classifier
conda run -n omnistory-judge-a python -B -m unittest discover -s tests -p test_inference_architecture.py -v
```

These commands check imports and protocol behavior without loading checkpoints. Use `--limit 1` with the main runner to check model loading and evaluation on your GPUs before a full run.
