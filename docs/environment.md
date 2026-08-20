# Unified environment

All seven backends use one Python environment. The tested compatibility point
is Python 3.10, PyTorch 2.10.0, and Transformers 4.57.6; exact transitive
versions are recorded in `uv.lock`.

This combination was selected after checking both declared constraints and
runtime imports. The important boundary is Transformers: vLLM 0.19.1 requires
Transformers 4.56 or newer, while the older research packages declare 4.44,
4.46, or 4.47 pins. Transformers 4.57.6 is accepted by vLLM and retains the
interfaces used by Qwen2.5-Omni, EMOVA, Emu3, MMaDA, and the local Parler-TTS
compatibility adapter. Transformers 5.5.4 is not suitable: the tested Emu3,
EMOVA, and Parler-TTS imports use APIs removed in Transformers 5.

## Compatibility overrides

`pyproject.toml` deliberately overrides four legacy transitive pins:

| Package | Unified version | Reason |
| --- | --- | --- |
| `transformers` | 4.57.6 | Satisfies vLLM while preserving the research backends' 4.x APIs. |
| `numpy` | 2.2.6 | Satisfies the vLLM/Numba stack; the original EMOVA metadata pins 1.23.5. |
| `protobuf` | 6.33.4 | Required by vLLM; the original EMOVA metadata pins 3.20. |
| `librosa` | 0.11.0 | Works with NumPy 2; the original EMOVA metadata pins 0.8.0. |

The EMOVA speech-tokenizer and Parler-TTS packages are installed from their
official repositories at immutable commits. Emu3 and MMaDA are source
checkouts managed by `third_party/manifest.json` because their adapters import
repository modules directly.

## Validation scope

The unified candidate passed dependency resolution and import-level checks for
the runner, vLLM, Qwen2.5-Omni, EMOVA model and processor remote code, the
EMOVA speech tokenizer, Emu3, MMaDA, Parler-TTS, VoxCPM, and Diffusers. The
checks used no model weights and do not establish numerical parity or
end-to-end GPU generation. Run at least one sample for every backend on the
target CUDA system before publishing results.

`flash-attn` is built for the target machine during installation. It requires
a CUDA toolkit compatible with the installed PyTorch build. The audio stack
also expects common system tools such as FFmpeg, SoX, and an eSpeak-compatible
phonemizer backend.
