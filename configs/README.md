# Configuration presets

These files preserve one canonical preset for every distinct research pipeline.
A separate file is retained when it changes at least one of:

- backbone or hosted model;
- text-to-image model;
- text-to-speech model.

Legacy aliases, superseded tuning copies, and files that differed only by
CUDA/server placement were removed. Renamed presets use explicit model-stack
names, for example `emova_nitro`, `qwen2_5omni_nitro`, and
`vllm_qwen35_fluxkontext_voxcpm`. The explicitly named preset contains the
canonical generation settings for its pipeline.

The remaining `orch_device`, `t2i_device`, `t2s_device`, and local vLLM URL
values record a known research placement. Users may need to change them for
their hardware; such a change does not require another checked-in preset.
