import torch

# TODO: Add TI2I models, e.g., Qwen-Image-Edit


class T2IGenerator:
    SEED = 0

    KONTEXT_PREFIX = "Keep the art style, characters, and color palette. "
    KONTEXT_CHECKPOINT = "black-forest-labs/FLUX.1-Kontext-dev"
    KONTEXT_GUIDANCE_SCALE = 2.5

    NITRO_TEXT_ENCODER = "meta-llama/Llama-3.2-1B"
    NITRO_RESOLUTION = 1024
    NITRO_GUIDANCE_SCALE = 4.0
    NITRO_NUM_INFERENCE_STEPS = 20

    FLUX_CHECKPOINT = "black-forest-labs/FLUX.1-dev"
    FLUX_RESOLUTION = 1024
    FLUX_GUIDANCE_SCALE = 3.5
    FLUX_NUM_INFERENCE_STEPS = 50
    FLUX_MAX_SEQUENCE_LENGTH = 512

    @staticmethod
    def _mode_for(model_name: str) -> str:
        name = model_name.lower()

        if "nitro-t-1.2b" in name:
            return "nitro"
        if "kontext" in name:
            return "kontext"
        if "flux" in name:
            return "flux"
        raise ValueError(f"Unknown T2I model: {model_name}")

    @classmethod
    def signature(cls, model_name: str) -> dict:
        """Everything that determines this generator's output, for cache keying.

        Resolvable without loading the pipeline, so a fully cached run never
        constructs the model.
        """
        mode = cls._mode_for(model_name)
        common = {"model": model_name, "mode": mode, "seed": cls.SEED}

        if mode == "kontext":
            return {
                **common,
                "checkpoint": cls.KONTEXT_CHECKPOINT,
                "prompt_prefix": cls.KONTEXT_PREFIX,
                "guidance_scale": cls.KONTEXT_GUIDANCE_SCALE,
            }
        if mode == "nitro":
            return {
                **common,
                "text_encoder": cls.NITRO_TEXT_ENCODER,
                "resolution": cls.NITRO_RESOLUTION,
                "guidance_scale": cls.NITRO_GUIDANCE_SCALE,
                "num_inference_steps": cls.NITRO_NUM_INFERENCE_STEPS,
            }
        return {
            **common,
            "checkpoint": cls.FLUX_CHECKPOINT,
            "resolution": cls.FLUX_RESOLUTION,
            "guidance_scale": cls.FLUX_GUIDANCE_SCALE,
            "num_inference_steps": cls.FLUX_NUM_INFERENCE_STEPS,
            "max_sequence_length": cls.FLUX_MAX_SEQUENCE_LENGTH,
        }

    def __init__(self, model_name: str, device: str = "cuda"):
        name = model_name.lower()

        if "nitro-t-1.2b" in name:
            from diffusers import DiffusionPipeline
            from transformers import AutoModelForCausalLM

            try:
                text_encoder = AutoModelForCausalLM.from_pretrained(
                    self.NITRO_TEXT_ENCODER,
                    torch_dtype=torch.bfloat16,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load {self.NITRO_TEXT_ENCODER} for {model_name}. "
                    "Nitro-T-1.2B depends on that external text encoder."
                ) from exc

            self.pipe = DiffusionPipeline.from_pretrained(
                model_name,
                text_encoder=text_encoder,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            self.pipe.to(device)
            self.mode = "nitro"
        elif "kontext" in name:
            # `FluxKontextPipeline` is only available in newer diffusers releases.
            # Older installs can still load the Kontext checkpoint via the FLUX
            # img2img pipeline, which supports the same `image=` + `prompt=` call shape.
            try:
                from diffusers import FluxKontextPipeline as _KontextPipeline
            except ImportError:
                from diffusers import FluxImg2ImgPipeline as _KontextPipeline

            self.pipe = _KontextPipeline.from_pretrained(
                self.KONTEXT_CHECKPOINT, torch_dtype=torch.bfloat16
            )
            self.pipe.to(device)
            self.mode = "kontext"
        elif "flux" in name:
            from diffusers import FluxPipeline
            self.pipe = FluxPipeline.from_pretrained(
                self.FLUX_CHECKPOINT, torch_dtype=torch.bfloat16
            )
            self.pipe.to(device)
            self.mode = "flux"
        else:
            raise ValueError(f"Unknown T2I model: {model_name}")

    def forward(self, prompt: str, output_path: str, input_image: str | None = None):
        """Generate an image from `prompt` and save it to `output_path`.

        For Kontext (image-editing) models, `input_image` is the reference image
        used to anchor the style of the generated image.
        """
        if self.mode == "kontext":
            from diffusers.utils import load_image
            ref_image = load_image(input_image)
            image = self.pipe(
                image=ref_image,
                prompt=self.KONTEXT_PREFIX + prompt,
                guidance_scale=self.KONTEXT_GUIDANCE_SCALE,
                generator=torch.Generator("cpu").manual_seed(self.SEED),
            ).images[0]
        elif self.mode == "nitro":
            image = self.pipe(
                prompt=prompt,
                height=self.NITRO_RESOLUTION,
                width=self.NITRO_RESOLUTION,
                num_inference_steps=self.NITRO_NUM_INFERENCE_STEPS,
                guidance_scale=self.NITRO_GUIDANCE_SCALE,
                generator=torch.Generator("cpu").manual_seed(self.SEED),
            ).images[0]
        else:
            image = self.pipe(
                prompt,
                height=self.FLUX_RESOLUTION,
                width=self.FLUX_RESOLUTION,
                guidance_scale=self.FLUX_GUIDANCE_SCALE,
                num_inference_steps=self.FLUX_NUM_INFERENCE_STEPS,
                max_sequence_length=self.FLUX_MAX_SEQUENCE_LENGTH,
                generator=torch.Generator("cpu").manual_seed(self.SEED),
            ).images[0]
        image.save(output_path)
