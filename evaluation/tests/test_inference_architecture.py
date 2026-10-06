from __future__ import annotations

import argparse
import json
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.evaluate_joint import resolve_adapter as resolve_joint_adapter
from scripts.evaluate_modality import resolve_adapter as resolve_modality_adapter
from scripts.repair_stage_results import assess_modality_quality
from src.common import JUDGE_CATEGORIES_BY_MODALITY
from src.dataset import (
    DEFAULT_DATASET_ROOT,
    canonical_record_from_parquet_row,
)
from src.evaluation_protocol import build_joint_messages
from src.inference import run_batch_inference
from src.inference.vllm_omni import _build_sampling_kwargs
from src.judge_parsing import (
    build_judge_json_schema,
    build_repair_json_schema,
    parse_judge_response,
)
from src.model_adapters import STAGE_CONFIGS, get_adapter, get_stage_adapter
from src.model_adapters.gemma4 import adapt_joint_messages
from src.model_adapters.registry import ALL_ADAPTERS


HARNESS_ROOT = Path(__file__).resolve().parents[1]

class DatasetAdapterTests(unittest.TestCase):
    @staticmethod
    def _row() -> dict[str, object]:
        return {
            "id": "source/book/page_001__page_002",
            "source": "source",
            "book": "book",
            "from_key": "page_001",
            "to_key": "page_002",
            "current_image": "images/source/book/page_001.png",
            "current_text": "Current narration.",
            "current_text_path": "texts/source/book/page_001.txt",
            "next_image": "images/source/book/page_002.png",
            "next_text": "Next narration.",
            "next_text_path": "texts/source/book/page_002.txt",
            "genre": "Picture Book",
            "topic": "Friendship",
            "style": "Whimsical",
            "narrative_tense": "Present",
            "narrative_perspective": "Third-person",
            "scene": "A garden at sunrise.",
            "text_instruction": "Continue the garden scene.",
            "ambient_sound": "Birdsong",
            "speech_utterance": "We found it!",
            "speech_speaker": "Mina",
            "speech_emotion": "happy",
            "speech_speed": "normal",
            "speech_pitch": "high",
            "speech_gender": "female",
            "condition_characters": json.dumps(
                [
                    {
                        "name": "Mina",
                        "emotion": "happy",
                        "visibility": "visible",
                        "speech_intent": "celebrates",
                        "action": "points at the flower",
                    }
                ]
            ),
            "book_characters": json.dumps(
                [
                    {
                        "name": "Mina",
                        "sex": "female",
                        "age_range": "child",
                        "species": "human",
                        "role": "protagonist",
                        "personality": ["curious"],
                        "appearance": "red coat",
                    }
                ]
            ),
            "sample_path": "samples/source/book/page_001__page_002.json",
            "metadata_path": "metadata/source/book.json",
        }

    def test_default_dataset_is_the_downloaded_hf_snapshot(self) -> None:
        self.assertEqual(
            DEFAULT_DATASET_ROOT.resolve(),
            (HARNESS_ROOT / "downloaded_dataset").resolve(),
        )

    def test_hf_row_maps_to_the_existing_prompt_contract(self) -> None:
        record = canonical_record_from_parquet_row(
            self._row(),
            HARNESS_ROOT / "downloaded_dataset",
            1,
            validate_files=False,
            validate_sample_json=False,
        )

        self.assertEqual(record["prev_key"], "page_001")
        self.assertEqual(
            record["metadata"]["metadata"]["characters"][0]["name"],
            "Mina",
        )
        self.assertEqual(
            record["next_page_condition"]["condition_json"]["text content"],
            "Continue the garden scene.",
        )
        self.assertEqual(
            record["speech_metadata_next_page"]["parsed"]["speaker"]["line"],
            "We found it!",
        )
        self.assertEqual(
            record["current_page"]["image_path"],
            str(
                (
                    HARNESS_ROOT
                    / "downloaded_dataset/images/source/book/page_001.png"
                ).resolve()
            ),
        )

    def test_hf_row_id_must_match_its_transition(self) -> None:
        row = self._row()
        row["id"] = "source/book/page_001__page_999"
        with self.assertRaisesRegex(ValueError, "id mismatch"):
            canonical_record_from_parquet_row(
                row,
                HARNESS_ROOT / "downloaded_dataset",
                1,
                validate_files=False,
                validate_sample_json=False,
            )



class AdapterRegistryTests(unittest.TestCase):
    def test_adapter_keys_and_models_are_unique(self) -> None:
        keys = [adapter.key for adapter in ALL_ADAPTERS]
        models = [adapter.model_name.casefold() for adapter in ALL_ADAPTERS]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len(models), len(set(models)))

    def test_stage_assignments_are_complete_and_exact(self) -> None:
        expected = {
            "1": {
                "text": "stage1_qwen3_text",
                "image": "stage1_qwen3_image",
                "speech": "stage1_audio_flamingo_next",
                "joint": "stage1_qwen3_omni",
            },
            "2": {
                "text": "stage2_seed_oss_text",
                "image": "stage2_internvl35_image",
                "speech": "stage2_moss_audio",
                "joint": "stage2_nemotron3_nano_omni",
            },
            "3": {
                "text": "stage3_llama_nemotron_text",
                "image": "stage3_exaone45_image",
                "speech": "stage3_kimi_audio",
                "joint": "stage3_gemma4_omni",
            },
        }
        self.assertEqual(set(STAGE_CONFIGS), set(expected))
        for stage, modalities in expected.items():
            for modality, adapter_key in modalities.items():
                with self.subTest(stage=stage, modality=modality):
                    adapter = get_stage_adapter(stage, modality)
                    self.assertEqual(adapter.key, adapter_key)
                    self.assertEqual(adapter.stage, stage)
                    self.assertEqual(adapter.modality, modality)

    def test_modality_resolver_uses_stage_default(self) -> None:
        args = argparse.Namespace(adapter=None, stage="2", modality="text")
        self.assertEqual(
            resolve_modality_adapter(args).key,
            "stage2_seed_oss_text",
        )

    def test_joint_resolver_uses_stage_default(self) -> None:
        args = argparse.Namespace(adapter=None, stage="3")
        self.assertEqual(
            resolve_joint_adapter(args).key,
            "stage3_gemma4_omni",
        )

    def test_adapter_override_must_match_job(self) -> None:
        args = argparse.Namespace(
            adapter="stage3_kimi_audio",
            stage="2",
            modality="speech",
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            resolve_modality_adapter(args)

    def test_legacy_consistency_options_remain_supported(self) -> None:
        adapter = get_adapter("stage2_seed_oss_text")
        args = argparse.Namespace(
            adapter=adapter.key,
            stage="2",
            modality="text",
            model_name=adapter.model_name,
            backend="auto",
        )
        self.assertIs(resolve_modality_adapter(args), adapter)

        args.model_name = "wrong/model"
        with self.assertRaisesRegex(ValueError, "model_name"):
            resolve_modality_adapter(args)

    def test_stage1_uses_existing_capability_families(self) -> None:
        expected = {
            "stage1_qwen3_text": "vllm_text",
            "stage1_qwen3_image": "vllm_image",
            "stage1_audio_flamingo_next": "audio_flamingo_transformers",
            "stage1_qwen3_omni": "qwen3_omni_vllm",
        }
        self.assertEqual(
            {
                key: get_adapter(key).inference_family
                for key in expected
            },
            expected,
        )

    def test_reasoning_and_model_controls_are_preserved(self) -> None:
        self.assertEqual(
            get_adapter("stage2_seed_oss_text").chat_template_kwargs,
            {"thinking_budget": 0},
        )
        for key in (
            "stage2_nemotron3_nano_omni",
            "stage3_exaone45_image",
            "stage3_gemma4_omni",
        ):
            self.assertEqual(
                get_adapter(key).chat_template_kwargs,
                {"enable_thinking": False},
            )

        nemotron = get_adapter("stage3_llama_nemotron_text")
        self.assertEqual(
            nemotron.control_metadata()["reasoning"]["system_prompt_directive"],
            "/no_think",
        )
        gemma = get_adapter("stage3_gemma4_omni")
        self.assertTrue(gemma.engine_kwargs["skip_mm_profiling"])
        self.assertEqual(
            gemma.engine_kwargs["mm_processor_kwargs"],
            {"max_soft_tokens": 1120},
        )
        self.assertEqual(gemma.omni_prompt_adapter, "gemma4")
        self.assertTrue(gemma.enforce_json_schema)
        self.assertEqual(
            gemma.retry_sampling_overrides,
            {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
        )
        self.assertEqual(
            gemma.structured_output_kwargs,
            {
                "disable_any_whitespace": True,
                "disable_additional_properties": True,
            },
        )
        self.assertEqual(
            gemma.repetition_detection_kwargs,
            {
                "max_pattern_size": 20,
                "min_pattern_size": 3,
                "min_count": 4,
            },
        )
        self.assertEqual(gemma.max_generation_attempts, 3)
        self.assertEqual(
            gemma.control_metadata()["omni_prompt_adapter"],
            "gemma4",
        )
        self.assertTrue(gemma.control_metadata()["enforce_json_schema"])
        self.assertFalse(
            get_adapter("stage2_nemotron3_nano_omni").enforce_json_schema
        )
        kimi = get_adapter("stage3_kimi_audio")
        self.assertEqual(kimi.audio_prompt_adapter, "kimi_audio")
        self.assertEqual(kimi.stop_token_ids, (151644,))

    def test_gemma_retry_sampling_only_changes_retry_attempts(self) -> None:
        adapter = get_adapter("stage3_gemma4_omni")
        common = {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": 2048,
            "repetition_penalty": 1.0,
            "base_seed": 1234,
        }

        first, first_seed, first_scope = _build_sampling_kwargs(
            adapter,
            attempt_index=0,
            **common,
        )
        self.assertEqual(
            first,
            {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 2048,
                "repetition_penalty": 1.0,
            },
        )
        self.assertEqual((first_seed, first_scope), (1234, "engine"))

        retry, retry_seed, retry_scope = _build_sampling_kwargs(
            adapter,
            attempt_index=1,
            **common,
        )
        self.assertEqual(retry["temperature"], 1.0)
        self.assertEqual(retry["top_p"], 0.95)
        self.assertEqual(retry["top_k"], 64)
        self.assertEqual(retry["seed"], 1235)
        self.assertEqual((retry_seed, retry_scope), (1235, "retry_request"))

    def test_gemma_joint_prompt_uses_documented_modality_order(self) -> None:
        messages = build_joint_messages(
            metadata="{}",
            condition_json="{}",
            current_text="current text",
            ground_truth_text="ground truth text",
            candidate_text="candidate text",
            ground_truth_speech_metadata="{}",
            current_image="current.png",
            ground_truth_image="ground-truth.png",
            candidate_image="candidate.png",
            candidate_audio="candidate.wav",
        )

        adapted = adapt_joint_messages(messages)
        content = adapted[1]["content"]
        content_types = [item["type"] for item in content]

        self.assertEqual(content_types[:3], ["image", "image", "image"])
        self.assertEqual(content_types[-1], "audio")
        self.assertTrue(
            all(item_type == "text" for item_type in content_types[3:-1])
        )
        rendered_text = "\n".join(
            item["text"] for item in content if item["type"] == "text"
        )
        self.assertIn("Image 1 above is the current scene image.", rendered_text)
        self.assertIn(
            "Image 3 above is the generated candidate next-scene image.",
            rendered_text,
        )
        self.assertLess(
            rendered_text.index("Score all required criteria."),
            rendered_text.index("<generated_candidate_speech_audio>"),
        )
        self.assertEqual(messages[1]["content"][0]["type"], "text")


class DispatchTests(unittest.TestCase):
    @patch(
        "src.inference.vllm_text.evaluate_text_batch",
        return_value=[{"status": "ok"}],
    )
    def test_stage1_text_dispatches_to_shared_executor(self, executor) -> None:
        adapter = get_adapter("stage1_qwen3_text")
        requests = [{"candidate_text": "candidate"}]
        result = run_batch_inference(
            adapter,
            requests,
            vllm_kwargs={"seed": 0},
            max_new_tokens=1024,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
        )
        self.assertEqual(result, [{"status": "ok"}])
        executor.assert_called_once()
        self.assertEqual(executor.call_args.args, (adapter.model_name, requests))


class HarnessDefaultsTests(unittest.TestCase):
    def test_experiment_defaults_are_unchanged(self) -> None:
        shell = (HARNESS_ROOT / "run_evaluation.sh").read_text(
            encoding="utf-8"
        )
        expected = {
            "TEXT_VLLM_MAX_MODEL_LEN": "8192",
            "IMAGE_VLLM_MAX_MODEL_LEN": "20000",
            "SPEECH_VLLM_MAX_MODEL_LEN": "32768",
            "KIMI_VLLM_MAX_MODEL_LEN": "8192",
            "JOINT_VLLM_MAX_MODEL_LEN": "32768",
            "VLLM_TENSOR_PARALLEL_SIZE": "1",
            "TEXT_BATCH_SIZE": "16",
            "IMAGE_BATCH_SIZE": "4",
            "SPEECH_BATCH_SIZE": "1",
            "JOINT_BATCH_SIZE": "4",
            "TEXT_SEED": "0",
            "IMAGE_SEED": "0",
            "SPEECH_SEED": "0",
            "JOINT_SEED": "1234",
            "EVAL_MAX_NEW_TOKENS": "1024",
            "EVAL_JOINT_MAX_NEW_TOKENS": "2048",
            "EVAL_TEMPERATURE": "0.0",
            "EVAL_TOP_P": "1.0",
            "EVAL_REPETITION_PENALTY": "1.0",
        }
        for name, value in expected.items():
            with self.subTest(name=name):
                self.assertRegex(
                    shell,
                    rf"(?m)^{re.escape(name)}={re.escape(value)}$",
                )

    def test_runner_uses_only_the_downloaded_hf_dataset(self) -> None:
        shell = (HARNESS_ROOT / "run_evaluation.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'DATASET_ROOT="${OMNISTORYBENCH_DATASET_ROOT:-$HARNESS_DIR/downloaded_dataset}"',
            shell,
        )
        self.assertIn("--dataset-manifest-path", shell)
        self.assertNotIn("dataset_speech_processed.jsonl", shell)
        self.assertNotIn("dataset_v6_final_with_codex.jsonl", shell)
        self.assertNotIn("--dataset-jsonl", shell)

    def test_legacy_stage1_wrappers_were_removed(self) -> None:
        for filename in (
            "inference_qwen3_eval_text.py",
            "inference_qwen3_eval_image.py",
            "inference_qwen3_eval_speech.py",
            "inference_qwen3_omni_eval.py",
        ):
            with self.subTest(filename=filename):
                self.assertFalse((HARNESS_ROOT / "src" / filename).exists())


class ParsingTests(unittest.TestCase):
    def test_missing_final_object_closer_is_recovered(self) -> None:
        categories = JUDGE_CATEGORIES_BY_MODALITY["text"]
        payload = {
            category: {"score": 7, "rationale": f"reason for {category}"}
            for category in categories
        }
        raw_response = json.dumps(payload)[:-1]
        result = parse_judge_response(raw_response, categories)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["parse_status"], "normalized")
        self.assertEqual(set(result["judge_eval"]), set(categories))

    def test_unterminated_string_is_not_closed(self) -> None:
        categories = JUDGE_CATEGORIES_BY_MODALITY["text"]
        result = parse_judge_response(
            '{"alignment_with_metadata": {"score": 7, "rationale": "cut off',
            categories,
        )
        self.assertEqual(result["status"], "parse_error")

    def test_unescaped_quotes_inside_rationale_are_recovered(self) -> None:
        categories = JUDGE_CATEGORIES_BY_MODALITY["text"]
        payload = {
            category: {"score": 10, "rationale": f"reason for {category}"}
            for category in categories
        }
        rationale = (
            'Smiles (implied by "eyes sparkled" and "declared happily"), '
            'then says "best kite", clearly.'
        )
        payload["satisfaction_of_generation_conditions"]["rationale"] = rationale
        raw_response = json.dumps(payload).replace('\\\"', '"')

        result = parse_judge_response(raw_response, categories)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["parse_status"], "normalized")
        self.assertEqual(
            result["judge_eval"]["satisfaction_of_generation_conditions"]["rationale"],
            rationale,
        )

    def test_unresolved_parse_error_is_nonfatal_by_default(self) -> None:
        records = [
            {"inference_status": "ok", "status": "ok"},
            {"inference_status": "ok", "status": "parse_error"},
        ]

        quality = assess_modality_quality(records, 0.5)

        self.assertEqual(quality["unresolved_parse_error_count"], 1)
        self.assertTrue(quality["systemic_rate_gate_passed"])
        self.assertTrue(quality["quality_gate_passed"])

    def test_zero_unresolved_policy_remains_explicit_opt_in(self) -> None:
        records = [{"inference_status": "ok", "status": "parse_error"}]

        quality = assess_modality_quality(
            records,
            0.5,
            require_zero_unresolved=True,
        )

        self.assertFalse(quality["zero_unresolved_gate_passed"])
        self.assertFalse(quality["quality_gate_passed"])

    def test_judge_schema_requires_every_category(self) -> None:
        categories = JUDGE_CATEGORIES_BY_MODALITY["joint"]
        schema = build_judge_json_schema(categories)

        self.assertEqual(schema["required"], categories)
        self.assertFalse(schema["additionalProperties"])
        for category in categories:
            self.assertEqual(
                schema["properties"][category]["required"],
                ["score", "rationale"],
            )

    def test_repair_schema_accepts_result_or_unrecoverable_sentinel(self) -> None:
        categories = JUDGE_CATEGORIES_BY_MODALITY["speech"]
        schema = build_repair_json_schema(categories)
        self.assertEqual(len(schema["oneOf"]), 2)
        repaired, unrecoverable = schema["oneOf"]
        self.assertEqual(repaired["required"], categories)
        self.assertEqual(
            unrecoverable["properties"]["unrecoverable"]["const"],
            True,
        )


if __name__ == "__main__":
    unittest.main()
