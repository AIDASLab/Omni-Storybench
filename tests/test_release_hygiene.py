import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TEXT_SUFFIXES = {".md", ".py", ".sh", ".toml", ".yaml", ".yml"}


def release_text_files():
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        if path.suffix in TEXT_SUFFIXES or path.name in {".env.example", ".gitignore"}:
            yield path


class ReleaseHygieneTest(unittest.TestCase):
    def test_default_config_exists(self):
        self.assertTrue((ROOT / "configs" / "openai_flux_voxcpm.yaml").is_file())

    def test_unified_environment_is_pinned(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('requires-python = ">=3.10,<3.11"', pyproject)
        for pin in (
            '"torch==2.10.0"',
            '"transformers==4.57.6"',
            '"vllm==0.19.1"',
            '"numpy==2.2.6"',
            '"protobuf==6.33.4"',
            '"librosa==0.11.0"',
        ):
            self.assertIn(pin, pyproject)
        self.assertTrue((ROOT / "uv.lock").is_file())

    def test_every_supported_backend_has_a_preset(self):
        names = {
            path.stem
            for path in (ROOT / "configs").glob("*.yaml")
        }
        expected_markers = (
            "openai",
            "claude",
            "vllm",
            "qwen2_5omni",
            "emova",
            "Emu3",
            "MMaDA",
        )
        missing = [
            marker
            for marker in expected_markers
            if not any(marker in name for name in names)
        ]
        self.assertEqual(missing, [])

    def test_device_only_duplicate_names_are_removed(self):
        names = {
            path.name
            for path in (ROOT / "configs").glob("*.yaml")
        }
        forbidden = {
            "emova_2b.yaml",
            "emova_flux_b.yaml",
            "emova_s1.yaml",
            "emova_s2.yaml",
            "emova_s3.yaml",
            "qwen2_5omni_flux_b.yaml",
            "qwen2_5omni_kontext_b.yaml",
            "qwen2_5omni_s1.yaml",
            "qwen2_5omni_s2.yaml",
            "qwen2_5omni_s3.yaml",
        }
        self.assertEqual(names & forbidden, set())

    def test_no_literal_api_credentials(self):
        secret_pattern = re.compile(
            r"(?:sk-(?:proj|ant)-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})"
        )
        matches = []
        for path in release_text_files():
            if secret_pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
                matches.append(str(path.relative_to(ROOT)))
        self.assertEqual(matches, [])

    def test_no_original_lab_absolute_paths(self):
        forbidden = ("/" + "home/work/", "/" + "mnt/nasdata/")
        matches = []
        for path in release_text_files():
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(prefix in text for prefix in forbidden):
                matches.append(str(path.relative_to(ROOT)))
        self.assertEqual(matches, [])


if __name__ == "__main__":
    unittest.main()
