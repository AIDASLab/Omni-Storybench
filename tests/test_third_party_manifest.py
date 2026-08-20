import json
import subprocess
import sys
import unittest
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "third_party" / "manifest.json"


class ThirdPartyManifestTest(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_pins_full_commits_for_required_backends(self):
        repositories = self.manifest["repositories"]
        self.assertEqual(set(repositories), {"emu3", "mmada"})
        for info in repositories.values():
            self.assertRegex(info["commit"], r"^[0-9a-f]{40}$")
            self.assertTrue(info["url"].startswith("https://github.com/"))
            self.assertTrue(info["required_paths"])

    def test_required_paths_are_relative_and_safe(self):
        for info in self.manifest["repositories"].values():
            for value in info["required_paths"]:
                path = PurePosixPath(value)
                self.assertFalse(path.is_absolute())
                self.assertNotIn("..", path.parts)

    def test_bootstrap_list_is_offline(self):
        result = subprocess.run(
            [sys.executable, "scripts/bootstrap_third_party.py", "--list"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertIn("emu3:", result.stdout)
        self.assertIn("mmada:", result.stdout)


if __name__ == "__main__":
    unittest.main()
