import tempfile
import unittest
from pathlib import Path

from src.cache import ArtifactCache


class ArtifactCacheTest(unittest.TestCase):
    def test_key_is_independent_of_mapping_order(self):
        self.assertEqual(
            ArtifactCache.key({"model": "example", "seed": 0}),
            ArtifactCache.key({"seed": 0, "model": "example"}),
        )

    def test_store_then_restore(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.txt"
            destination = root / "restored.txt"
            source.write_text("artifact", encoding="utf-8")

            cache = ArtifactCache(str(root / "cache"))
            cache.store(
                "raw",
                "abc123",
                {"artifact.txt": str(source)},
                {"stage": "raw"},
            )

            self.assertTrue(
                cache.restore(
                    "raw",
                    "abc123",
                    {"artifact.txt": str(destination)},
                )
            )
            self.assertEqual(destination.read_text(encoding="utf-8"), "artifact")


if __name__ == "__main__":
    unittest.main()
