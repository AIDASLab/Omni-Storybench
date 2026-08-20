import tempfile
import unittest
from pathlib import Path

from src.dataset import PARQUET_NAME, _resolve_index


class DatasetPathTest(unittest.TestCase):
    def test_resolves_release_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "data").mkdir()
            (root / "images").mkdir()
            parquet = root / "data" / PARQUET_NAME
            parquet.touch()

            parquet_path, dataset_root = _resolve_index(str(root))

            self.assertEqual(parquet_path, str(parquet))
            self.assertEqual(dataset_root, str(root))

    def test_requires_images_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "data").mkdir()
            (root / "data" / PARQUET_NAME).touch()

            with self.assertRaisesRegex(FileNotFoundError, "no images/ directory"):
                _resolve_index(str(root))


if __name__ == "__main__":
    unittest.main()
