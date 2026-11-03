"""Inventory must work in an extracted package and leave input files untouched."""

import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.inventory_project import inventory


class ProjectInventoryTests(unittest.TestCase):
    def test_inventory_excludes_its_output_and_handles_no_git_repository(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "backend/main.py"
            source.parent.mkdir()
            source.write_text('"""Fixture, never imported."""\nraise RuntimeError("must not execute")\n', encoding="utf-8")
            model = root / "models/model.pt"
            model.parent.mkdir()
            model.write_bytes(b"opaque model fixture")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            output = root / "inventory"
            output.mkdir()
            (output / "old.txt").write_text("exclude this output", encoding="utf-8")
            result = inventory(root, output)
            self.assertEqual(result["files"], 2)
            self.assertEqual(result["bytes"], source.stat().st_size + model.stat().st_size)
            self.assertFalse(result["read_errors"])
            self.assertFalse(result["source_syntax_errors"])
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)
            self.assertTrue((output / "files.csv").is_file())


if __name__ == "__main__":
    unittest.main()
