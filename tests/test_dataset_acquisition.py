"""Check archive trust boundaries using local fixtures, without network access."""

from pathlib import Path
import tempfile
import unittest
import zipfile

from ml.download_recommended_datasets import extract, safe_member, transfer, verify_git_archive
from ml.download_dataset import blob_hash


class DatasetAcquisitionTests(unittest.TestCase):
    def test_rejects_archive_traversal_and_renames_windows_names(self):
        for name in ("../escape.jpg", "/absolute.jpg", "folder/../../escape.jpg", "folder\\escape.jpg"):
            with self.assertRaises(ValueError):
                safe_member(name)
        for name in ("bad:name.jpg", "CON.jpg", "trailing.", "leaf.jpg?147253"):
            safe = safe_member(name)
            self.assertTrue(safe.name.startswith("__"))
            self.assertEqual(safe, safe_member(name))
            self.assertFalse(any(c in str(safe) for c in '<>:"|?*'))
        self.assertEqual(safe_member("picture.aspx?id=22.jpg").suffix, ".jpg")
        self.assertEqual(safe_member("picture.aspx?id=22.xml").suffix, ".xml")

    def test_extraction_preserves_bytes_and_records_renamed_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "original.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("source/leaf.jpg", b"original image bytes")
                output.writestr("source/bad:name.jpg", b"second image bytes")
            report = extract(archive, root / "extracted")
            self.assertEqual(report["files"], 2)
            self.assertEqual(report["renamed_windows_paths"], 1)
            self.assertEqual((root / "extracted/source/leaf.jpg").read_bytes(), b"original image bytes")
            self.assertTrue(report["zip_crc_verified"])

    def test_existing_wrong_checksum_is_rejected_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "archive.zip"
            path.write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                transfer("https://unused.example/", path, 5, "0" * 64)
            self.assertEqual(path.read_bytes(), b"wrong")

    def test_case_distinct_files_are_preserved_on_windows(self):
        import json
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "original.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("Leaf.jpg", b"first")
                output.writestr("leaf.jpg", b"second")
            report = extract(archive, root / "original")
            mapping = json.loads((root / "extraction_map.json").read_text())
            self.assertEqual(report["renamed_windows_paths"], 1)
            self.assertEqual((root / "original/Leaf.jpg").read_bytes(), b"first")
            self.assertEqual((root / "original" / mapping["leaf.jpg"]).read_bytes(), b"second")

    def test_git_archive_verification_checks_content_and_completeness(self):
        import json
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "original.zip"
            tree = root / "tree.json"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("repo/leaf.jpg", b"original")
            expected = [{"path": "leaf.jpg", "type": "blob", "sha": blob_hash(b"original")}]
            tree.write_text(json.dumps({"tree": expected}))
            self.assertEqual(verify_git_archive(archive, tree), 1)
            expected.append({"path": "missing.jpg", "type": "blob", "sha": "0" * 40})
            tree.write_text(json.dumps({"tree": expected}))
            with self.assertRaisesRegex(ValueError, "Missing files"):
                verify_git_archive(archive, tree)


if __name__ == "__main__":
    unittest.main()
