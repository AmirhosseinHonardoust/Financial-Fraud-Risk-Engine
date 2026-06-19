from __future__ import annotations

import json
import py_compile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ProjectIntegrityTests(unittest.TestCase):
    def test_python_sources_compile(self) -> None:
        paths = [*ROOT.joinpath("src").glob("*.py"), ROOT / "app.py"]
        self.assertTrue(paths, "No Python source files found to compile.")

        for path in paths:
            with self.subTest(path=path.relative_to(ROOT)):
                py_compile.compile(str(path), doraise=True)

    def test_required_project_files_exist(self) -> None:
        expected = [
            "data/raw/synthetic_fraud_dataset.csv",
            "requirements.txt",
            "src/config.py",
            "src/data_prep.py",
            "src/features.py",
            "src/train_model.py",
            "src/evaluate.py",
            "src/score_new_transactions.py",
            "app.py",
        ]

        for relative_path in expected:
            with self.subTest(path=relative_path):
                self.assertTrue((ROOT / relative_path).exists(), f"Missing {relative_path}")

    def test_existing_metrics_json_files_are_valid_when_present(self) -> None:
        metrics_dir = ROOT / "reports" / "metrics"
        json_files = list(metrics_dir.glob("*.json"))

        for path in json_files:
            with self.subTest(path=path.relative_to(ROOT)):
                data = json.loads(path.read_text())
                self.assertIsInstance(data, (dict, list))
