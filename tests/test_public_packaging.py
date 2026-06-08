from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class PublicPackagingTests(unittest.TestCase):
    def test_public_data_policy_docs_exist(self) -> None:
        license_text = (REPO_ROOT / "DATA_LICENSE.md").read_text(encoding="utf-8")
        readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("does not redistribute vendor market data", license_text)
        self.assertIn("scripts/recreate_dataset.ps1", license_text)
        self.assertIn("data/fixtures/", readme_text)

    def test_synthetic_fixture_is_present(self) -> None:
        fixture_root = REPO_ROOT / "data" / "fixtures" / "synthetic_equities"
        expected = [
            fixture_root / "raw" / "daily_market_bars.csv",
            fixture_root / "raw" / "eodhd_equity_metadata.csv",
            fixture_root / "processed" / "daily_features.csv",
            fixture_root / "processed" / "market_context_features.csv",
        ]

        for path in expected:
            self.assertTrue(path.exists(), f"missing fixture file: {path}")

    def test_vendor_data_and_artifacts_are_not_tracked(self) -> None:
        result = subprocess.run(
            ["git", "ls-files", "data", "artifacts"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        tracked = [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]
        disallowed_prefixes = (
            "artifacts/",
            "data/eodhd_",
            "data/massive_",
            "data/eodhd_training_panels/",
        )

        bad = [path for path in tracked if path.startswith(disallowed_prefixes)]
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
