"""Authorization flags stay at the Stage 8 starting values."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT_YAML = ROOT / "configs" / "project.yaml"
PROJECT_YAML_SHA256 = "c32b6a35f07245ff194bf9904fddab1b0054b6e1268223a2df77c741e2bbbfea"


def test_project_yaml_authorization_block_is_unchanged() -> None:
    raw = PROJECT_YAML.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PROJECT_YAML_SHA256
    text = raw.decode("utf-8")
    assert "licence_status: NOT_VERIFIED" in text
    assert "human_approval: false" in text
    assert "full_model_run_authorized: false" in text
    assert "raw_data_download_authorized: false" in text
    assert "wrds_authorized: false" in text
    assert "bloomberg_authorized: false" in text
    assert "deep_learning_authorized" not in text
    assert "ffiec_download_authorized" not in text
