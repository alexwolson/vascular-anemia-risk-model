from pathlib import Path

import pytest

from src import build_vqi_dataset


def test_validate_raw_excel_file_rejects_text_placeholder(tmp_path: Path) -> None:
    fake_excel = tmp_path / "VQI_Database_MTAEdits.xlsx"
    fake_excel.write_text("/some/other/path.xlsx")

    with pytest.raises(ValueError, match=r"valid \.xlsx workbook"):
        build_vqi_dataset.validate_raw_excel_file(fake_excel)


def test_validate_raw_excel_file_accepts_zip_signature(tmp_path: Path) -> None:
    fake_excel = tmp_path / "VQI_Database_MTAEdits.xlsx"
    fake_excel.write_bytes(b"PK\x03\x04dummy")

    build_vqi_dataset.validate_raw_excel_file(fake_excel)


def test_asa_analysis_uses_canonical_raw_data_path() -> None:
    source = Path("src/analysis_asa_emergent.py").read_text()
    assert 'REPO_ROOT / "data" / "raw" / "VQI_Database_MTAEdits.xlsx"' in source
