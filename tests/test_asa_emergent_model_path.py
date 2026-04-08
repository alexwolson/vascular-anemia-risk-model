from pathlib import Path


def test_asa_analysis_has_model_path_resolver() -> None:
    source = Path("src/analysis_asa_emergent.py").read_text()
    assert "def resolve_model_path(" in source


def test_asa_analysis_uses_resolved_model_path_instead_of_constant() -> None:
    source = Path("src/analysis_asa_emergent.py").read_text()
    assert "resolve_model_path(meta)" in source
    assert "h2o.load_model(str(model_path))" in source
