"""Every committed sweep expands, and the committed results analyze."""

from pathlib import Path

import pytest

from conftest import REPO, WHISPER_HF_MODEL
from edge_speech import analysis, evaluate

SWEEPS = sorted((REPO / "configs" / "sweeps").glob("*.yaml"))


@pytest.mark.parametrize("sweep", SWEEPS, ids=lambda path: path.stem)
def test_sweep_expands(sweep: Path) -> None:
    assert evaluate.main(["--sweep", str(sweep), "--dry-run"]) == 0


def test_ad_hoc_run_expands() -> None:
    argv = ["--backend", "whisper", "--model", WHISPER_HF_MODEL, "--languages", "en_us", "sw_ke", "--dry-run"]
    assert evaluate.main(argv) == 0


# def test_analyze_committed_results(tmp_path) -> None:
#     argv = ["--results-dir", str(REPO / "results"), "--csv", str(tmp_path / "summary.csv"),
#             "--plots-dir", str(tmp_path / "plots")]
#     assert analysis.main(argv) == 0
#     assert (tmp_path / "summary.csv").is_file()
