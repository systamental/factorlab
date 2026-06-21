import subprocess
import sys


def test_targets_import_in_fresh_interpreter():
    code = "import factorlab.targets as targets; import factorlab.targets.forward as forward"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_learning_search_exports_available_via_package_import():
    code = (
        "from factorlab.learning import WalkForwardGridSearch, parameter_grid, set_pipeline_params; "
        "assert WalkForwardGridSearch is not None; "
        "assert callable(parameter_grid); "
        "assert callable(set_pipeline_params)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
