"""Every metric must identify its results by its registry name.

The GUI catalog, the metric palette and every per-metric filter key on the
class name, so a metric that writes a different string into DQmetric produces
results that can never be traced back to it.
"""
import ast
import pathlib

import pytest

_METRIC_ROOT = pathlib.Path(__file__).parent.parent / "metis" / "metric"


def _metric_modules():
    return sorted(
        p for p in _METRIC_ROOT.glob("*/*.py")
        if not p.name.endswith("_config.py")
        and not p.name.startswith("_")
        and p.name != "__init__.py"
    )


@pytest.mark.parametrize("path", _metric_modules(), ids=lambda p: p.stem)
def test_dqmetric_is_never_a_hardcoded_string(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name != "DQResult":
            continue
        for kw in node.keywords:
            if kw.arg == "DQmetric" and isinstance(kw.value, ast.Constant):
                offenders.append((node.lineno, kw.value.value))
    assert not offenders, (
        f"{path.name} hardcodes DQmetric: {offenders}. "
        f"Use self.__class__.__name__ so the GUI can resolve the metric."
    )


@pytest.mark.parametrize("path", _metric_modules(), ids=lambda p: p.stem)
def test_dqresult_never_uses_the_removed_mesTime_keyword(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name != "DQResult":
            continue
        for kw in node.keywords:
            if kw.arg == "mesTime":
                offenders.append(node.lineno)
    assert not offenders, (
        f"{path.name} passes mesTime= at lines {offenders}. "
        f"DQResult.__init__ takes `timestamp`."
    )
