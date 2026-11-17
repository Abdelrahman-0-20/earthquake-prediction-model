"""Application smoke tests.

These are the acceptance tests for Week 4: every page must *exist* and must
*render without raising*. Streamlit's ``AppTest`` executes a page in-process
and exposes any exception, so a broken import, a missing artifact, or a bad
column name fails here rather than in front of a reviewer.

Each page must bootstrap ``sys.path`` itself (shown below) so it can be run
standalone by AppTest as well as in-process by Streamlit's page router.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
for candidate in (ROOT, ROOT / "src", str(APP)):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

streamlit = pytest.importorskip("streamlit", reason="streamlit not installed")
from streamlit.testing.v1 import AppTest  # noqa: E402

TIMEOUT = 90

#: The six analytical pages, in navigation order.
PAGES = [
    ("1_Overview.py", "Overview"),
    ("2_Seismic_Map.py", "Seismic Map"),
    ("3_Historical_Analysis.py", "Historical Analysis"),
    ("4_Forecasting.py", "Forecasting"),
    ("5_Geospatial_Analysis.py", "Geospatial Analysis"),
    ("6_Model_Performance.py", "Model Performance"),
]


def _run(path: Path) -> AppTest:
    at = AppTest.from_file(str(path), default_timeout=TIMEOUT)
    at.run()
    return at


def _assert_healthy(at: AppTest, label: str) -> None:
    # AppTest exposes `exception` as an ElementList — possibly empty — not as
    # an exception instance or None, so test its length rather than identity.
    exceptions = at.exception
    if len(exceptions) == 0:
        return
    detail = "\n".join(str(e) for e in exceptions)
    raise AssertionError(f"{label} raised during render:\n{detail}")


# ---------------------------------------------------------------------------
# Page existence — the Week 4 deliverable list
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename,label", PAGES)
def test_page_exists(filename: str, label: str):
    assert (APP / "pages" / filename).is_file(), (
        f"Missing page {filename!r} ({label}) — Week 4 requires all six."
    )


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------

class TestShell:
    def test_shell_renders(self):
        at = _run(APP / "streamlit_app.py")
        _assert_healthy(at, "streamlit_app.py")
        assert at.title, "shell should render a title"

    def test_shell_surfaces_standing_limitations(self):
        """The scientific caveat must be visible on the shell itself."""
        at = _run(APP / "streamlit_app.py")
        _assert_healthy(at, "streamlit_app.py")
        sidebar_text = " ".join(str(m.value) for m in at.sidebar.markdown)
        assert "does <b>not</b> predict" in sidebar_text or "not</b> predict" in sidebar_text


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename,label", PAGES)
def test_page_renders_without_exception(filename: str, label: str):
    path = APP / "pages" / filename
    if not path.is_file():
        pytest.skip(f"{filename} not built yet")
    at = _run(path)
    _assert_healthy(at, label)


#: Element accessors vary by Streamlit version and element type — e.g.
#: `plotly_chart` is not an attribute on AppTest in 1.67, `dataframe` is
#: becoming `arrow_data_frame`, and pydeck charts register as
#: `deck_gl_json_chart` rather than `pydeck_chart`. Probe defensively so a
#: rename degrades to "not counted" instead of crashing the suite.
ELEMENT_KEYS = (
    "title", "header", "subheader", "markdown", "metric",
    "dataframe", "arrow_data_frame", "table", "json", "code", "text",
    "plotly_chart", "pydeck_chart", "deck_gl_json_chart", "chart",
    "line_chart", "area_chart",
    "error", "warning", "info", "success", "exception",
    "selectbox", "multiselect", "radio", "checkbox", "slider",
    "date_input", "expander", "tabs", "button",
)


def _count_elements(at: AppTest) -> int:
    total = 0
    for key in ELEMENT_KEYS:
        try:
            total += len(getattr(at, key))
        except Exception:  # noqa: BLE001 - absent accessor means zero elements
            continue
    return total


@pytest.mark.parametrize("filename,label", PAGES)
def test_page_produces_visible_content(filename: str, label: str):
    """A page that renders nothing is a failure, not a stub."""
    path = APP / "pages" / filename
    if not path.is_file():
        pytest.skip(f"{filename} not built yet")

    at = _run(path)
    _assert_healthy(at, label)

    total = _count_elements(at)
    assert total > 0, (
        f"{label} rendered no visible content "
        f"(probed {len(ELEMENT_KEYS)} element types)."
    )


# ---------------------------------------------------------------------------
# Data-access contract
# ---------------------------------------------------------------------------

class DataAccessContract:
    """Pages must read through data_access, never touching data files directly.

    Enforced structurally rather than by banning ``Path(``, which would also
    ban the ``Path(__file__)`` bootstrap every page legitimately needs to run
    standalone. The real rule: no page may open a data file itself, nor name
    an artifact location — that keeps the six pages consistent and makes the
    app's data dependencies enumerable in exactly one file.
    """

    #: Direct data-file access.
    FORBIDDEN = ("read_parquet", "read_json", "open(")

    #: Artifact locations must be reached via data_access, never spelled out.
    FORBIDDEN_PATHS = (
        "data/raw", "data/processed", "data/features", "data/export",
        "models/", "reports/model_metrics",
    )


@pytest.mark.parametrize("filename,label", PAGES)
def test_page_reads_through_data_access(filename: str, label: str):
    path = APP / "pages" / filename
    if not path.is_file():
        pytest.skip(f"{filename} not built yet")

    source = path.read_text(encoding="utf-8")
    for token in DataAccessContract.FORBIDDEN:
        assert token not in source, (
            f"{label} must not call {token!r} directly — import a loader "
            f"from data_access instead (single filesystem layer rule)."
        )
    for token in DataAccessContract.FORBIDDEN_PATHS:
        assert token not in source, (
            f"{label} names the artifact location {token!r} — reach data "
            f"through data_access so the filesystem layer stays in one file."
        )


@pytest.mark.parametrize("filename,label", PAGES)
def test_page_bootstraps_sys_path(filename: str, label: str):
    """Pages run standalone under AppTest, so they must set up imports."""
    path = APP / "pages" / filename
    if not path.is_file():
        pytest.skip(f"{filename} not built yet")

    source = path.read_text(encoding="utf-8")
    assert "sys.path" in source, (
        f"{label} must bootstrap sys.path so it can run standalone "
        f"(AppTest and `streamlit run app/pages/...`)."
    )
