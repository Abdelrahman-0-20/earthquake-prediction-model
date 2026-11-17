"""CLI contract tests: help, exit codes, JSON output, error messages."""

from __future__ import annotations

import json

import pytest

from seismo.cli import (
    EXIT_API,
    EXIT_MISSING,
    EXIT_OK,
    EXIT_USAGE,
    EXIT_VALIDATION,
    main,
)

# ---------------------------------------------------------------------------
# Discoverability — --help at every level
# ---------------------------------------------------------------------------

class TestHelp:
    def test_root_help_exits_zero(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == EXIT_OK
        text = capsys.readouterr().out
        assert "seismo" in text
        assert "Examples:" in text
        assert "Exit codes:" in text

    def test_no_args_prints_help_not_error(self, capsys):
        assert main([]) == EXIT_OK
        assert "Examples:" in capsys.readouterr().out

    def test_data_help(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["data", "--help"])
        assert exc.value.code == EXIT_OK
        out = capsys.readouterr().out
        assert "fetch" in out and "validate" in out and "status" in out

    def test_fetch_help_lists_flags(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["data", "fetch", "--help"])
        assert exc.value.code == EXIT_OK
        out = capsys.readouterr().out
        for flag in ("--start", "--end", "--min-mag", "--limit", "--force"):
            assert flag in out, flag

    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == EXIT_OK
        assert "seismo" in capsys.readouterr().out

    def test_data_without_subcommand_prints_help(self, capsys):
        code = main(["data"])
        assert code == EXIT_USAGE
        assert "fetch" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Error messages must name the fix
# ---------------------------------------------------------------------------

class TestErrors:
    def test_bad_date_message_names_format(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["data", "fetch", "--start", "01-01-2024"])
        assert exc.value.code == EXIT_USAGE
        err = capsys.readouterr().err
        assert "YYYY-MM-DD" in err

    def test_unknown_command_exits_usage(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["frobnicate"])
        assert exc.value.code == EXIT_USAGE

    def test_missing_config_points_at_fix(self, tmp_path, capsys):
        # Empty directory: no config/config.yaml anywhere above.
        import os

        old = os.getcwd()
        os.chdir(tmp_path)
        try:
            code = main(["data", "status"])
        finally:
            os.chdir(old)
        assert code == EXIT_USAGE
        assert "config/config.yaml" in capsys.readouterr().err

    def test_missing_raw_data_tells_user_to_fetch(self, project_root, capsys):
        code = main(["--root", str(project_root), "data", "validate"])
        assert code == EXIT_MISSING
        assert "seismo data fetch" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Exit-code contract
# ---------------------------------------------------------------------------

class TestExitCodes:
    def test_constants_match_documented_contract(self):
        assert (EXIT_OK, EXIT_USAGE, EXIT_VALIDATION, EXIT_API, EXIT_MISSING) == (
            0, 2, 3, 4, 5,
        )

    def test_status_on_empty_dir_is_ok(self, project_root, capsys):
        code = main(["--root", str(project_root), "data", "status"])
        assert code == EXIT_OK
        assert "No fetch manifest" in capsys.readouterr().out

    def test_validate_after_fetch_is_ok(self, project_root, capsys, monkeypatch):
        """fetch → validate happy path returns 0 and produces artifacts."""
        from conftest import FakeResponse, FakeSession, csv_body

        from seismo.data import ingestion

        rows = [{
            "time": "2024-01-02T00:00:00.000Z", "latitude": 35.0, "longitude": 31.0,
            "depth": 10.0, "mag": 4.1, "id": "ev1", "place": "Cairo, Egypt",
            "type": "earthquake", "updated": "2024-01-03T00:00:00.000Z",
        }]

        fake = FakeSession([FakeResponse(200, csv_body(rows))])
        monkeypatch.setattr(ingestion, "requests", type("R", (), {
            "Session": staticmethod(lambda: fake),
        }))
        assert main(["--root", str(project_root), "data", "fetch", "--limit", "1"]) == EXIT_OK
        assert main(["--root", str(project_root), "data", "validate"]) == EXIT_OK

        assert (project_root / "data/processed/earthquakes.parquet").is_file()
        assert (project_root / "reports/data_quality/data_quality.json").is_file()


# ---------------------------------------------------------------------------
# Scriptability — --json produces parseable output
# ---------------------------------------------------------------------------

class TestJsonOutput:
    def test_status_json_is_parseable(self, project_root, capsys):
        code = main(["--root", str(project_root), "--json", "data", "status"])
        assert code == EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert payload["months_recorded"] == 0
        assert "status_counts" in payload

    def test_validate_json_is_parseable(self, project_root, capsys, monkeypatch):
        from conftest import FakeResponse, FakeSession, csv_body

        from seismo.data import ingestion

        rows = [{
            "time": "2024-01-02T00:00:00.000Z", "latitude": 10.0, "longitude": 20.0,
            "depth": 5.0, "mag": 3.5, "id": "a1", "type": "earthquake",
            "updated": "2024-01-03T00:00:00.000Z", "place": "X",
        }]
        fake = FakeSession([FakeResponse(200, csv_body(rows))])
        monkeypatch.setattr(ingestion, "requests", type("R", (), {
            "Session": staticmethod(lambda: fake),
        }))
        main(["--root", str(project_root), "data", "fetch", "--limit", "1"])

        capsys.readouterr()
        assert main(["--root", str(project_root), "--json", "data", "validate"]) == EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert payload["processed_rows"] == 1
        assert payload["hard_violations"] == []

    def test_fetch_json_has_summary_fields(self, project_root, capsys, monkeypatch):
        from conftest import FakeResponse, FakeSession

        from seismo.data import ingestion

        fake = FakeSession([FakeResponse(204, "")])
        monkeypatch.setattr(ingestion, "requests", type("R", (), {
            "Session": staticmethod(lambda: fake),
        }))
        assert main(["--root", str(project_root), "--json",
                     "data", "fetch", "--limit", "1"]) == EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        for key in ("total", "fetched", "empty", "skipped", "failed", "rows"):
            assert key in payload, key


# ---------------------------------------------------------------------------
# CSV export — the Parquet originals stay canonical, CSV is for interchange
# ---------------------------------------------------------------------------

class TestDataExport:
    def _build_processed(self, project_root, capsys, monkeypatch):
        from conftest import FakeResponse, FakeSession, csv_body

        from seismo.data import ingestion

        rows = [{
            "time": "2024-01-02T00:00:00.000Z", "latitude": 10.0, "longitude": 20.0,
            "depth": 5.0, "mag": 4.5, "id": "a1", "type": "earthquake",
            "updated": "2024-01-03T00:00:00.000Z", "place": "Somewhere",
        }]
        fake = FakeSession([FakeResponse(200, csv_body(rows))])
        monkeypatch.setattr(ingestion, "requests", type("R", (), {
            "Session": staticmethod(lambda: fake),
        }))
        assert main(["--root", str(project_root), "data", "fetch", "--limit", "1"]) == EXIT_OK
        capsys.readouterr()
        assert main(["--root", str(project_root), "data", "validate"]) == EXIT_OK
        capsys.readouterr()

    def test_export_writes_csv(self, project_root, capsys, monkeypatch):
        self._build_processed(project_root, capsys, monkeypatch)

        assert main(["--root", str(project_root), "data", "export",
                     "--dataset", "processed"]) == EXIT_OK

        target = project_root / "data" / "export" / "processed.csv"
        assert target.is_file()

        import pandas as pd
        round_trip = pd.read_csv(target)
        assert len(round_trip) == 1
        assert round_trip.loc[0, "id"] == "a1"
        assert float(round_trip.loc[0, "mag"]) == 4.5

    def test_export_missing_dataset_is_actionable(self, project_root, capsys):
        code = main(["--root", str(project_root), "--json",
                     "data", "export", "--dataset", "features"])
        assert code == EXIT_MISSING
        assert "seismo features build" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Startup discipline — lazy imports keep --help fast
# ---------------------------------------------------------------------------

def test_help_does_not_import_pandas(capsys):
    """`seismo --help` must not pull in heavy data libraries."""
    import subprocess
    import sys

    code = (
        "import sys, io, contextlib\n"
        "from seismo.cli import main\n"
        "buf = io.StringIO()\n"
        "with contextlib.redirect_stdout(buf):\n"
        "    try:\n"
        "        main(['--help'])\n"
        "    except SystemExit:\n"
        "        pass\n"
        "loaded = [m for m in ('pandas', 'pyarrow', 'numpy') if m in sys.modules]\n"
        "print('LOADED:' + ','.join(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "LOADED:" in result.stdout
    assert "LOADED:" == result.stdout.strip() or result.stdout.strip() == "LOADED:"
