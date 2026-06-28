"""ezmodel's pyclawd config — drives `pyclawd test/lint/typecheck/...` for this repo."""

from pyclawd import (
    DescriptionConfig,
    DoctorConfig,
    GoldenConfig,
    Project,
    QualityConfig,
    TestConfig,
)

project = Project(
    name="ezmodel",
    conda_env=None,
    root_markers=["pyproject.toml", "ezmodel/__init__.py"],
    # The pyclawd this config was built on. `pyclawd doctor` WARNs if the
    # running pyclawd has drifted to a different minor (migration may be needed).
    pyclawd_version="0.1.0",
    # Default directory `pyclawd ls` lists (the code/source root).
    src_dir="ezmodel",
    quality=QualityConfig(
        lint_cmd=["ruff", "check"],
        lint_fix_cmd=["ruff", "check", "--fix"],
        format_cmd=["ruff", "format"],
        format_check_cmd=["ruff", "format", "--check", "--quiet"],
        typecheck_cmd=["mypy"],
        check_sequence=["format-check", "lint", "typecheck", "descriptions", "test"],
    ),
    # One-line-per-file code-map gate. experimental/ (rough, partially-broken
    # scratch code) and scratch.py are out of scope and excluded.
    descriptions=DescriptionConfig(
        include=[r"\.py$"],
        exclude=[r"experimental/", r"scratch\.py$"],
    ),
    test=TestConfig(
        tests_dir="tests",
        classname_prefix="tests.",
        integration_files=[],
        # default tier is the env-portable gate: excludes optional-backend usage
        # tests (need GPy/gpflow/smt/pySOT) and golden snapshots (own gate).
        markers={
            "fast": "not slow and not optional and not golden",
            "default": "not slow and not optional and not golden",
            "all": "not golden",
        },
    ),
    # Behavior-regression oracle (run via `pyclawd golden`); see tests/golden/.
    golden=GoldenConfig(),
    doctor=DoctorConfig(
        core_deps=["numpy", "scipy", "pandas", "sklearn"],
        dev_deps=["pytest", "pytest-xdist", "pytest-cov"],
        tool_files=[],
        binaries=[
            ("ruff", "pip install ruff"),
            ("mypy", "pip install mypy"),
        ],
    ),
)
