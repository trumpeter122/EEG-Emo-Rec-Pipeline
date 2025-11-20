# Repository Guidelines

## Project Structure & Module Organization
- Core source lives in `src/`, split by pipeline phase: `preprocessor/`, `feature_extractor/`, `model_trainer/`, `pipeline_runner/`, and shared utilities in `config/` and `utils/`. Each module carries its option dataclasses in `module/types.py` and exposes executors via `module/core.py`.
- Reference assets for experiments sit in `docs/` (see `docs/references/` for notebooks) while reproducible artifacts land in `results/`. Raw and interim EEG assets belong under `data/`; keep personal datasets outside the repo to avoid accidental commits.
- Scripts that orchestrate dataset prep or automation live in `scripts/`. Add tests or fixtures under a new `tests/` directory mirroring the `src/` layout (e.g., `tests/model_trainer/test_core.py`).

## Build, Test, and Development Commands
- `uv run python src/main.py` — runs the sample end-to-end regression/classification experiment defined in `main.py`.
- `uv run ruff format src` — applies the canonical formatter (PEP 8 layout, import sorting).
- `uv run ruff check --fix src` — enforces lint rules and autocorrects safe issues.
- `uv run mypy src && uv run pyright src` — static typing passes; both must succeed before a PR.
- `uv run pytest` — execute behavioral tests once they exist; prefer focused unit suites over notebooks when validating logic.

## Coding Style & Naming Conventions
- Follow 4-space indentation, explicit typing everywhere, and keyword arguments for readability; avoid default parameters unless they represent true invariants.
- Keep option/config dataclasses colocated with their module (e.g., `model_trainer/types.py:ModelTrainingOption`) and import them via `config.__getattr__` only when shared.
- Prefer descriptive module names over abbreviations (e.g., `feature_extractor/core.py`), but keep variable names short (`feature_option`, `seg_frame`).
- Run Ruff + type checks before pushing; code review expects clean linters.

## Testing Guidelines
- Use `pytest` with files named `test_<module>.py`. Arrange tests by pipeline stage and rely on lightweight fixtures or generated arrays (see `_SegmentDataset` usage) instead of large EEG files.
- Add regression tests whenever fixing bugs in preprocessing or model training; replicate doc-reference baselines via saved configs in `results/`.
- Verify the exemplar `python src/main.py` path to catch integration regressions before submitting.

## Commit & Pull Request Guidelines
- Follow the existing Conventional Commit flavor (`feat(scope): message`, `fix(*)`, `refactor(preprocessor): rename submodule`). Squash trivial WIP commits locally.
- Pull requests should summarize the motivation, list key modules touched, note dataset/model impacts, and mention how validation was performed (commands + dataset variants). Link related issues or reference docs/notebooks when relevant and include screenshots or metrics if behavior changes.
