# Project Documentation

## Overview
This repository implements a modular EEG emotion-recognition pipeline built around the DEAP dataset. The codebase emphasizes explicit configuration via typed “option” objects that describe every stage of the workflow. A run consists of three phases:

1. **Preprocessing** – converts the raw DEAP .mat files into cleaned NumPy arrays and structured trial/feature stores.
2. **Feature Extraction** – slices the cleaned signals into segments, derives engineered features, and persists per-segment data frames.
3. **Model Training** – consumes the extracted features, builds train/test datasets, and trains PyTorch models with reproducible hyper-parameters.

Each phase is authored as its own module with well-defined APIs so that new options (preprocessors, feature sets, models, etc.) can be added without touching the rest of the pipeline. `pipeline_runner` ties these modules together and materializes experiment grids defined in `main.py`.

## Repository Layout

```
src/
  config/                # Global constants and shared option utilities
  preprocessor/          # Preprocessing methods, registries, and option definitions
  feature_extractor/     # Feature extraction logic and option registries
  model_trainer/         # Training orchestration, option registries, neural models
    options/             # Model and training method registries
    types/               # Fine-grained option/data class definitions
  pipeline_runner/       # End-to-end orchestration helpers
  utils/                 # Progress logging and filesystem helpers
main.py                  # Development entry point using pipeline_runner
```

Results are written into `results/<feature_option>/models/...`, while intermediate preprocessing artifacts live under `data/DEAP/generated/<preprocessor-name>/`.

## Processing Pipeline
1. **Preprocessor (`preprocessor/core.py`)**  
   `run_preprocessor(preprocessing_option=...)` executes a registered preprocessing method (e.g., clean, ICA) against every DEAP subject. The `PreprocessingOption` (defined in `preprocessor/types.py`) describes the root output directory and holds helpers to obtain subject/trial/feature paths. Preprocessing methods live under `preprocessor/options` and must conform to the `PreprocessingMethod` protocol (`config/option_utils.py`).

2. **Feature Extractor (`feature_extractor/core.py`)**  
   `run_feature_extractor(feature_extraction_option=...)` loads the preprocessed feature frames, picks channels, applies the configured `FeatureOption` extraction callable, and writes per-segment joblib frames. A `FeatureExtractionOption` combines four lower-level options: the preprocessing run, feature method, channel pick, and segmentation parameters.

3. **Model Trainer (`model_trainer/core.py`)**  
   `run_model_trainer(model_training_option=...)` builds datasets from the extracted features, constructs PyTorch models, executes training loops with `rich`-powered progress bars, and writes metrics/state under the option-specific results directory. Types for datasets, training methods, models, and experiment aggregation live in `model_trainer/types/`.

4. **Pipeline Runner (`pipeline_runner/core.py`)**  
   Provides the bridge between option registries and executable runs. `TrainingExperiment` captures downstream training intent (target column, scaler, method/model names, data splits). `run_pipeline(...)` first ensures selected preprocessing options are materialized, builds all `FeatureExtractionOption` combinations, and finally pairs each with the supplied training experiments, invoking `run_model_trainer`.

## Configuration & Option System
All selectable knobs are represented as dataclasses living next to the module they configure:

| Module | Option classes | Registries |
| ------ | -------------- | ---------- |
| `preprocessor` | `PreprocessingOption` | `PREPROCESSING_OPTIONS` |
| `feature_extractor` | `ChannelPickOption`, `FeatureOption`, `SegmentationOption`, `FeatureExtractionOption` | `CHANNEL_PICK_OPTIONS`, `FEATURE_OPTIONS`, `SEGMENTATION_OPTIONS` |
| `model_trainer` | `TrainingDataOption`, `TrainingMethodOption`, `TrainingOption`, `ModelOption`, `ModelTrainingOption` | `MODEL_OPTIONS`, `TRAINING_METHOD_OPTIONS` |

Registries are implemented via `OptionList` (in `config/option_utils.py`). `OptionList.get_name(name="...")` retrieves a concrete option, while iterating over the list walks the available options in order. The option dataclasses expose `to_params()` helpers so metrics metadata can be serialized without referencing live code objects.

The `config` package also contains `constants.py` with DEAP-specific parameters (channel names, sampling frequencies, directory roots) and `option_utils.py` with callable protocols (`PreprocessingMethod`, `FeatureChannelExtractionMethod`, `ModelBuilder`, etc.). `ModelBuilder` now accepts `output_size` explicitly to avoid implicit kwargs overrides.

## Module Highlights
### Model Trainer
- **Types (`model_trainer/types/`)** – split into small files:  
  - `dataset.py` defines `SegmentDataset`, a torch-compatible dataset wrapper.  
  - `training_data.py` constructs datasets from joblib frames, including scaling, target encoding, and split generation.  
  - `training_method.py`, `training.py`, `model.py`, and `model_training.py` encapsulate optimization configs, dataloader wiring, model construction, and artifact locations respectively.
- **Options (`model_trainer/options/`)** – `options_model` registers neural architectures (currently `CNN1D_N1`), while `options_training_method` lists optimizer/criterion configs for regression and classification. New training methods belong here.
- **Core (`model_trainer/core.py`)** – houses the training loop, evaluation helpers, and metric calculations. It validates class counts vs. model output sizes and saves checkpoints atomically through `utils.fs.atomic_directory`.

### Pipeline Runner
The new module cleanly separates orchestration logic from `main.py`. `run_pipeline` receives explicit sequences of options (rather than implicit kwargs) and executes the full grid:

```python
run_pipeline(
    preprocessing_options=PREPROCESSING_OPTIONS.get_names(names=["clean"]),
    feature_options=list(FEATURE_OPTIONS),
    channel_pick_options=list(CHANNEL_PICK_OPTIONS),
    segmentation_options=list(SEGMENTATION_OPTIONS),
    experiments=[...],
    model_options=MODEL_OPTIONS,
    training_method_options=TRAINING_METHOD_OPTIONS,
)
```

This design keeps `main.py` declarative. Adding CLI arguments or YAML-based experiment descriptions later is straightforward: parse inputs, build `TrainingExperiment` instances, and reuse `run_pipeline`.

## Extending the Pipeline

### Adding a New Preprocessing Method
1. Implement a callable that follows the `PreprocessingMethod` protocol and place it under `preprocessor/options`.
2. Register a `PreprocessingOption` inside `preprocessor/options/__init__.py`, passing the method and a unique output directory name.
3. Reference the option by name via `PREPROCESSING_OPTIONS.get_name(name="...")` when building experiments or feeding it into `build_feature_extraction_options`.

### Adding Feature Extraction Techniques
1. Implement a function with the `FeatureChannelExtractionMethod` signature (inputs: numpy array `trial_data`, `channel_pick` list).  
2. Wrap it in a `FeatureOption` within `feature_extractor/options/options_feature/...` and expose it via the module’s registry.
3. Optionally add new `ChannelPickOption` or `SegmentationOption` entries under their respective registries.
4. `build_feature_extraction_options` will automatically generate the cross-product once the option is registered.

### Adding Models or Training Methods
1. Define the model class (e.g., under `model_trainer/options/options_model/`). Ensure it only accepts keyword arguments—`output_size` is always provided by `ModelOption`.
2. Create a builder function returning the instantiated model and register a `ModelOption` in `options_model/__init__.py`.
3. For optimizers/criteria, add a new module under `options_training_method/` that constructs the optimizer and criterion with explicit keyword-only builders. Register it in the `TRAINING_METHOD_OPTIONS` list.
4. Reference the new names from `TrainingExperiment` objects so `run_pipeline` can resolve them.

### Training Data Variants
When experiment logic requires custom target encodings or scalers, introduce new `TrainingExperiment` entries in `main.py` (or a future config file) with unique names. `TrainingDataOption` already supports specifying `class_labels_expected`, `feature_scaler`, `use_size`, and `test_size`, so most variants can be expressed declaratively without touching the implementation.

## Development Workflow
1. **Formatting & Linting** – run `uv run ruff format src && uv run ruff check --fix src`.
2. **Type Checking** – run `uv run mypy src && uv run pyright src`.
3. **Execution** – run `uv run python src/main.py` to execute the default experiment suite. This orchestrates preprocessing, feature extraction, and both regression/classification training runs and may take considerable time on the full dataset.
4. **Results Inspection** – check `results/<feature-option>/models/<model>/<method>/metrics.json` for training/test metrics and `splits.json` for the exact train/test segment IDs. Each option also persists its `params.json` for reproducibility.

## Data & Results Management
- **Preprocessing outputs** are organized under `data/DEAP/generated/<preprocessing-name>/subject|trial|feature`. Each `PreprocessingOption` builds these directories lazily.
- **Feature extraction outputs** live under `<feature_path>/<feature-extraction-name>`, containing joblib files per subject as well as baseline metadata in the nested `metadata/` directory.
- **Training outputs** reside under `results/<feature-extraction-name>/models/<model-name>/<training-method-name>/`, containing params, metrics, splits, and `best_model.pt`.
- **Atomic Writes** – `utils.fs.atomic_directory` ensures that partially written model artifacts never pollute existing results.

## Design Principles Recap
- **Explicitness** – Every configurable value belongs to a dataclass field; no hidden kwargs overrides remain in builders or registries.
- **Modularity** – Each module hosts its own types and options, avoiding the previous monolithic `config/types.py`.
- **Reproducibility** – Random seeds, split fractions, scalers, and label mappings are recorded in `TrainingDataOption.to_params()`, while pipeline metadata is persisted alongside outputs.
- **Extendability** – New experiments usually require only registering additional options, then referencing them from `TrainingExperiment`.

With these patterns in place, contributors can reason about the pipeline by reading a small number of focused modules, confidently add new techniques, and rely on the shared registries + runner to assemble complex experiment grids without duplicating orchestration code.
