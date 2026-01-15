# EEG-Based Emotion Recognition Pipeline

This project provides a modular and reproducible pipeline for EEG-based emotion recognition, designed for real-world applications with consumer-grade devices. It includes a comprehensive pipeline for preprocessing, feature extraction, model training, and evaluation, as well as a prototype mobile application for real-time emotion monitoring.

## Features

- **Modular Pipeline:** A five-module pipeline (Preprocessor, Feature Extractor, Model Trainer, Pipeline Runner, Result Explorer) for systematic evaluation of EEG-based emotion recognition techniques.
- **Reproducibility:** Explicit option sets and detailed configurations for reproducible experiments.
- **Data Source:** Utilizes the DEAP (Database for Emotion Analysis using Physiological Signals) dataset.
- **Extensive Experimentation:** The project includes 1176 configurations, spanning various models, features, and classification tasks.
- **Low-Channel Support:** Analysis and configurations for low-channel EEG devices (<= 4, <= 8, <= 12 channels).
- **Real-time Application:** A React Native-based mobile application for real-time emotion visualization.
- **Comprehensive Reporting:** A detailed report summarizing the methodology, results, and future directions.

## Architecture

The project is organized into two main parts: the machine learning pipeline and the mobile application.

### Machine Learning Pipeline

The pipeline consists of five main modules:
1.  **Preprocessor:** Handles data loading, cleaning, and formatting.
2.  **Feature Extractor:** Extracts various features from the EEG signals, such as Power Spectral Density (PSD) and Differential Entropy (DE).
3.  **Model Trainer:** Trains and evaluates both classical machine learning and deep learning models.
4.  **Pipeline Runner:** Orchestrates the execution of the entire pipeline with different configurations.
5.  **Result Explorer:** Provides tools for analyzing and visualizing the results.

### Mobile Application

The mobile application is built with React Native and provides the following features:
-   Real-time monitoring of valence and arousal.
-   Historical data analytics.
-   User-friendly interface with neumorphic design.

## Getting Started

### Prerequisites

-   Python 3.12
-   `uv` for package management

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/trumpeter122/EEG-Emo-Rec-Pipeline.git
    cd EEG-Emo-Rec-Pipeline
    ```
2.  Install the dependencies:
    ```bash
    uv pip install -r requirements.txt
    ```

### Running the Pipeline

The pipeline can be run using the `main.py` script. The configuration of the pipeline is defined in the `src/config` directory.

## Usage

The project is organized into several modules. Here's a brief overview of how to use them:

-   **`src/preprocessor`**: Contains functions for preprocessing the DEAP dataset.
-   **`src/feature_extractor`**: Includes methods for extracting features from the preprocessed data.
-   **`src/model_trainer`**: Provides a framework for training and evaluating models.
-   **`src/pipeline_runner`**: Use this module to run the entire pipeline with your desired configurations.
-   **`src/result_explorer`**: Analyze the results of your experiments using the notebooks and scripts in this directory.

## Results

The project conducted an extensive evaluation of 1176 configurations. The key findings include:

-   **9-class and 5-class classification:** The best performance is achieved with `clean` preprocessing, `DE` features, and a `CF-CNN` classifier on a 32-channel montage.
-   **3-class classification:** `PSD` features with a tree-based model yield the best results.
-   **Low-channel configurations:**
    -   For **<= 4 channels**, the recommended stack is `ICA_clean` preprocessing, `DE` features, and a `CF-CNN` classifier with the `minimal_temporal_augmented` montage.
    -   For **<= 8 and <= 12 channels**, the `optimized_lateral_mix` montage with `clean` preprocessing and `DE` features is recommended.
-   **Regression tasks:** Regression models showed weak performance, indicating that classification is a more reliable approach for this problem.

For a detailed analysis of the results, please refer to the `report.pdf`.

## Contributing

We welcome contributions to this project. Please follow these guidelines:

-   **Consistency:**
    -   Naming, commenting, and general coding styles should be consistent across the project.
    -   For example: Do not use `src` for variable names somewhere and `source` somewhere else.
    -   Prefer abbreviations for common words and full names for uncommon words.
    -   Consistent verb tenses and singular/plural forms.
-   **Documentation & Commenting:** Implement docstrings and comments.
-   **Explicitness:**
    -   Do not set default values for function arguments.
    -   No implicit handling of issues.
    -   Pass keyword arguments instead of positional arguments except for very mundane function calls like `print()`.
-   **Linting & Formatting:** Run `uv run ruff format src && uv run ruff check --fix src && uv run mypy src && uv run pyright src`.
-   **Modularity:** Ensure modular structure.
-   **Style:** Write clear, human-readable codes; prioritize readability.
-   **Typing:** Implement full-typing.
-   **Verbosity:** No printing by default.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.