# EEG-Based Emotion Recognition Research Replication

Replicate the research papers by adding them as pipelines into the project.

If the research paper does not provide enough information for building a pipeline for replication (for example, not providing the list of used EEG channels or not providing the model architecture), approximate by using already existent options or adding new options.

If the research paper's pipeline is in compatible with the project's pipeline framework, coerce them to follow the project's framework (for example, classification pipelines using different classes than 1-9 should be forced to use the 1-9 classes in this project). Try to approximate the research pipeline as close as possible as long as it follows the projects set up.

Write the implementation details in the sections down below.

## Research Paper 01

Implementation Detail: Added a db4 wavelet energy/entropy + statistical feature extractor (`wavelet_energy_entropy_stats`), a 4-channel prefrontal/frontal pick (`frontal_prefrontal_4`), and a 2 s non-overlapping window (`w2.00_s2.00`). Replication runs use `ica_clean` preprocessing (aligns with the paper’s BSS artifact removal), the new 70/30 dataset split options (`test0.30` variants), and sklearn classifiers covering SVM RBF (`svc_rbf_sklearn`), KNN, and QDA via `sklearn_default_classification`; class labels are coerced to the project’s 1–9 valence/arousal scheme.

## Research Paper 02

Implementation Detail: Emotiv-style minimal pipeline using AF3/F4/FC6 channels (`emotiv_frontal_3`) with Higuchi fractal-dimension features plus AF3–F4 asymmetry (`higuchi_fd_frontal`), 4 s non-overlapping windows (`w4.00_s4.00`), unfiltered DEAP resample (`unclean` preprocessing), linear SVM classifier (`linear_svc_sklearn`), and 70/30 valence classification split with standard scaling (`valence+use1.00+test0.30+seed42+classification+standard`). This approximates the paper’s 1024-sample Higuchi FD thresholds and 6-emotion mapping within the project’s valence/arousal framework.

## Research Paper 03

Implementation Detail: Tutorial/review-inspired deep pipeline using full DEAP montage (`standard_32`), clean preprocessing (`clean`), differential entropy features (`de`), 2 s sliding windows with 0.25 s step (`w2.00_s0.25`), CNN-based classifier (`cnn1d_n1_classification`) trained with Adam, and a 70/30 valence classification split with standard scaling (`valence+use1.00+test0.30+seed42+classification+standard`). This approximates the paper’s handcrafted-feature + deep classifier route discussed for DBNs/CNNs over PSD/DE features.

## Research Paper 04

Implementation Detail: **TODO**

## Research Paper 05

Implementation Detail: **TODO**

## Research Paper 06

Implementation Detail: **TODO**

## Research Paper 07

Implementation Detail: **TODO**

## Research Paper 08

Implementation Detail: **TODO**

## Research Paper 09

Implementation Detail: **TODO**

## Research Paper 10

Implementation Detail: **TODO**

## Research Paper 11

Implementation Detail: **TODO**

## Research Paper 12

Implementation Detail: **TODO**

## Research Paper 13

Implementation Detail: **TODO**
