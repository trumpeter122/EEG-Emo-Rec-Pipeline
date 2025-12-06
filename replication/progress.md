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

Implementation Detail: Subject-dependent Emotiv replication with the paper’s 4 frontal channels (FC5, F4, F7, AF3 as `emotiv_frontal_4`), Higuchi FD + 36 HOC + 6 statistical features combined and z-scored across channels (`hoc_stat_fd_frontal4`), 4 s windows with 1 s step (`w4.00_s1.00`), clean preprocessing, and SVM classifier (`svc_rbf_sklearn`) on a 70/30 valence classification split (`valence+use1.00+test0.30+seed42+classification+standard`). Window length matches the 512-sample (~4 s) segments with 75% overlap used in the paper.

## Research Paper 05

Implementation Detail: Expert epileptic-detection model approximated with DEAP by using clean preprocessing, full 32-channel montage (`standard_32`), PSD bandpower features (`psd`), 2 s/0.25 s sliding windows (`w2.00_s0.25`), and an RBF SVM classifier (`svc_rbf_sklearn`) on a 70/30 valence classification split with standard scaling (`valence+use1.00+test0.30+seed42+classification+standard`). This mirrors the paper’s energy-based spectral focus and SVM classifier within the project’s valence/arousal framework.

## Research Paper 06

Implementation Detail: Neural-mass-inspired spectral replication using clean preprocessing, full 32-channel montage (`standard_32`), PSD bandpower features (`psd`), 2 s / 0.25 s sliding windows (`w2.00_s0.25`), ridge regression model (`ridge_regression_sklearn`) with the sklearn regression trainer, and a 70/30 valence regression split with standard scaling (`valence+use1.00+test0.20+seed42+regression+standard`). This emphasizes frequency-domain dynamics akin to the paper’s spectral modeling focus within the project’s valence target schema.

## Research Paper 07

Implementation Detail: Wavelet-coefficient + neural network approximation using clean preprocessing with full 32 channels (`standard_32`), wavelet energy/entropy/statistical features (`wavelet_energy_entropy_stats`), 2 s/0.25 s windows (`w2.00_s0.25`), and CNN classifier (`cnn1d_n1_classification`) trained with Adam on a 70/30 valence classification split (`valence+use1.00+test0.30+seed42+classification+standard`). This mirrors the paper’s DWT feature extraction feeding neural network ensembles within the project’s pipeline constraints.

## Research Paper 08

Implementation Detail: Wavelet-coefficient NN ensemble approximated with DEAP by using clean preprocessing, full 32 channels (`standard_32`), differential entropy features (`de`) as surrogate wavelet/statistical inputs, 2 s/0.25 s windows (`w2.00_s0.25`), linear SVM classifier (`linear_svc_sklearn`), and a 70/30 valence classification split with standard scaling (`valence+use1.00+test0.30+seed42+classification+standard`). This reflects the paper’s time–frequency DWT feature extraction feeding neural networks/linear separators.

## Research Paper 09

Implementation Detail: Visual RSVP/object-recognition inspired replication using minimal filtering (`unclean`), a posterior/parietal occipital pick approximating the paper’s 17-channel layout (`posterior_parietal_14`), zero-mean raw waveform features (`raw_waveform`), short 1 s windows with 0.25 s hop to mirror the paper’s downsampled epochs (`w1.00_s0.25`), and a linear regression mapper (`linear_regression_sklearn`) on a valence regression split with standard scaling (`valence+use1.00+test0.20+seed42+regression+standard`). This mirrors the paper’s linear DNN-to-EEG encoding focus with downsampled posterior activity while fitting the project’s DEAP framework.

## Research Paper 10

Implementation Detail: Epilepsy-oriented DWT + K-means + MLP setup approximated with clean preprocessing, full 32-channel DEAP montage (`standard_32`), db4 wavelet energy/entropy/statistical features (`wavelet_energy_entropy_stats`) as a surrogate for clustered wavelet distributions, long 4 s non-overlapping windows (`w4.00_s4.00`), and a torch MLP classifier (`mlp_classification`) trained with Adam on a valence classification split using full data and 30% test holdout with standard scaling (`valence+use1.00+test0.30+seed42+classification+standard`). Mirrors the paper’s wavelet subband representation feeding an MLP while fitting the project’s pipeline.

## Research Paper 11

Implementation Detail: SEEG computational-model approximation using clean preprocessing, temporal-focused montage (`minimal_temporal_augmented`) to emulate hippocampal depth emphasis, PSD bandpower features (`psd`), short 1 s windows with 0.25 s hop to capture pre-onset fast oscillations (`w1.00_s0.25`), and an RBF SVR regressor (`svr_rbf_sklearn`) on a valence regression split with full data and 20% test holdout using standard scaling (`valence+use1.00+test0.20+seed42+regression+standard`). This mirrors the paper’s spectral parameter-identification over interictal-to-ictal transitions within the project framework.

## Research Paper 12

Implementation Detail: EEGLAB/BCILAB-style end-to-end preprocessing and feature pipeline using ICA-cleaned data (`ica_clean`), full 32-channel montage (`standard_32`), differential entropy plus asymmetry features to approximate source-informed measures (`deasm`), standard 2 s / 0.25 s sliding windows (`w2.00_s0.25`), and a fast logistic regression classifier (`logreg_sklearn`) trained via the sklearn classification method on a valence split with full data and 30% test holdout with standard scaling (`valence+use1.00+test0.30+seed42+classification+standard`). This mirrors the paper’s ICA-centric, source/BCI-friendly workflow within the project constraints.

## Research Paper 13

Implementation Detail: Bi-hemispheric discrepancy-inspired deep pipeline using clean preprocessing, full 32-channel montage (`standard_32`), differential entropy with asymmetry features to emphasize hemispheric differences (`deasm`), 2 s / 0.25 s sliding windows (`w2.00_s0.25`), and a CNN classifier (`cnn1d_n1_classification`) trained with Adam on a full-use 70/30 valence classification split with standard scaling (`valence+use1.00+test0.30+seed42+classification+standard`). This approximates the paper’s hemispheric traversal + discrepancy network within the project framework.

## Research Paper 14

Implementation Detail: Multimodal-modeling inspired regression using clean preprocessing, full 32-channel montage (`standard_32`), differential entropy features (`de`) representing EEG spectral content, standard 2 s / 0.25 s sliding windows (`w2.00_s0.25`), and an elastic net regressor (`elasticnet_regression_sklearn`) on a valence regression split with full data and 20% test holdout plus standard scaling (`valence+use1.00+test0.20+seed42+regression+standard`). This approximates the paper’s joint modeling emphasis by pairing rich spectral features with regularized regression within the project pipeline.
