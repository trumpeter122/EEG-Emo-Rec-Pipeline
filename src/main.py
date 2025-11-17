if __name__ == "__main__":
    # Currently for test runs only
    # So, ignore the codes here at present

    from preprocessor import run_preprocessor
    from preprocessor.options import PREPROCESSING_OPTIONS

    run_preprocessor(PREPROCESSING_OPTIONS.get_name("clean"))
    run_preprocessor(PREPROCESSING_OPTIONS.get_name("ica_clean"))

    # import joblib

    # df = joblib.load("./data/DEAP/generated/cleaned/trial/t01.joblib")
    # print(df)

    # print("data shape: ", df.iloc[0].get("data").shape)

    from config import FeatureExtractionOption
    from feature_extractor import (
        CHANNEL_PICK_OPTIONS,
        FEATURE_OPTIONS,
        SEGMENTATION_OPTIONS,
        run_feature_extractor,
    )

    from itertools import product
    import random

    ppop = PREPROCESSING_OPTIONS.get_name("clean")

    feop = FeatureExtractionOption(
        preprocessing_option=ppop,
        feature_option=FEATURE_OPTIONS.get_name("psd"),
        channel_pick_option=CHANNEL_PICK_OPTIONS.get_name("standard_32"),
        segmentation_option=SEGMENTATION_OPTIONS.get_name("2.00s_0.25s"),
    )

    run_feature_extractor(feop)
