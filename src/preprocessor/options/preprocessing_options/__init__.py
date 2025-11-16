from config import (
    OptionList,
    PreprocessingOption,
)

from .option_clean import _clean_bdf as _clean_bdf
from .option_ica_clean import _ica_clean_bdf as _ica_clean_bdf
from .option_unclean import _unclean_bdf as _unclean_bdf

PREPROCESSING_OPTIONS: OptionList = OptionList(
    [
        PreprocessingOption(
            name="clean",
            root_dir="cleaned",
            preprocessing_method=_clean_bdf,
        ),
        PreprocessingOption(
            name="ica_clean",
            root_dir="ica_cleaned",
            preprocessing_method=_ica_clean_bdf,
        ),
        PreprocessingOption(
            name="unclean",
            root_dir="uncleaned",
            preprocessing_method=_unclean_bdf,
        ),
    ]
)
