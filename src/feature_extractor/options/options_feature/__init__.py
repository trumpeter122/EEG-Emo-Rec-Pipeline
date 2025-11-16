from config import FeatureOption, OptionList

from .option_dasm import _extract_dasm
from .option_de import _extract_de
from .option_deasm import _extract_deasm
from .option_psd import _extract_psd

FEATURE_OPTIONS: OptionList = OptionList(
    [
        FeatureOption(name="psd", feature_channel_extraction_method=_extract_psd),
        FeatureOption(name="de", feature_channel_extraction_method=_extract_de),
        FeatureOption(name="deasm", feature_channel_extraction_method=_extract_deasm),
        FeatureOption(name="dasm", feature_channel_extraction_method=_extract_dasm),
    ]
)
