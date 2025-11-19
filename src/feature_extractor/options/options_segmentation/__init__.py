"""Available segmentation strategies for the sliding-window extractor."""

from config.option_utils import OptionList
from feature_extractor.types import SegmentationOption

__all__ = ["SEGMENTATION_OPTIONS"]

SEGMENTATION_OPTIONS: OptionList = OptionList(
    options=[SegmentationOption(time_window=2, time_step=0.25)],
)
