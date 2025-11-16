from config import (
    OptionList,
    SegmentationOption,
)

SEGMENTATION_OPTIONS: OptionList = OptionList(
    [SegmentationOption(time_window=2, time_step=0.25)]
)
