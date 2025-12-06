"""Predefined channel pick configurations."""

from config import GENEVA_32
from config.option_utils import OptionList
from feature_extractor.types import ChannelPickOption

__all__ = ["CHANNEL_PICK_OPTIONS"]

CHANNEL_PICK_OPTIONS: OptionList = OptionList(
    options=[
        ChannelPickOption(name="minimal_frontal", channel_pick=["F3", "F4"]),
        ChannelPickOption(
            name="minimal_frontal_parietal",
            channel_pick=["F3", "F4", "P3", "P4"],
        ),
        ChannelPickOption(name="minimal_frontopolar", channel_pick=["Fp1", "Fp2"]),
        ChannelPickOption(
            name="frontal_prefrontal_4",
            channel_pick=["Fp1", "Fp2", "F3", "F4"],
        ),
        ChannelPickOption(
            name="emotiv_frontal_3",
            channel_pick=["AF3", "F4", "FC6"],
        ),
        ChannelPickOption(
            name="emotiv_frontal_4",
            channel_pick=["FC5", "F4", "F7", "AF3"],
        ),
        ChannelPickOption(
            name="minimal_temporal_augmented",
            channel_pick=["Fp1", "Fp2", "T7", "T8"],
        ),
        ChannelPickOption(
            name="balanced_classic_6",
            channel_pick=["Fp1", "Fp2", "F3", "F4", "P3", "P4"],
        ),
        ChannelPickOption(
            name="balanced_temporal",
            channel_pick=["F3", "F4", "T7", "T8", "P3", "P4"],
        ),
        ChannelPickOption(
            name="balanced_data_driven_5",
            channel_pick=["Fp1", "AF3", "FC2", "P3", "O1"],
        ),
        ChannelPickOption(
            name="optimized_gold_standard_8",
            channel_pick=["Fp1", "Fp2", "F3", "F4", "P3", "P4", "T7", "T8"],
        ),
        ChannelPickOption(
            name="optimized_lateral_mix",
            channel_pick=["F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz"],
        ),
        ChannelPickOption(
            name="optimized_relief_nmi",
            channel_pick=["AF3", "AF4", "F3", "F4", "FC1", "FC2", "P3", "P4"],
        ),
        ChannelPickOption(
            name="posterior_parietal_14",
            channel_pick=[
                "PO3",
                "PO4",
                "O1",
                "O2",
                "Oz",
                "P3",
                "P4",
                "P7",
                "P8",
                "Pz",
                "CP1",
                "CP2",
                "CP5",
                "CP6",
            ],
        ),
        ChannelPickOption(
            name="frontal_full_10",
            channel_pick=[
                "Fp1",
                "Fp2",
                "AF3",
                "AF4",
                "F3",
                "F4",
                "F7",
                "F8",
                "Fz",
                "FC1",
            ],
        ),
        ChannelPickOption(name="standard_32", channel_pick=GENEVA_32),
    ],
)
