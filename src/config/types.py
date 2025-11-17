from collections.abc import Callable
from functools import partial
from pathlib import Path

import mne
import numpy as np

from .constants import BASELINE_SEC, DEAP_ROOT, GENEVA_32


class OptionList(list):
    def get_name(self, name: str):
        for option in self:
            if option.name == name:
                return option

        raise KeyError(f'Name "{name}" does not exist')


class PreprocessingOption:
    def __init__(
        self,
        name: str,
        root_dir: str | Path,
        preprocessing_method: Callable[[mne.io.BaseRaw, int], np.ndarray],
    ):
        self.name = name
        self.root_path = DEAP_ROOT / "generated" / root_dir
        self.preprocessing_method = preprocessing_method

        self.root_path.mkdir(exist_ok=True, parents=True)

    def get_subject_path(self) -> Path:
        path = self.root_path / "subject"
        path.mkdir(exist_ok=True)
        return path

    def get_trial_path(self) -> Path:
        path = self.root_path / "trial"
        path.mkdir(exist_ok=True)
        return path

    def get_feature_path(self) -> Path:
        path = self.root_path / "feature"
        path.mkdir(exist_ok=True)
        return path

    def to_params(self) -> dict:
        return {
            "preprocessing option": {
                "name": self.name,
                "root_path": str(self.root_path),
                "subject_path": str(self.get_subject_path()),
                "trial_path": str(self.get_trial_path()),
                "feature_path": str(self.get_feature_path()),
            }
        }


class ChannelPickOption:
    def __init__(self, name: str, channel_pick: list[str]):
        self.name = name
        if all(channel_name in GENEVA_32 for channel_name in channel_pick):
            self.channel_pick = channel_pick
        else:
            raise ValueError("Invalid channel names")


class FeatureOption:
    def __init__(
        self,
        name: str,
        feature_channel_extraction_method: Callable[
            [np.ndarray, list[str]], np.ndarray
        ],
    ):
        self.name = name
        self.feature_channel_extraction_method = feature_channel_extraction_method


class SegmentationOption:
    def __init__(self, time_window: float, time_step: float):
        if time_window > BASELINE_SEC:
            raise ValueError("Window cannot be longer than the baseline seconds (5s)")

        self.name = f"{time_window:.2f}s_{time_step:.2f}s"
        self.time_window = time_window
        self.time_step = time_step


class FeatureExtractionOption:
    def __init__(
        self,
        preprocessing_option: PreprocessingOption,
        feature_option: FeatureOption,
        channel_pick_option: ChannelPickOption,
        segmentation_option: SegmentationOption,
    ):
        self.name = "+".join(
            [
                preprocessing_option.name,
                feature_option.name,
                channel_pick_option.name,
                segmentation_option.name,
            ]
        )
        self.preprocessing_option = preprocessing_option
        self.feature_option = feature_option
        self.channel_pick_option = channel_pick_option
        self.segmentation_option = segmentation_option

        self.extraction_method = partial(
            self.feature_option.feature_channel_extraction_method,
            channel_pick=self.channel_pick_option.channel_pick,
        )

    # def to_params(self) -> dict:
    #     # TODO
    #     pass


class ModelOption:
    def __init__(self, name: str, model):
        # TODO
        pass


class TrainingOption:
    def __init__(self, name: str):
        # TODO
        pass


class ModelTrainingOption:
    def __init__(
        self,
        feature_extraction_option: FeatureExtractionOption,
        model_option: ModelOption,
        training_option: TrainingOption,
    ):
        self.feature_extraction_option = feature_extraction_option
        self.model_option = model_option
        self.training_option = training_option
