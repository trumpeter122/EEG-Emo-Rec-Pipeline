"""Feature extractors offered by the pipeline."""

from config import OptionList

from .option_dasm import _dasm
from .option_de import _de
from .option_deasm import _deasm
from .option_psd import _psd
from .option_wavelet_energy import _wavelet_energy_entropy_stats

__all__ = ["FEATURE_OPTIONS"]

FEATURE_OPTIONS: OptionList = OptionList(
    options=[
        _psd,
        _de,
        _deasm,
        _dasm,
        _wavelet_energy_entropy_stats,
    ],
)
