import mne
import numpy as np

from config import (
    EEG_ELECTRODES_NUM,
    EPOCH_TMAX,
    EPOCH_TMIN,
    SFREQ_TARGET,
)

from .utils import (
    _apply_filter_reference,
    _base_bdf_process,
    _get_events,
    _prepare_channels,
)


def _ica_clean_bdf(
    raw: mne.io.BaseRaw,
    subject_id: int,
) -> np.ndarray:
    eeg_channels, stim_ch, raw_stim = _prepare_channels(raw=raw)
    events = _get_events(
        raw_stim=raw_stim,
        stim_ch_name=stim_ch,
        subject_id=subject_id,
    )

    _apply_filter_reference(raw=raw)

    epochs = mne.Epochs(
        raw=raw,
        events=events,
        event_id=4,
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        picks=eeg_channels,
        baseline=None,
        preload=True,
        verbose=False,
    )

    n_ica = EEG_ELECTRODES_NUM - 1
    ica = mne.preprocessing.ICA(
        n_components=n_ica,
        method="fastica",
        random_state=23,
        max_iter="auto",
    )
    ica.fit(inst=epochs, verbose=False)

    eog_inds, ecg_inds = [], []
    try:
        eog_inds, _ = ica.find_bads_eog(raw)
    except RuntimeError as e:
        if not (e.args and e.args[0] == "No EOG channel(s) found"):
            raise e

    comp_var = np.var(ica.get_sources(inst=raw).get_data(), axis=1)
    highpower_inds = np.where(comp_var > np.percentile(comp_var, 99))[0].tolist()

    ica.exclude = sorted(set(eog_inds + ecg_inds + highpower_inds))

    cleaned = ica.apply(inst=epochs.copy(), verbose=False)
    data_down = cleaned.resample(sfreq=SFREQ_TARGET).get_data()

    return _base_bdf_process(
        data_down=data_down,
        eeg_channels=eeg_channels,
        subject_id=subject_id,
    )
