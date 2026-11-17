from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import ttest_ind

HERE = Path(__file__).resolve().parent
PULSES_PER_TRIAL = 50
BASELINE_SECONDS = 2

def scalar_int(value: object) -> int:
    return int(np.asarray(value).squeeze())

def reconstruct_trials(params_path: Path, times_path: Path, offset: int) -> pd.DataFrame:
    trials = np.asarray(sio.loadmat(params_path)["trials"])
    stim_arr = sio.loadmat(times_path)["stim_times_arr"]
    by_channel = {
        scalar_int(stim_arr[i, 0]): np.asarray(stim_arr[i, 1]).astype(np.int64).ravel()
        for i in range(stim_arr.shape[0])
    }
    cursors = {channel: 0 for channel in by_channel}
    rows = []
    for trial_index, trial in enumerate(trials):
        current = float(trial[0])
        channel = int(trial[1])
        cursor = cursors[channel]
        block = by_channel[channel][cursor : cursor + PULSES_PER_TRIAL]
        cursors[channel] += PULSES_PER_TRIAL
        if len(block) != PULSES_PER_TRIAL:
            raise ValueError(f"Trial {trial_index} channel {channel} has {len(block)} pulses")
        rows.append(
            {
                "trial_index_global_order": trial_index,
                "current_uA": current,
                "stim_channel": channel,
                "trial_start_local_sample": int(block[0]),
                "trial_end_local_sample": int(block[-1]),
                "trial_start_absolute_sample": int(block[0] + offset),
                "trial_end_absolute_sample": int(block[-1] + offset),
            }
        )
    if any(cursors[channel] != len(values) for channel, values in by_channel.items()):
        raise ValueError("Not all channel-grouped stimulation timestamps were consumed")
    result = pd.DataFrame(rows)
    if not result["trial_start_local_sample"].is_monotonic_increasing:
        raise ValueError("Reconstructed randomized trials are not chronological")
    return result

def rates_for_trials(spikes: np.ndarray, trials: pd.DataFrame, fs: float) -> tuple[np.ndarray, np.ndarray]:
    starts = trials["trial_start_absolute_sample"].to_numpy(dtype=np.int64)
    ends = trials["trial_end_absolute_sample"].to_numpy(dtype=np.int64)
    durations = np.maximum((ends - starts) / fs, 1e-9)
    stim_counts = np.searchsorted(spikes, ends, side="left") - np.searchsorted(spikes, starts, side="left")
    baseline_starts = starts - int(BASELINE_SECONDS * fs)
    base_counts = np.searchsorted(spikes, starts, side="left") - np.searchsorted(
        spikes, baseline_starts, side="left"
    )
    return stim_counts / durations, base_counts / BASELINE_SECONDS

def main(out):
    all_events = []
    ledgers = []
    for session in pd.read_csv(HERE/'source'/'sessions.csv').itertuples():
        folder = HERE / 'source' / session.animal
        trials = reconstruct_trials(folder / (session.session_date+'_params.mat'),
                                    folder / (session.session_date+'_pulses.mat'), session.segment_offset_samples)
        trials.insert(0,'session_date',session.session_date)
        trials.insert(0,'animal',session.animal)
        ledgers.append(trials)
        with np.load(folder/'spikes.npz',allow_pickle=False) as z:
            ids = z['unit_ids']
            samples, labels, fs = z['samples'], z['unit'], float(z['sampling_frequency'][0])
        spikes = [np.sort(samples[labels == u]) for u in ids]
        for (channel, current), group in trials.groupby(['stim_channel','current_uA']):
            assert len(group) == 10
            rates = [rates_for_trials(train, group, fs) for train in spikes]
            stim, baseline = np.transpose(np.asarray(rates), (1,0,2))
            p_values = ttest_ind(stim,baseline,equal_var=False,axis=1).pvalue
            mean_stim, mean_base = stim.mean(axis=1), baseline.mean(axis=1)
            included = (p_values <= 0.001) & (mean_stim > mean_base)
            for idx in np.flatnonzero(included):
                all_events.append([session.animal, session.session_date, ids[idx], channel, current,
                                   mean_stim[idx],mean_base[idx],p_values[idx]])
    events = pd.DataFrame(all_events, columns=['animal','session_date','unit_id','stim_channel',
        'current_uA','stim_rate_Hz','baseline_rate_Hz','p_value'])
    reference = pd.read_csv(HERE/'offset_corrected_activation_events.csv')
    keys = ['animal','session_date','unit_id','stim_channel','current_uA']
    events = events.sort_values(keys).reset_index(drop=True)
    reference = reference.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(events[keys],reference[keys],check_dtype=False)
    np.testing.assert_allclose(events[['stim_rate_Hz','baseline_rate_Hz','p_value']],
                               reference[['stim_rate_Hz','baseline_rate_Hz','p_value']],atol=1e-14)
    reconstructed = pd.concat(ledgers,ignore_index=True)
    pd.testing.assert_frame_equal(reconstructed, pd.read_csv(HERE/'reconstructed_trial_ledger.csv'))
    out.mkdir(parents=True,exist_ok=True)
    events.to_csv(out/'reconstructed_activation_events.csv',index=False)
    print(f'{len(events)} positive events and {len(reconstructed)} trial intervals matched.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reconstruct and verify S8C activations from timed, curated spikes.')
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
