from pathlib import Path
import argparse
import gzip
import json
import hashlib
import numpy as np
import pandas as pd
import scipy.io as sio

HERE = Path(__file__).resolve().parent
PULSES_PER_TRAIN = 50

def pulse_train_ledger(waveforms: Path) -> pd.DataFrame:
    trials = np.asarray(sio.loadmat(waveforms / "stim_params.mat")["trials"])
    stim_arr = sio.loadmat(waveforms / "stim_times_arr.mat")["stim_times_arr"]
    by_channel = {
        int(np.asarray(stim_arr[index, 0]).squeeze()): np.asarray(stim_arr[index, 1], dtype=np.int64).ravel()
        for index in range(stim_arr.shape[0])
    }
    cursors = {channel: 0 for channel in by_channel}
    train_index_by_condition: dict[tuple[int, float], int] = {}
    rows = []
    for trial_index, trial in enumerate(trials):
        current = float(trial[0])
        channel = int(trial[1])
        cursor = cursors[channel]
        block = by_channel[channel][cursor : cursor + PULSES_PER_TRAIN]
        cursors[channel] += PULSES_PER_TRAIN
        if len(block) != PULSES_PER_TRAIN:
            raise ValueError(f"{waveforms.name} trial {trial_index} channel {channel} has {len(block)} pulses")
        key = (channel, current)
        train_index = train_index_by_condition.get(key, 0) + 1
        train_index_by_condition[key] = train_index
        rows.append(
            {
                "stim_channel": channel,
                "current_uA": current,
                "global_trial_index": trial_index,
                "train_index_within_channel_current": train_index,
                "first_pulse_sample": int(block[0]),
                "last_pulse_sample": int(block[-1]),
                "n_pulses": int(len(block)),
                "pulse_samples": block,
            }
        )
    return pd.DataFrame(rows)


def main(out):
    with gzip.open(HERE / 'spikes.npy.gz','rb') as compressed:
        data = compressed.read()
    assert hashlib.sha256(data).hexdigest() == (HERE/'source_sha256.txt').read_text().split()[0]
    import io
    spikes_all = np.load(io.BytesIO(data), allow_pickle=False)
    parent = json.loads((HERE/'numpysorting_info.json').read_text())
    selected = pd.read_csv(HERE.parent/'S11_S12'/'observations.csv')
    pulses = pd.read_csv(HERE.parent/'S10B'/'pulses.csv')
    scan = pd.read_csv(HERE.parent/'S10B'/'current_scan.csv')
    raster = pd.read_csv(HERE.parent/'S10B'/'raster.csv')
    units = np.asarray(parent['unit_ids'])[spikes_all['unit_index']]
    rate_rows = []
    for sess in json.loads((HERE/'session_slices.json').read_text()):
        idx = sess['session']
        sr = sess['sampling_frequency_hz']
        samples = spikes_all['sample_index']
        keep = (samples >= sess['start_frame']) & (samples < sess['end_frame']) & np.isin(units,sess['unit_ids'])
        local_unit = units[keep]
        local_sample = samples[keep] - sess['start_frame']
        trains = pulse_train_ledger(HERE/f'session_{idx}')
        selected_events = selected[selected.trial == idx]
        for evt in selected_events.itertuples():
            condition = trains[(trains.stim_channel==evt.ch)&(trains.current_uA==evt.current)]
            assert len(condition)==10 and evt.unit_id in sess['unit_ids']
            t = local_sample[local_unit==evt.unit_id]
            starts = condition.first_pulse_sample.to_numpy()
            ends = condition.last_pulse_sample.to_numpy()
            counts = np.searchsorted(t,ends)-np.searchsorted(t,starts)
            pre = np.searchsorted(t,starts)-np.searchsorted(t,np.maximum(0,starts-int(2*sr)))
            observed = counts.sum()/((ends-starts)/sr).sum()
            baseline = pre.sum()/(2*len(pre))
            np.testing.assert_allclose([observed, baseline], [evt.stim_train_rate_Hz, evt.baseline_pre2s_rate_Hz])
            rate_rows.append([idx, evt.unit_id, evt.ch, evt.current, observed, baseline])
        for current, group in trains[trains.stim_channel == 121].groupby('current_uA'):
            pul = np.concatenate(group.pulse_samples.to_numpy())
            known = pulses[(pulses.session_index==idx)&(pulses.current_uA==current)].sort_values('pulse_index_within_channel_current')
            np.testing.assert_array_equal(pul,known.sample_index)
            unit253 = local_sample[local_unit==253]
            lo = np.searchsorted(unit253,pul-int(round(3.5*sr/1000)),side='left')
            hi = np.searchsorted(unit253,pul+int(round(9.5*sr/1000)),side='right')
            scan_entry = scan[(scan.session_index==idx)&(scan.current_uA==current)].iloc[0]
            np.testing.assert_equal((hi-lo).sum(),scan_entry.aligned_spike_count)
            if current == 4.0:
                expected = raster[raster.session_index==idx]
                np.testing.assert_array_equal(np.repeat(pul,hi-lo),expected.pulse_sample)
                np.testing.assert_array_equal(np.concatenate([unit253[a:b] for a,b in zip(lo,hi)]),expected.spike_sample)
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rate_rows,columns=['session','unit_id','stim_channel','current_uA','stim_rate_hz','baseline_rate_hz']).to_csv(out/'replayed_rates.csv',index=False)
    print(f'{len(rate_rows)} session-level rate pairs and all seven selected 500-pulse rasters verified from curated spikes.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check S10B and S12 rates from the curated longitudinal sorter output.')
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
