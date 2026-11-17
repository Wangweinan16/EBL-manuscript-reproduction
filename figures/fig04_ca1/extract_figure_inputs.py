"""Build figure-level CA1 inputs from the trusted local rat data delivery."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np


def read_pickle(path):
    with path.open('rb') as stream:
        return pickle.load(stream)


def extract(source, destination):
    analyzer = source/'waveforms_curated3'
    ids = json.loads((analyzer/'sorting/numpysorting_info.json').read_text())['unit_ids']
    idx = ids.index(75)
    train = np.load(analyzer/'sorting/spikes.npy', mmap_mode='r')
    selected = np.flatnonzero(train['unit_index'] == idx)
    frames = train['sample_index'][selected]
    waveform = np.load(analyzer/'extensions/waveforms/waveforms.npy', mmap_mode='r')
    random = np.load(analyzer/'extensions/random_spikes/random_spikes_indices.npy', mmap_mode='r')
    np.testing.assert_array_equal(random, np.arange(len(train)))
    coords = np.array(json.loads((analyzer/'recording_info/probegroup.json').read_text())['probes'][0]['contact_positions'])
    mask = np.load(analyzer/'sparsity_mask.npy')[idx]
    data = {'all_contact_um':coords, 'unit_contact_um':coords[mask], 'isi_edges_ms':np.arange(0,50,2)}
    stamps = read_pickle(source/'ephys_time.pkl')
    position = read_pickle(source/'position.pkl')
    postime = read_pickle(source/'position_time.pkl')
    offset = 0
    for session, (date, time) in enumerate(stamps.items()):
        # Original notebook indexes spike time in the true concatenation.
        keep = (frames >= offset) & (frames < offset + len(time))
        spike_time = time[frames[keep]-offset]
        # Preserve the separately defined inclusive frame cutoffs of the notebook for waveforms.
        start = offset + session
        wave_ids = selected[(frames >= start) & (frames <= start + len(time))]
        mean = np.asarray(waveform[wave_ids]).mean(axis=0)
        suffix = str([90,91,92,97,104][session])
        data['mean_waveform_uv_'+suffix] = mean[:,:mask.sum()]
        data['waveform_count_'+suffix] = len(wave_ids)
        data['isi_counts_'+suffix] = np.histogram(np.diff(spike_time)*1000,bins=data['isi_edges_ms'])[0]
        xy = position[date]
        is_valid = ~np.isnan(xy).any(axis=1)
        xy, times = xy[is_valid], postime[date][is_valid]
        # kyutils 0.1.101 assigns each spike to the next available tracking sample.
        ix = np.searchsorted(times, spike_time)
        ix = ix[ix < len(xy)]
        bins = ((xy-xy.min(axis=0))//10).astype(int)
        counts = np.zeros(np.ceil((xy.max(axis=0)-xy.min(axis=0))/10).astype(int),dtype='int64')
        occupancy = np.zeros_like(counts)
        np.add.at(occupancy, (bins[:,0],bins[:,1]),1)
        np.add.at(counts,(bins[ix,0],bins[ix,1]),1)
        data['tracking_rate_hz_'+suffix] = 1/np.diff(times).mean()
        data['occupancy_frames_'+suffix] = occupancy
        data['spike_histogram_'+suffix] = counts
        offset += len(time)
    destination.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination,**data)


if __name__ == '__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('source',type=Path)
    cli.add_argument('destination',type=Path)
    opts=cli.parse_args()
    extract(opts.source,opts.destination)
