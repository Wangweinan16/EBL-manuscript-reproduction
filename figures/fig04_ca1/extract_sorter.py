"""Extract the curated result and verify the Figure 4 unit against the successful raw sorter output."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('sorter_root',type=Path,help='Directory with sorter_output and waveforms_curated3.')
    cli.add_argument('out',type=Path)
    args = cli.parse_args()
    source, out = args.sorter_root, args.out
    out.mkdir(parents=True,exist_ok=True)
    folder = source/'waveforms_curated3'
    info = json.loads((folder/'sorting/numpysorting_info.json').read_text())
    spikes = np.load(folder/'sorting/spikes.npy',mmap_mode='r')
    unit75 = spikes['sample_index'][spikes['unit_index']==info['unit_ids'].index(75)]
    with np.load(source/'sorter_output/firings.npz') as initial:
        before = initial['spike_indexes_seg0'][initial['spike_labels_seg0']==75]
        np.testing.assert_array_equal(before,unit75)
        np.testing.assert_array_equal(initial['sampling_frequency'],[info['sampling_frequency']])
    np.savez_compressed(out/'curated_sorting.npz',spikes=spikes,unit_ids=info['unit_ids'],sampling_frequency=info['sampling_frequency'])
    for path in [source/'spikeinterface_params.json',source/'spikeinterface_log.json',folder/'recording_info/probegroup.json',
                 folder/'extensions/waveforms/params.json',folder/'spikeinterface_info.json']:
        shutil.copyfile(path,out/path.name)
    print(f'Unit 75 matches all {len(unit75)} original MountainSort4 spike frames exactly.')


if __name__ == '__main__':
    main()
