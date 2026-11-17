"""Verify the released Figure 3B ADC window and complete stimulus train against the original acquisition."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent


def main(recording, params_path, times_path):
    meta = json.loads((HERE/"provenance.json").read_text(encoding="utf-8"))
    with recording.open("rb") as stream:
        first = stream.read(14)
        assert first[:8] == b"NEURALCD" and first[8:10] == bytes([2,2])
        header_size = struct.unpack_from("<I",first,10)[0]
        stream.seek(0)
        header = stream.read(header_size)
        assert hashlib.sha256(header).hexdigest() == meta["ns5_header_sha256"]
        packet = stream.read(9)
        assert packet[0]==1 and struct.unpack_from("<I",packet,1)[0]==0
        frames = struct.unpack_from("<I",packet,5)[0]
        channels = struct.unpack_from("<I",header,310)[0]
        start,end = meta["window_start_frame"],meta["window_end_frame_exclusive"]
        assert end <= frames
        stream.seek(header_size+9+2*channels*start)
        raw = np.frombuffer(stream.read((end-start)*2*channels),dtype="<i2").reshape(-1,channels)
        with np.load(HERE/"acquisition_window.npz",allow_pickle=False) as cache:
            np.testing.assert_array_equal(raw[:,meta["zero_based_interleaved_index"]], cache["adc_counts"])
    trials = loadmat(params_path)["trials"]
    condition = (trials[:,0]==meta["current_uA"]) & (trials[:,1]==meta["stimulation_channel"])
    trial = np.flatnonzero(condition)[meta["condition_trial_1based"]-1]
    assert trial+1 == meta["global_trial_1based"]
    n_before = np.sum(trials[:trial,1] == meta["stimulation_channel"])
    times = loadmat(times_path)["stim_times_arr"]
    source_times = next(np.asarray(a[1]).ravel() for a in times if int(np.asarray(a[0]).squeeze())==meta["stimulation_channel"])
    n=meta["pulse_count"]
    recorded = np.genfromtxt(HERE/"stimulus_train.csv",delimiter=",",names=True,dtype=None,encoding="utf-8")
    np.testing.assert_array_equal(source_times[n_before*n:(n_before+1)*n],recorded["sample_frame"])
    print(f"{end-start} original ADC values and {n} original stimulus timestamps matched.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns5",type=Path,required=True)
    parser.add_argument("--stim-params",type=Path,required=True)
    parser.add_argument("--stim-times",type=Path,required=True)
    args=parser.parse_args()
    main(args.ns5,args.stim_params,args.stim_times)
