"""Replay archived curation graphs on the original rat or tracking sorter firings."""
import argparse
import json
import os
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", choices=("rat","tracking"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str((args.output_dir / ".numba_cache").resolve()))
    import spikeinterface as si
    if si.__version__ != "0.99.1":
        raise ValueError("Use SpikeInterface 0.99.1 for the archived curation semantics.")
    source = Path(__file__).resolve().parent / args.session
    graph = json.loads((source / "curation_graph.json").read_text())
    node = graph
    while True:
        node.setdefault("annotations", {})
        node.setdefault("properties", {})
        if "parent_sorting" not in node["kwargs"]:
            break
        node = node["kwargs"]["parent_sorting"]
    if not node["class"].endswith("NpzSortingExtractor"):
        raise ValueError("Unexpected historical sorter input")
    node["kwargs"]["file_path"] = str(source / "ms5_firings.npz")
    sorting = si.load_extractor(graph)
    ids = sorting.get_unit_ids()
    trains = {str(int(u)):sorting.get_unit_spike_train(u) for u in ids}
    if args.session == "tracking":
        info = json.loads((source / "numpysorting_info.json").read_text())
        saved = np.load(source / "curated_spikes.npz")["spikes"]
        for i, unit in enumerate(info["unit_ids"]):
            np.testing.assert_array_equal(trains[str(unit)], saved["sample_index"][saved["unit_index"]==i])
    np.savez_compressed(args.output_dir / f"{args.session}_curated_spike_frames.npz", **trains, unit_ids=ids, sampling_frequency=sorting.get_sampling_frequency())
    print(f"{args.session}: {len(ids)} curated units, {sum(map(len,trains.values()))} spikes")

if __name__ == "__main__":
    main()
