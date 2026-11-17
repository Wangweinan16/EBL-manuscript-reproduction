"""Reconstruct the recorded sorter call with the five NWB sessions in acquisition order."""
import argparse
import json
from pathlib import Path
import spikeinterface.full as si
from probeinterface import read_probeinterface


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--sessions',type=Path,nargs=5,required=True,metavar='NWB')
    cli.add_argument('--out',type=Path,required=True)
    args=cli.parse_args()
    if args.out.exists():
        raise FileExistsError(f'Please choose a new output directory: {args.out}')
    metadata=Path(__file__).resolve().parent/'sorting'
    inputs=[si.read_nwb_recording(path) for path in args.sessions]
    rec=si.concatenate_recordings(recording_list=inputs)
    group=read_probeinterface(metadata/'probegroup.json')
    rec=rec.set_probe(group.probes[0])
    parameters=json.loads((metadata/'spikeinterface_params.json').read_text())
    si.run_sorter(sorter_name=parameters['sorter_name'],recording=rec,
                  folder=args.out,**parameters['sorter_params'])


if __name__=='__main__':
    main()
