"""Run a figure workflow from anywhere. No external data, network or git needed for the default example."""
from pathlib import Path
import argparse,subprocess,sys,os
ROOT=Path(__file__).resolve().parent
# key: [(folder, script, arguments)] ; @out resolved to package-local outputs
RUN={
 '1B':[('figure1_simulation','plot.py','--rsite-ohm 2000000 --outdir @out')],
 '1JK':[('figure1_electrochemistry','plot.py','--output-dir @out')],
 '2AC':[('main2ac','plot.py','--output-dir @out')],
 '2B':[('figure2_density','validate.py',''),('figure2_density','plot.py','--output-dir @out')],
 '2DFGH':[('figure2_longitudinal','waveforms_and_maps.py','--output-dir @out'),('figure2_longitudinal','location_stability.py','--output-dir @out')],
 '2E':[('figure2_longitudinal','plot_isi.py','--unit 29 --output-dir @out'),('figure2_longitudinal','plot_isi.py','--unit 31 --output-dir @out')],
 '3B':[('fig3b_trace','plot.py','--out @out')],
 '3ACDG':[('figure3_stimulation','plot.py','--output-dir @out')],
 '3F':[('figure3_stimulation','plot.py','--output-dir @out --umap')],
 '3E':[('main3e','plot.py','')],
 '4B-E_S13':[('fig04_acc','plot.py','--out @out')],
 '4GH':[('fig04_ca1','plot.py','--out @out')],
 '5C-F':[('fig05','plot.py','--out @out')],
 'S1EF':[('s01','plot_large_sweep_publishable_panels.py','--rsite-ohm 2000000 --outdir @out')],
 'S4A-check':[('S04A','verify_workbooks.py','')],
 'S4A':[('S04A','render.py','--out @out')],
 'S4B':[('S04B','render.py','--out @out')],
 'S5':[('S05','render.py','--out @out')],
 'S6':[('S06','render.py','--out @out')],
 'S8':[('S08','render.py','--out @out')],
 'S8-check':[('S08','verify_spike_replay.py','--out @out')],
 'S9':[('S09','render.py','--out @out')],
 'S10A':[('S10A','render.py','--out @out')],
 'S10B':[('S10B','render.py','--out @out')],
 'chronic-check':[('chronic_spikes','verify_chronic_spikes.py','--out @out')],
 'S11-S12':[('S11_S12','render.py','--out @out')],
 'S14B':[('s14','plot.py','')],
}
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('workflow',nargs='?',default='4GH',choices=list(RUN))
 parser.add_argument('--list',action='store_true')
 args=parser.parse_args()
 if args.list:
  print('\n'.join(f'{k:12} '+', '.join(f'{d}/{s}' for d,s,a in specs) for k,specs in RUN.items()));return
 env={**os.environ,'MPLBACKEND':'Agg','PYTHONDONTWRITEBYTECODE':'1'}
 for d,s,a in RUN[args.workflow]:
  wd=ROOT/'figures'/d
  out=ROOT/'outputs'/args.workflow
  argv=[sys.executable,str(wd/s),*[str(out) if v=='@out' else v for v in a.split()]]
  subprocess.run(argv,check=True,cwd=wd,env=env)
 print('Output:',out)
if __name__=='__main__':main()


