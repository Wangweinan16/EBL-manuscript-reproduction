# EBL manuscript figure reproduction

This package follows the November 2026 Neuron figure numbering. Each workflow reads its inputs from `figures/`, writes plots to `outputs/`, and runs locally.

**Start with Figure 4G–H.** From this directory, use Python 3.12:

```text
python -m pip install -r requirements.txt
python reproduce.py 4GH
```

Open the resulting `outputs/4GH/Fig04gh.png` ([reference image](reference/4GH.png)). It shows CA1 waveforms, inter-spike intervals and place fields across five days. The script checks the printed peak rates: **62.19, 70.79, 65.59, 67.94 and 59.35 Hz**.

**Find a figure.** Run `python reproduce.py <workflow>` with a key below. `python reproduce.py --list` prints each underlying script. Folders under `figures/` hold plotting code, figure data and acquisition or sorting inputs; [reference/](reference/) holds selected comparison plots. Diagram and photograph panels have no plotting command.

| November manuscript | Workflow | What it draws |
| --- | --- | --- |
| 1B; 1J–K | `1B`; `1JK` | Interconnect simulation; voltage transients and cyclic voltammetry |
| 1I | MATLAB¹ | Probe footprint versus channel count |
| 2A, C; 2B | `2AC`; `2B` | Example traces, waveforms and layouts; density comparison |
| 2D, F–H; 2E | `2DFGH`; `2E` | Ten-day waveforms, positions and connections; ISI histograms |
| 3A, C, D, G; 3B; 3E; 3F | `3ACDG`; `3B`; `3E`; `3F`² | Stimulation maps and rasters; raw voltage and spike arrows; distance selectivity; UMAP |
| 4B–E and S13; 4G–H | `4B-E_S13`; `4GH` | Rat ACC population and behavior; CA1 place cell |
| 5C–F | `5C-F` | Human waveform map, threshold events, displacement and semantic encoding |
| S1A, S2A–B; S1E–F | MATLAB¹; `S1EF` | Resistance and design models; out-of-plane signal transfer and crosstalk |
| S4A, B; S5; S6 | `S4A`, `S4B`; `S5`; `S6` | Device yield and impedance; waveform/CCG; recording stability |
| S8; S9; S10A, B | `S8`; `S9`; `S10A`, `S10B` | Evoked responses, neighboring stimulation sites and chronic responses |
| S11–S12; S14B | `S11-S12`; `S14B` | Threshold and modulation; motor-cortex heatmaps, days 6–10 |

**Check from source.** `S4A-check` reads the original instrument workbooks. `S8-check` recomputes activation from the sorter and trial inputs. `chronic-check` checks session rasters and modulation against the longitudinal sorted spikes.

¹ In MATLAB run `replot_models` from this directory for S1A/S2. For 1I, run `replot_probe_area3` from `figures/figure1_benchmark/`, then run `python place_figure1i_legend.py` there.

² Run 3F in a separate Python environment with the versions in [requirements-umap.txt](requirements-umap.txt). For native curation replay, use SpikeInterface 0.99.1 with Python 3.10: `python figures/sorting_provenance/restore_curated.py tracking --output-dir outputs/tracking` (or use `rat`). Session-specific sorter settings and channel maps are supplied alongside the firing arrays.

