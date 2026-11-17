"""Render the CA1 unit waveform, ISI and place-field panels from the curated unit data."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


DAYS = (90,91,92,97,104)


def place_field(occupancy, spikes, tracking_hz):
    """Use the occupancy normalization and smoothing of kyutils 0.1.101."""
    ratio = np.divide(spikes, occupancy, out=np.zeros(occupancy.shape), where=occupancy>0)
    rates = gaussian_filter(np.rot90(ratio)*tracking_hz, sigma=1)
    return np.where(np.rot90(occupancy)>0,rates,np.nan), np.max(rates)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out', type=Path, required=True)
    out = cli.parse_args().out
    out.mkdir(parents=True,exist_ok=True)
    fig, axes = plt.subplots(3,5, figsize=(7,4),gridspec_kw={'height_ratios':[1.2,.5,1.4]})
    fig.subplots_adjust(wspace=.65,hspace=.28,top=.9,bottom=.10)
    maxima = []
    with np.load(Path(__file__).with_name('unit75.npz')) as data:
        for column,day in enumerate(DAYS):
            wave,isi,place = axes[:,column]
            template = data[f'mean_waveform_uv_{day}']
            n_samples = template.shape[0]
            for contact,(x,y) in enumerate(data['unit_contact_um']):
                wave.plot(np.arange(n_samples)+4*x, template[:,contact]+4*y,color='black',lw=.7)
            wave.set_title(f'Day {day}',fontsize=8)
            wave.axis('off')
            bins = data['isi_edges_ms']
            isi.bar(bins[:-1],data[f'isi_counts_{day}'],np.diff(bins),align='edge',color='black')
            isi.spines[['top','right']].set_visible(False)
            isi.tick_params(labelsize=5)
            if column == 0:
                isi.set_ylabel('Count',fontsize=6)
            if column == 2:
                isi.set_xlabel('Interspike interval (ms)',fontsize=6)
            field,maximum = place_field(data[f'occupancy_frames_{day}'],data[f'spike_histogram_{day}'],data[f'tracking_rate_hz_{day}'])
            maxima.append(float(maximum))
            place.imshow(field,cmap='viridis',interpolation='nearest',vmax=maximum)
            # The original notebook retained the last column index when it set these limits.
            place.set_xlim((1.5,9.5))
            place.set_ylim((.3,50.3))
            place.text(.5,-.06,f'{maximum:.2f} spikes/s',fontsize=6,ha='center',va='top',transform=place.transAxes)
            place.axis('off')
    np.testing.assert_array_equal(np.round(maxima,2),[62.19,70.79,65.59,67.94,59.35])
    for extension in ('png','pdf','svg'):
        fig.savefig(out/f'Fig04gh.{extension}',dpi=300)
    plt.close(fig)
    print('Place field maxima in spikes/s:', ', '.join(f'{value:.5f}' for value in maxima))


if __name__ == '__main__':
    main()
