"""Render Figures 5C–F from figure-level templates, events, drift and unit statistics."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import bubble


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out', type=Path, required=True)
    out = cli.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    data = Path(__file__).resolve().parent
    with np.load(data/'map.npz') as source:
        assert source['approved'].sum() == 100
    bubble.plot_bubble(data, out/'Fig05c.png', anchors=bubble.DEFAULT_ANCHORS,
                       solid_contacts=True, panel_contact_dots=True,
                       panel_bubbles=True, figure5_panel_c_style=True)
    # The historical motion layout configures Matplotlib at module import.
    import motion
    import semantic
    motion.apply_figure6_style()
    motion.render_composite(data, out/'Fig05de')
    table = pd.read_csv(data/'per_unit_semantic.csv')
    for (probe, phase), unit in table.groupby(['probe','phase']):
        denominator = unit['count_matrix_units'].unique()
        if len(denominator) != 1 or unit['neuron'].duplicated().any():
            raise ValueError('Inconsistent per-unit semantic source')
        semantic.COUNTS[probe-1,phase-1] = (unit['p_val_ll_diff_xshuf'] < 0.05).sum()
        semantic.TOTALS[probe-1,phase-1] = denominator[0]
    np.testing.assert_array_equal(semantic.COUNTS, [[12,21],[20,21]])
    np.testing.assert_array_equal(semantic.TOTALS, [[110,109],[102,100]])
    semantic.apply_figure6_style()
    stats = semantic.calculate_statistics()
    semantic.render_panel(out/'Fig05f', mode='points', statistics=stats,
                          figsize=semantic.WIDE_FIGSIZE, legend_side=True, compact=True)


if __name__ == '__main__':
    main()
