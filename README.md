# Peptoid-RPA-2.0
Adapted from SCFT Tools Written by D. Zhao

RPA Code Specifically Designed For Polypeptoid Systems

Accounting for Charge Neutrality, and varying degrees of GLU residue deprotonation

## Layout

- `rpa_acid_base/` contains the production RPA package.
- `run_rpa.py` is a convenience launcher; `python -m rpa_acid_base` is the package entry point.
- `input.in` is the default runtime configuration.
- `force_fields/current/` contains the active non-clustered force fields.
- `force_fields/clustered_water/` contains the six-water-bead force fields.
- `force_fields/previous/` and `force_fields/system_offsets/` keep historical/reference force fields.
- `results/rpa/` stores generated RPA text/CSV outputs.
- `results/figures/` stores generated figures.
- `scripts/` contains plotting and utility scripts.
- `styles/` contains reusable Matplotlib style sheets.
- `misc/` contains legacy/reference scripts that are not part of the main run path.
- `data/` stores input sequence/block-length data.

## Input

Edit `input.in` to set:

```text
chain_sequences = AGGGGAGGGGAGGGGAGGGG
max_conc = 0.09
tmah_conc = 0.0238
water_model = single
output_file = OUT.txt
stop_at_init_spinodal = false
```

For multiple sequences, comma-separate the values:

```text
chain_sequences = AGGGGAGGGGAGGGGAGGGG, AGAGAGAGAGAGAGAGAGAG
```

Run with `python -m rpa_acid_base` or `python run_rpa.py`. Use `water_model = six-water` for the clustered-water model.
Single-sequence runs write the detailed stability file for every composition point.
Multi-sequence runs write a spinodal summary with sequence, aromatic fraction, initial spinodal polymer weight fraction, and spinodal type.
Set `output_file` to choose the output filename. Plain filenames are written under `results/rpa/`.
Set `stop_at_init_spinodal = true` to move to the next sequence as soon as the first `0 -> 1` or `0 -> 2` spinodal transition is found after a completed RPA sub-run.

## Plotting

Plotting scripts live in `scripts/`, and the shared Matplotlib style is `styles/academic.mplstyle`.
