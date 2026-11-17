# Notebooks

**Source of truth is the `.py` file** — written in
[jupytext percent format](https://jupytext.readthedocs.io/en/latest/formats.html#percent-format).
The `.ipynb` is generated and executed so outputs are visible on GitHub.

```bash
make notebooks                       # convert + execute every notebook
tools/build_notebooks.py 03          # just notebook 03
tools/build_notebooks.py --convert   # convert only, skip execution
```

| Notebook | Purpose |
|---|---|
| `01_data_collection` | Reproduce/inspect the USGS ingestion run and manifest |
| `02_data_cleaning` | Cleaning decisions, validation gates, missingness |
| `03_eda` | Magnitude distribution, Gutenberg-Richter b-value, temporal trends, depth, most-active regions |
| `04_geospatial_analysis` | Grid system, spatial activity, hotspots, regional comparison |
| `05_feature_engineering` | Rolling features, targets, class balance, leakage verification |

Do not edit `.ipynb` files by hand — regenerate them.
