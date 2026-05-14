# Internship Archive

Static GitHub page for browsing two internship work folders:

- `2023_NoeScharer_Internship_DThPh`
- `2024_NoeScharer_Internship_DPNC`

It supports:

- PDF reports packaged as zip downloads
- Python source visualization
- Jupyter notebook cell visualization
- A directory tree that mirrors the source folders

## Structure

```text
archive/
├── 2023_NoeScharer_Internship_DThPh/
│   └── NoeScharer_report_internship_DThPh.zip
└── 2024_NoeScharer_Internship_DPNC/
    ├── NoeScharer_report_internship_DPNC.zip
    ├── CNN_final/
    │   ├── executable_CNN.py
    │   ├── executable_production_MDC.py
    │   ├── executable_production_SCALES.py
    │   └── plot_final.py
    └── MF_final/
        └── MF.ipynb
```

## Run Locally

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## GitHub Pages

Push the repository to GitHub, then enable GitHub Pages from the `main` branch in the repository settings.
