# Controlling Smoothness in Nonparametric Regression

## Brief description

This project contains the final project package for the course report:

**Controlling Smoothness in Nonparametric Regression: Theory, Simulations, and Real Data Analysis**

The paper studies nonparametric smoothing through the common principle that smoothing parameters control the bias-variance trade-off. It covers KDE, Nadaraya-Watson regression, local polynomial regression, B-spline regression, smoothing splines, simulations, and an exploratory Bike Sharing Dataset analysis.

GitHub repository: https://github.com/YuxiaDing/NonparametricAnalysis-FinalProject

## Folder structure

```text
FinalProject/
  main.tex
  references.bib
  README.md
  requirements.txt
  .gitignore
  code/
  data/
  figures/
  tables/
```

## How to compile the LaTeX paper

The paper uses the course template style.

Recommended Overleaf workflow:

1. Zip the whole `FinalProject/` folder.
2. Upload the folder to Overleaf.
3. Set the main file to `main.tex`.
4. Compile with `pdfLaTeX`.
5. Run BibTeX if Overleaf does not do it automatically.

Local command-line workflow:

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

For the versioned PDF used during revision, compile with:

```text
pdflatex -jobname=main_v6 main.tex
bibtex main_v6
pdflatex -jobname=main_v6 main.tex
pdflatex -jobname=main_v6 main.tex
```

## How to install dependencies

Create and activate a Python environment, then run:

```text
pip install -r requirements.txt
```

The code uses common scientific Python packages: NumPy, pandas, SciPy, Matplotlib, and scikit-learn.

## How to prepare the Bike Sharing Dataset

Dataset name:

```text
Bike Sharing Dataset
```

Source:

```text
UCI Machine Learning Repository
```

Dataset page:

```text
https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
```

The dataset is associated with:

```text
Fanaee-T, H. and Gama, J. (2014).
Event labeling combining ensemble detectors and background knowledge.
Progress in Artificial Intelligence, 2(2-3), 113-127.
```

The script `code/real_data_bike.py` tries to download the ZIP archive automatically from:

```text
https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip
```

If automatic download fails, manually download the Bike Sharing Dataset from:

```text
https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
```

The paper and code use `day.csv`. The hourly file `hour.csv` is not required for the reported analysis. Raw data files are not tracked by this project. If the code cannot download the data automatically, the user must download and place `day.csv` manually as described above.

## How to reproduce figures

From the `FinalProject/` directory, run:

```text
python code/make_all_figures.py
```

Figures are written to:

```text
figures/
```

Most figures are saved as PDF files suitable for LaTeX inclusion.

Main figure-producing scripts:

```text
code/simulate_bandwidth.py
code/simulate_methods.py
code/simulate_noise.py
code/simulate_boundary.py
code/real_data_bike.py
```

## How to reproduce tables

From the `FinalProject/` directory, run:

```text
python code/make_all_tables.py
```

Tables are written to:

```text
tables/
```

Each generated table is saved as both CSV and LaTeX.

Main table-producing scripts:

```text
code/simulate_methods.py
code/simulate_noise.py
code/simulate_boundary.py
code/real_data_bike.py
```

## Random seed and reproducibility notes

The main random seed is set in:

```text
code/config.py
```

The default seed is `20260602`. Simulation scripts use fixed seeds and deterministic grids. Minor numerical differences may occur across SciPy or scikit-learn versions, especially for smoothing spline fits.
