# Data

The real-data analysis uses the Bike Sharing Dataset from the UCI Machine Learning Repository.

Dataset name:

```text
Bike Sharing Dataset
```

Source:

```text
UCI Machine Learning Repository
```

Original dataset page:

https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

The dataset is associated with the following paper:

```text
Fanaee-T, H. and Gama, J. (2014).
Event labeling combining ensemble detectors and background knowledge.
Progress in Artificial Intelligence, 2(2-3), 113-127.
```

The scripts try to download the ZIP archive automatically from:

```text
https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip
```

If automatic download fails, download the archive manually from the UCI page above and extract it so that the daily data file is available as:

```text
FinalProject/data/Bike-Sharing-Dataset/day.csv
```

The paper uses `day.csv`. The hourly file `hour.csv` is not required for the reported analysis. The raw data file is not included in this project package. The paper uses the daily data and treats the analysis as exploratory nonparametric smoothing, not as a full time-series or count-data model.
