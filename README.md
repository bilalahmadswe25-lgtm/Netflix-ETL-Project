# Netflix README

## Dataset

-   **Name:** Netflix Movies and TV Shows
-   **File:** `netflix_titles.csv`
-   **Original size:** 8,807 rows and 12 columns
-   **Source:** https://www.kaggle.com/datasets/shivamb/netflix-shows

## ETL Process

-   **Extract:** Read the original CSV file with Python.
-   **Transform:** Check and handle missing values, check duplicates,
    clean the data, create `content_age` and `is_recent`, apply One-Hot
    Encoding to `type`, and apply Standard Scaling to selected numeric
    columns.
-   **Load:** Save the cleaned and preprocessed data to separate CSV
    files and verify the saved output.

## Output Files

-   `dirty_data.csv`
-   `clean_data.csv`
-   `preprocessed_data.csv`
-   `data_quality_report.csv`

## Run the Code

Keep `netflix_titles.csv` and `etl.py` in the same folder. In the VS
Code terminal, run:

``` bash
python etl.py
```

## Results

The terminal screenshot shows: - Original dataset: **8,807 rows × 12
columns** - Preprocessed dataset: **8,807 rows × 16 columns** - Missing
values after preprocessing: **0** - Duplicate rows after preprocessing:
**0** - Load verification: **Successful**

## Note

`duration_value` represents minutes for movies and seasons for TV shows.
These units are different, so they should not be directly compared.
