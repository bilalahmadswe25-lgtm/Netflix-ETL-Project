# Dataset Name: Netflix Movies and TV Shows
# Dataset Source: Kaggle
# Source URL: https://www.kaggle.com/datasets/shivamb/netflix-shows

# Import pandas to read, clean, and save CSV data.
import pandas as pd

# Import StandardScaler to scale numerical values.
from sklearn.preprocessing import StandardScaler


# ================= 1. EXTRACT =================

# Read the original Netflix dataset from the CSV file.
df = pd.read_csv("netflix_titles.csv")

# Save an unchanged copy of the original dataset.
df.to_csv("dirty_data.csv", index=False)

# Display the first five rows of the original dataset.
print("First Five Rows of Data Set:")
print(df.head())

# Display the total number of rows and columns.
print("Number of Rows and Columns:")
print(df.shape)

# Count missing values in every column.
print("Total Null Value Check:")
print(df.isnull().sum())

# Display the data type of every column.
print("Check Data Types:")
print(df.dtypes)

# Count repeated rows in the original dataset.
print("Check Duplicated Value:")
print(df.duplicated().sum())

# Display all original column names.
print("Column Names:")
print(df.columns)


# Save missing-value counts before cleaning.
# This information will be used in the quality report.
missing_before = df.isnull().sum()


# ================= 2. CLEAN COLUMN NAMES =================

# Remove extra spaces from column names.
df.columns = df.columns.str.strip()

# Display the cleaned column names.
print("Column Names After Cleaning:")
print(df.columns)


# ================= 3. CLEAN TEXT VALUES =================

# Select columns that contain text values.
text_columns = df.select_dtypes(include=["str", "object"]).columns

# Visit each text column one by one.
for col in text_columns:

    # Remove extra spaces from the beginning and end.
    df[col] = df[col].str.strip()

    # Convert empty text values into missing values.
    df[col] = df[col].replace("", pd.NA)


# ================= 4. DIRECTOR =================

# Count missing values in the director column.
print("Missing Values in Director:")
print(df["director"].isnull().sum())

# Replace missing director names with Unknown.
df["director"] = df["director"].fillna("Unknown")


# ================= 5. CAST =================

# Count missing values in the cast column.
print("Missing Values in Cast:")
print(df["cast"].isnull().sum())

# Replace missing cast information with Unknown.
df["cast"] = df["cast"].fillna("Unknown")


# ================= 6. COUNTRY =================

# Count missing values in the country column.
print("Missing Values in Country:")
print(df["country"].isnull().sum())

# Replace missing country information with Unknown.
df["country"] = df["country"].fillna("Unknown")


# ================= 7. DATE ADDED =================

# Count missing values in the date_added column.
print("Missing Values in Date Added:")
print(df["date_added"].isnull().sum())

# Fill missing dates with Unknown.
df["date_added"] = df["date_added"].fillna("Unknown")


# ================= 8. RATING =================

# Count missing values in the rating column.
print("Missing Values in Rating:")
print(df["rating"].isnull().sum())

# Find the most frequent rating value.
print("Most Frequent Rating:")
print(df["rating"].mode()[0])

# Fill missing ratings with the most frequent rating.
df["rating"] = df["rating"].fillna(df["rating"].mode()[0])


# ================= 9. LISTED IN =================

# Count missing values in the listed_in column.
print("Missing Values in Listed In:")
print(df["listed_in"].isnull().sum())

# Fill missing genre information with Unknown.
df["listed_in"] = df["listed_in"].fillna("Unknown")


# ================= 10. DESCRIPTION =================

# Fill missing descriptions with Unknown.
df["description"] = df["description"].fillna("Unknown")


# ================= 11. DURATION =================

# Count missing values in the duration column.
print("Missing Values in Duration:")
print(df["duration"].isnull().sum())

# Fill missing original duration text with Unknown.
df["duration"] = df["duration"].fillna("Unknown")

# Extract the numerical part of duration.
# Example: "90 min" becomes 90 and "2 Seasons" becomes 2.
df["duration_value"] = pd.to_numeric(
    df["duration"].str.extract(r"(\d+)")[0],
    errors="coerce"
)

# Fill missing numerical durations with their median.
df["duration_value"] = df["duration_value"].fillna(
    df["duration_value"].median()
)

# Display the extracted duration values.
print("Duration Values After Cleaning:")
print(df["duration_value"].head())


# ================= 12. TYPE =================

# Fill missing content types with the most frequent type.
df["type"] = df["type"].fillna(df["type"].mode()[0])

# Make the content type format consistent.
df["type"] = df["type"].str.title()


# ================= 13. RELEASE YEAR =================

# Convert release_year into a numeric column.
# Invalid text values are changed into missing values.
df["release_year"] = pd.to_numeric(
    df["release_year"], errors="coerce"
)

# Display missing release years.
print("Missing Values in Release Year:")
print(df["release_year"].isnull().sum())


# ================= 14. REMOVE DUPLICATES =================

# Remove repeated rows from the dataset.
df = df.drop_duplicates()

# Check duplicates after cleaning.
print("Duplicates After Cleaning:")
print(df.duplicated().sum())


# ================= 15. FILTER RECORDS =================

# Remove records that have no title or release year.
df = df.dropna(subset=["title", "release_year"])

# Keep records with release years from 1900 to 2026.
df = df[
    (df["release_year"] >= 1900) &
    (df["release_year"] <= 2026)
].copy()

# Convert release_year from numeric float to integer.
df["release_year"] = df["release_year"].astype(int)

# Display rows and columns after filtering.
print("Rows and Columns After Filtering:")
print(df.shape)


# ================= 16. CREATE DERIVED COLUMNS =================

# Calculate content age using 2026 as the reference year.
df["content_age"] = 2026 - df["release_year"]

# Create a flag: 1 means released in 2020 or later.
# 0 means released before 2020.
df["is_recent"] = (df["release_year"] >= 2020).astype(int)

# Display the two new calculated columns.
print("New Columns:")
print(df[["content_age", "is_recent"]].head())


# ================= 17. SAVE CLEAN DATA =================

# Save cleaned data before One-Hot Encoding and Scaling.
df.to_csv("clean_data.csv", index=False)

# Confirm that the cleaned file was created.
print("Clean Data Saved Successfully")


# ================= 18. DATA QUALITY REPORT =================

# Count missing values after the cleaning steps.
missing_after = df.isnull().sum()

# Combine before-and-after missing counts in one table.
quality_report = pd.DataFrame({
    "Missing Before Cleaning": missing_before,
    "Missing After Cleaning": missing_after
})

# Save the quality report as a separate CSV file.
quality_report.to_csv("data_quality_report.csv")

# Display the quality report.
print("Data Quality Report:")
print(quality_report)

# Confirm that the report was saved.
print("Data Quality Report Saved Successfully")


# ================= 19. ONE-HOT ENCODING =================

# Display content types before encoding.
print("Before Encoding:")
print(df["type"].head())

df = pd.get_dummies(df, columns=["type"], dtype=int)

print("After Encoding - First Five Rows:")
print(df.filter(like="type_").head())

print("Encoded Column Names:")
print(df.filter(like="type_").columns)




# ================= 20. SCALING =================

# Select the numerical columns that need scaling.
scale_columns = ["release_year", "duration_value"]

# Create the StandardScaler object.
standard_scaler = StandardScaler()

# Calculate scaled values for the selected columns.
df[scale_columns] = standard_scaler.fit_transform(
    df[scale_columns]
)

# Display the scaled numerical values.
print("Values After Standard Scaling:")
print(df[scale_columns].head())


# ================= 21. LOAD FINAL DATA =================

# Save the fully preprocessed data into a separate CSV file.
df.to_csv("preprocessed_data.csv", index=False)

# Confirm that the final file has been saved.
print("Final Preprocessed Data Saved Successfully")


# ================= 22. DISPLAY FINAL DATA =================

# Display the first five rows after all transformations.
print("\nFirst Five Rows After All Preprocessing:")
print(df.head())

# Display final row and column counts.
print("Final Rows and Columns:")
print(df.shape)

# Display all final column names.
print("Final Column Names:")
print(df.columns)

# Count missing values in each final column.
print("Final Missing Values:")
print(df.isnull().sum())

# Calculate the total number of missing cells.
print("Total Missing Values:")
print(df.isnull().sum().sum())

# Display the final data types.
print("Final Data Types:")
print(df.dtypes)

# Count any remaining duplicate rows.
print("Final Duplicate Records:")
print(df.duplicated().sum())


# ================= 23. VERIFY LOADING =================

# Read the saved final CSV file again.
check = pd.read_csv("preprocessed_data.csv")

# Compare saved data with the data loaded from the file.
print("\nLOAD VERIFICATION")

# Display the number of rows saved and loaded.
print("Rows Saved:", len(df))
print("Rows Loaded:", len(check))

# Display the number of columns saved and loaded.
print("Columns Saved:", len(df.columns))
print("Columns Loaded:", len(check.columns))

# Check whether row and column counts match.
if df.shape == check.shape:
    print("Loading Verification Successful")
else:
    print("Please Check the Output")

# Display the final completion message.
print("ETL Pipeline Completed Successfully")
