import numpy as np
import pandas as pd

# sample edit


# -------------------------------------------------------
# 1. Load all CSV files
# -------------------------------------------------------
def load_csv():
    """
    Load all raw CSV files needed for the project.

    Returns:
        data_patients: patients-level data (age, gender, etc.)
        data_edstays : ED stay information (intime, race, arrival, etc.)
        data_triage  : Triage snapshot (vital signs, pain, chief complaint)
        data_vitals  : Time-series vitals from monitors
        data_diagnosis, data_medrecon, data_pyxis: reserved for later use
    """
    data_patients = pd.read_csv("patients.csv")
    data_edstays = pd.read_csv("edstays.csv")
    data_triage = pd.read_csv("triage.csv")
    data_vitals = pd.read_csv("vitalsign.csv")
    data_diagnosis = pd.read_csv("diagnosis.csv")
    data_medrecon = pd.read_csv("medrecon.csv")
    data_pyxis = pd.read_csv("pyxis.csv")

    return (
        data_patients,
        data_edstays,
        data_triage,
        data_vitals,
        data_diagnosis,
        data_medrecon,
        data_pyxis,
    )


# -------------------------------------------------------
# 2. Select only necessary columns and merge tables
# -------------------------------------------------------
def merge_select_columns(data_patients, data_edstays, data_triage, data_vitals):
    """
    Keep only the columns that are needed for Pre_Level_of_care,
    then merge patients + edstays + triage (+ vitals as backup).

    We aim to build a table with:
    subject_id, stay_id, age, gender, race, intime, arrival_transport,
    temperature, heartrate, resprate, o2sat, sbp, dbp, pain, chiefcomplaint
    """

    # Patients: subject-level demographic and age
    df_pat = data_patients[["subject_id", "gender", "anchor_age"]]

    # ED stays: stay-level info (intime, race, arrival mode)
    df_ed = data_edstays[
        ["subject_id", "stay_id", "intime", "race", "arrival_transport"]
    ]

    # Triage: snapshot vitals and chief complaint
    df_tri = data_triage[
        [
            "subject_id",
            "stay_id",
            "temperature",
            "heartrate",
            "resprate",
            "o2sat",
            "sbp",
            "dbp",
            "pain",
            "chiefcomplaint",
        ]
    ]

    # Vitalsign: backup vitals if triage is missing
    df_vitals = data_vitals[
        [
            "stay_id",
            "temperature",
            "heartrate",
            "resprate",
            "o2sat",
            "sbp",
            "dbp",
            "pain",
        ]
    ]

    # merge1: ED stays + patients
    merge1 = pd.merge(df_ed, df_pat, on="subject_id", how="left")

    # merge2: merge1 + triage (add triage vitals and chief complaint)
    merge2 = pd.merge(
        merge1,
        df_tri,
        on=["subject_id", "stay_id"],
        how="left",
        suffixes=("", "_tri"),
    )

    # merge3: merge2 + vitalsign (vitals as backup if triage is missing)
    merged_df = pd.merge(
        merge2,
        df_vitals,
        on="stay_id",
        how="left",
        suffixes=("", "_v"),
    )

    # Use vitalsign values as backup when triage values are missing
    vitals_cols = [
        "temperature",
        "heartrate",
        "resprate",
        "o2sat",
        "sbp",
        "dbp",
        "pain",
    ]

    for col in vitals_cols:
        # merged_df[col] is triage value
        # merged_df[col + "_v"] is vitalsign value
        merged_df[col] = merged_df[col].fillna(merged_df[col + "_v"])

    # Remove backup columns from vitalsign
    merged_df.drop(columns=[col + "_v" for col in vitals_cols], inplace=True)

    # Rename anchor_age -> age (clearer for modeling)
    merged_df = merged_df.rename(columns={"anchor_age": "age"})

    # Convert intime to datetime
    merged_df["intime"] = pd.to_datetime(merged_df["intime"], errors="coerce")

    # Reorder columns in a clean, logical order
    final_cols = [
        "subject_id",
        "stay_id",
        "age",
        "gender",
        "race",
        "intime",
        "arrival_transport",
        "temperature",
        "heartrate",
        "resprate",
        "o2sat",
        "sbp",
        "dbp",
        "pain",
        "chiefcomplaint",
    ]

    merged_df = merged_df[final_cols]

    return merged_df


# -------------------------------------------------------
# 3. Preprocessing: handle missing values, text cleaning
# -------------------------------------------------------
def preprocess(df):
    """
    Basic preprocessing for PRE_LOC features:
    - Clean chiefcomplaint text
    - Clean pain column (non-numeric to NaN)
    - Fill missing numerical vitals with median
    - Fill missing categorical (gender, race, arrival_transport) with 'UNKNOWN'
    """

    # --- Categorical columns: gender, race, arrival_transport ---
    df["gender"] = df["gender"].fillna("UNKNOWN")

    df["race"] = df["race"].fillna("UNKNOWN")

    df["arrival_transport"] = df["arrival_transport"].fillna("UNKNOWN")

    # --- Chief complaint cleaning ---
    # Convert to string, strip spaces, and upper-case for consistency
    df["chiefcomplaint"] = df["chiefcomplaint"].astype(str).str.strip()

    # Treat some special tokens as unknown
    df["chiefcomplaint"] = df["chiefcomplaint"].replace(
        ["UNKNOWN-CC", "unknown", "Unknown", ""], np.nan
    )

    # Fill remaining missing with 'UNKNOWN'
    df["chiefcomplaint"] = df["chiefcomplaint"].fillna("UNKNOWN")

    # --- Pain cleaning ---
    if "pain" in df.columns:
        # Convert to numeric; non-numeric (e.g. "UA", "UTA", "UNABLE") become NaN
        df["pain"] = pd.to_numeric(df["pain"], errors="coerce")

    # --- Numerical columns: vitals + age ---
    numeric_cols = [
        "age",
        "temperature",
        "heartrate",
        "resprate",
        "o2sat",
        "sbp",
        "dbp",
        "pain",
    ]

    for col in numeric_cols:
        # Ensure numeric type
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Fill missing values with median
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    return df


# -------------------------------------------------------
# 4. Save merged and preprocessed data
# -------------------------------------------------------
def save_csv(df, filename):
    """
    Save the final preprocessed DataFrame as a CSV file.
    """
    df.to_csv(filename, index=False)


# -------------------------------------------------------
# 5. Main
# -------------------------------------------------------
def main():
    # 1: load raw CSVs
    (
        data_patients,
        data_edstays,
        data_triage,
        data_vitals,
        data_diagnosis,
        data_medrecon,
        data_pyxis,
    ) = load_csv()

    # 2: merge and select only necessary columns
    df = merge_select_columns(data_patients, data_edstays, data_triage, data_vitals)

    # 3: basic preprocessing (missing values, cleaning text)
    df = preprocess(df)

    # check 10 rows
    print(df.head(10))

    # outcome: save to CSV
    save_csv(df, "15attributes.csv")


# Run the full preprocessing, save as csv file
main()
