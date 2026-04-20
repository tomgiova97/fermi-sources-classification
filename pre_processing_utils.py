import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from constants import *
from hardness_ratio_calculator import get_source_hardn_ratios

def remove_corrupted_rows(table_catalog):
    table_catalog.remove_rows(DR2_FILE_CORRUPTED_ROWS)  # Built-in Table function
    return table_catalog


def get_associated_sources_dataframe(table_catalog):
    sources_category = []
    sources_data = []

    for i in range(0, len(table_catalog)):
        source_row = table_catalog[i]
        if source_row["CLASS1"] != "     ":  # Excluding unidentified sources
            sources_category.append(source_row["CLASS1"].split()[0])

            source_data = []
            for attribute in SOURCES_ORIGINAL_ATTRIBUTES:
                source_data.append(source_row[attribute])

            source_hardn_ratios = get_source_hardn_ratios(source_row)

            sources_data.append(source_data + source_hardn_ratios)

    df = pd.DataFrame(
        sources_data,
        index=sources_category,
        columns=SOURCES_ORIGINAL_ATTRIBUTES + HARDN_RATIO_ATTRIBUTES,
    )
    print("Number on Nan values= " + str(df.isnull().sum().sum()))
    # df = df.loc[PULS_LABELS + AGN_LABELS]  # Filter by index only pulsars and AGNs
    return df

def get_unassociated_sources_dataframe(table_catalog):
    sources_data = []

    for i in range(0, len(table_catalog)):
        source_row = table_catalog[i]
        if source_row["CLASS1"] == "     ":
            source_data = []
            for attribute in SOURCES_ORIGINAL_ATTRIBUTES:
                source_data.append(source_row[attribute])

            source_hardn_ratios = get_source_hardn_ratios(source_row)

            sources_data.append(source_data + source_hardn_ratios)

    sources_category = [UNASSOCIATED_SOURCES_LABEL]*len(sources_data)
    df = pd.DataFrame(
        sources_data,
        index= sources_category,
        columns=SOURCES_ORIGINAL_ATTRIBUTES + HARDN_RATIO_ATTRIBUTES,
    )
    print("Number on Nan values= " + str(df.isnull().sum().sum()))
    return df


def scale_df_data(df, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df.values)
    else:
        scaled_data = scaler.transform(df.values)
    scaled_df = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)

    return scaled_df, scaler


def get_sources_df_by_lables(df, labels):
    return df.loc[labels]


def get_train_and_test_random_dataframes(
    df, train_split_perc, val_split_perc, random_state=42
):
    # Shuffle the rows randomly
    df_shuffled = df.sample(
        frac=1, random_state=random_state
    )  # random_state for reproducibility

    # Calculate the index to split the DataFrame
    train_split_index = int(train_split_perc * len(df_shuffled))
    val_split_index = train_split_index + int(val_split_perc * len(df_shuffled))

    # Split the DataFrame
    train_df = df_shuffled[:train_split_index]
    val_df = df_shuffled[train_split_index:val_split_index]
    test_df = df_shuffled[val_split_index:]

    return (train_df, val_df, test_df)


def get_network_data_from_df_data(df, random_state=42):
    # 1. Map labels to integers
    mapping = {label: 0 for label in PULS_LABELS}
    mapping.update({label: 1 for label in AGN_LABELS})

    #Mapping to "2" for the "Others" sources
    y_integers = df.index.map(mapping).fillna(2).astype(int)

    if y_integers.isna().any():
        raise Exception("Indices found that don't match provided label lists.")

    # 2. Calculate Class Weights (based on the WHOLE dataset)
    classes = np.unique(y_integers)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_integers
    )
    class_weights_dict = dict(zip(classes, weights))

    # 3. One-Hot Encode the labels (0 -> [1,0], 1 -> [0,1])
    y_one_hot = pd.get_dummies(y_integers).values.astype(int)

    # 4. Extract numeric features as NumPy array
    X = df.select_dtypes(include=[np.number]).values

    # 5. FIRST SPLIT: Separate Test set (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_one_hot, test_size=0.15, random_state=42, stratify=y_integers
    )

    # We need the integer version of y_temp to stratify the second split properly
    y_temp_integers = np.argmax(y_temp, axis=1)

    # 6. SECOND SPLIT: Split the remaining data into Train and Val (15% of temp)
    # This results in: 70% Train, 15% Val, 15% Test
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.176,
        random_state=random_state,
        stratify=y_temp_integers,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, class_weights_dict
