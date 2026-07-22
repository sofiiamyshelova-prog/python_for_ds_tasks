import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def split_data(raw_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Split the dataset into training and validation sets.
    """

    input_cols = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]

    X = raw_df[input_cols]
    y = raw_df["Exited"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    result = {
        "train_inputs": X_train.copy(),
        "train_targets": y_train.copy(),
        "val_inputs": X_val.copy(),
        "val_targets": y_val.copy(),
        "input_cols": input_cols,
    }

    return result


def scale_numeric_features(data: Dict[str, Any]) -> StandardScaler:
    """
    Scale numerical features using StandardScaler.
    """

    numeric_cols = data["train_inputs"].select_dtypes(include="number").columns

    scaler = StandardScaler().fit(data["train_inputs"][numeric_cols])

    for split in ["train", "val"]:
        data[f"{split}_inputs"][numeric_cols] = scaler.transform(
            data[f"{split}_inputs"][numeric_cols]
        )

    return scaler


def encode_categorical_features(data: Dict[str, Any]) -> OneHotEncoder:
    """
    One-hot encode categorical features.
    """

    categorical_cols = data["train_inputs"].select_dtypes(exclude="number").columns

    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
    ).fit(data["train_inputs"][categorical_cols])

    encoded_cols = encoder.get_feature_names_out(categorical_cols)

    for split in ["train", "val"]:

        encoded = pd.DataFrame(
            encoder.transform(data[f"{split}_inputs"][categorical_cols]),
            columns=encoded_cols,
            index=data[f"{split}_inputs"].index,
        )

        data[f"{split}_inputs"] = pd.concat(
            [
                data[f"{split}_inputs"].drop(columns=categorical_cols),
                encoded,
            ],
            axis=1,
        )

    return encoder


def preprocess_data(
    raw_df: pd.DataFrame,
    scale_numeric: bool = True,
    ) -> Dict[str, Any]:
    """
    Preprocess the Bank Churn dataset.
    """

    data = split_data(raw_df)

    scaler = None

    if scale_numeric:
        scaler = scale_numeric_features(data)

    encoder = encode_categorical_features(data)

    result = {
        "train_X": data["train_inputs"],
        "train_y": data["train_targets"],
        "val_X": data["val_inputs"],
        "val_y": data["val_targets"],
        "input_cols": data["input_cols"],
        "scaler": scaler,
        "encoder": encoder,
    }

    return result


def preprocess_new_data(
    new_df: pd.DataFrame,
    input_cols: list,
    scaler: StandardScaler,
    encoder: OneHotEncoder,
) -> pd.DataFrame:
    """
    Preprocess new data using fitted scaler and encoder.
    """

    inputs = new_df[input_cols].copy()

    numeric_cols = inputs.select_dtypes(include="number").columns
    categorical_cols = inputs.select_dtypes(exclude="number").columns

    if scaler is not None:
        inputs[numeric_cols] = scaler.transform(inputs[numeric_cols])

    encoded = pd.DataFrame(
        encoder.transform(inputs[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=inputs.index,
    )

    inputs = pd.concat(
        [
            inputs.drop(columns=categorical_cols),
            encoded,
        ],
        axis=1,
    )

    return inputs