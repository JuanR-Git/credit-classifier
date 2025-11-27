import torch
from torch import nn
import sys
import re
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

TRAIN_DATA = 'train.csv'
NUMBER_OF_DATAPOINTS = 1000
NUMBER_OF_TOTAL_COLUMNS = 21
COLUMNS = {
    "Age": {
        "type": "int",
        "valid_range": (18, 122),
        "regex": r"\d+(?:\.0+)?",
        "cleaning": "Ensure numeric, no extra characters.",
        "notes": "Reject if outside human age range, 18 (assuming youngest age to have a credit score) to 122 (oldest person alive is 122)."
    },

    "Num_Credit_Card": {
        "type": "int",
        "valid_range": (0, 11),
        "regex": r"\d+(?:\.0+)?",
        "cleaning": "Ensure integer only.",
        "notes": "Reject non-integer or real numbers."
    },

    "Num_Bank_Accounts": {
        "type": "int",
        "valid_range": (0, 11),
        "regex": r"\d+(?:\.0+)?",
        "cleaning": "Ensure integer only.",
        "notes": "Reject if outside 0–11 range."
    },

    "Occupation": {
        "type": "str",
        "regex": None,
        "values": ["Scientist", "_______", "Teacher", "Engineer", "Entrepreneur", "Developer", "Lawyer", "Media_Manager", "Doctor", "Journalist", "Manager", "Accountant", "Musician", "Mechanic", "Writer", "Architect"], 
        "cleaning": "Bag-of-words classification. Normalize capitalization and whitespace.",
        "notes": "Use ‘_______’ to mark ‘other’. Assign numeric encoding after NLP preprocessing."
    },

    "Annual_Income": {
        "type": "float",
        "valid_range": (0, None),
        "regex": r"\d+(?:\.\d+)?",
        "cleaning": "Remove underscores and non-numeric symbols before conversion.",
        "notes": "Check if real number."
    },

    "Monthly_Inhand_Salary": {
        "type": "float",
        "valid_range": (0, None),
        "regex": r"\d+(?:\.\d+)?",
        "cleaning": "Remove underscores and non-numeric characters.",
        "notes": "Tentative column—verify its interpretation."
    },

    "Interest_Rate": {
        "type": "float",
        "valid_range": (0, 100),
        "regex": r"\d+(?:\.\d+)?",
        "cleaning": "Ensure percentage format without % sign.",
        "notes": "Reject >100."
    },

    "Num_of_Loan": {
        "type": "int",
        "valid_range": (0, None),
        "regex": r"\d+(?:\.0+)?",
        "cleaning": "Ensure integer only.",
        "notes": "Should match length of 'Type of Loan' list."
    },

    "Type_of_Loan": { # multiple values possible in a list
        "type": "list",
        "regex": None,
        "values": ["Not Specified", "Mortgage Loan", "Home Equity Loan",  "Student Loan", "Credit-Builder Loan", "Auto Loan", "Personal Loan", "Debt Consolidation Loan",  "Payday Loan"],
        "cleaning": "Tokenize categories, possibly split on commas.",
        "notes": "Correlates to 'Number of Loans'. Verify consistency."
    },

    "Delay_from_due_date": {
        "type": "int",
        "valid_range": (None, None),
        "regex": r"\d+(?:\.0+)?",
        "cleaning": "Raw numeric values only.",
        "notes": "May include negatives if early payments are encoded as negative delays. We'll have to analyse that after"
    },

    "Num_of_Delayed_Payment": {
        "type": "int",
        "valid_range": (0, None),
        "regex": r"\d+(?:\.0+)?",
        "cleaning": "Remove blanks and negatives.",
        "notes": "Must be non-negative integer."
    },

    "Changed_Credit_Limit": {
        "type": "float",
        "valid_range": (None, None),
        "regex": r"[+-]?\d+(?:\.\d+)?",
        "cleaning": "Ensure numeric string, use regex validation.",
        "notes": "Could be positive or negative depending on direction of change."
    },

    "Num_Credit_Inquiries": {
        "type": "int",
        "valid_range": (0, 18),
        "regex": r"\d+(?:\.0+)?",
        "cleaning": "Ensure integer only.",
        "notes": "Usually small whole number count."
    },

    "Credit_Mix": {
        "type": "str",
        "regex": None,
        "values": ["Good", "Standard", "Bad"],
        "cleaning": "Categorical encoding via bag-of-words or label mapping.",
        "notes": "Indicates mix of secured/unsecured credit types."
    },

    "Outstanding_Debt": {
        "type": "float",
        "valid_range": (0, None),
        "regex": r"\d+(?:\.\d+)?",
        "cleaning": "Ensure numeric, remove formatting symbols.",
        "notes": "Should be non-negative."
    },

    "Credit_Utilization_Ratio": {
        "type": "float",
        "valid_range": (0, None),
        "regex": r"\d+(?:\.\d+)?",
        "cleaning": "Use the supplied ratio.",
        "notes": "Should be expressed as given ratio."
    },

    "Credit_History_Age": {
        "type": "time",
        "valid_range": (0, None),
        "regex": r"\d+",
        "cleaning": "Ensure numeric representation (years).",
        "notes": "Derived metric; must be non-negative."
    },

    "Payment_of_Min_Amount": {
        "type": "str",
        "valid_range": None,
        "values": ["No", "NM", "Yes"],
        "cleaning": "Ensure Yes, No, or NM (not much).",
        "notes": "Collects Yes, No, or NM (not much) into a small bag of words / Binary one hot encoding."
    },

    "Total_EMI_per_month": {
        "type": "float",
        "valid_range": (None, None),
        "regex": r"\d+(?:\.\d+)?",
        "cleaning": "Ensure numeric, remove underscores/commas.",
        "notes": "Non-negative; may be related to number of loans."
    },

    "Amount_invested_monthly": {
        "type": "float",
        "valid_range": (0, None),
        "regex": r"\d+(?:\.\d+)?",
        "cleaning": "Ensure numeric.",
        "notes": "Non-negative; financial variable."
    },

    "Payment_Behaviour": {
        "type": "str",
        "regex": None,
        "values": ["High_spent_Medium_value_payments", "High_spent_Large_value_payments", "High_spent_Small_value_payments", "Low_spent_Medium_value_payments", "Low_spent_Large_value_payments", "Low_spent_Small_value_payments"],
        "cleaning": "Categorical encoding. NLP or one-hot encoding likely required.",
        "notes": "Represents qualitative behavior descriptors."
    },

    "Monthly_Balance": {
        "type": "float",
        "valid_range": (None, None),
        "regex": r"\d+(?:\.\d+)?",
        "cleaning": "Ensure numeric, remove formatting symbols.",
        "notes": "May be negative if overdrawn."
    },
    "Credit_Score": {
        "type": "output",
        "regex": None,
        "values": ["Good", "Standard", "Poor"],
        "cleaning": "Categorical encoding via bag-of-words or label mapping.",
        "notes": "Indicates categorization of credit score."
    }
}

def dataCreation(filePath: str):
    ##init fp
    ##expecting Z x 100,000 Pandas DS
    # Get the script's directory and build path to data folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    data_path = os.path.join(data_dir, filePath)
    df = pd.read_csv(data_path, usecols=list(COLUMNS.keys()), encoding="utf-8")#, nrows=NUMBER_OF_DATAPOINTS)
    df = df.dropna()  # Filter out rows with NULL values
    return df

def filterData(df: pd.DataFrame):
    XFiltered = []
    YFiltered = []

    # loop through each row in the dataframe
    for x in df.index:
        row_passed = True
        filtered_row = []

        # loop through each column in the dataframe
        for col_name in COLUMNS.keys():
            val = validateColumnValue(col_name, df.loc[x, col_name])
            # If validation fails
            if val is False:
                row_passed = False
                break
            elif isinstance(val, list):
                filtered_row.extend(val)
            else:
                filtered_row.append(val)

        # Append only if all columns passed validation
        if row_passed:
            credit_score_index = len(filtered_row)-3
            XFiltered.append(filtered_row[:credit_score_index])  # Append the entire row
            YFiltered.append(filtered_row[credit_score_index:])
        # Print progress every 1000 rows
        if x % 1000 == 0 and x > 0:
            print(f"Processed {x} rows, {len(XFiltered)} passed validation...")
    print(f"Filtered data size: {len(XFiltered)} rows, {len(XFiltered[0])} columns, first entry: {XFiltered[0] if XFiltered else 'N/A'}, outputs: {YFiltered[0] if YFiltered else 'N/A'}")

    return XFiltered, YFiltered

# fire
def validateColumnValue(col_name, value):
    if col_name not in COLUMNS.keys():
        print("failed val bc of bad column:", col_name, value)
        return False

    col_info = COLUMNS[col_name]
    
    # Numeric type: return value if within range3
    if col_info.get("type") in ["float", "int"]:
        min_val, max_val = col_info["valid_range"]
        try:
            num = None
            match = re.findall(col_info["regex"], str(value).strip())

            # check is the regex invoked
            if len(match):
                num = float(match[0])

            # empty match case
            else:
                print("failed val regex:", col_name, value, col_info["regex"], str(value).strip())
                return False

            if ((min_val is None or num >= min_val) and
                (max_val is None or num <= max_val)):
                # print("successful num val:", num)
                return num  # return numeric value

        # crash fixing
        except ValueError:
            pass

    # String type: return value if valid
    elif col_info.get("type") == "str":
        if col_info.get("values") and value in col_info["values"]:
            return col_info["values"].index(value)

    # List type: return list if all valid
    elif col_info.get("type") == "list":
        
        # convert value into a list
        list_items = [item.strip() for item in str(value).split(',')]

        # if not empty
        if list_items:
            list_items[-1] = list_items[-1].removeprefix("and ").strip()

        binary_cumulation = 0
        for item in col_info["values"]:
            if item in list_items:
                binary_cumulation += 2 ** col_info["values"].index(item)
        return binary_cumulation

    elif col_info.get("type") == "time":
        match = re.findall(col_info["regex"], str(value).strip())
        if match and len(match) >= 1:
            years = int(match[0])
            months = int(match[1]) if len(match) > 1 else 0
            total_years = years + months / 12.0
            if total_years >= 0:
                return total_years

    elif col_info.get("type") == "output":
        is_item_in_list_present = []
        for item_in_col in col_info["values"]:
            if item_in_col == value:
                is_item_in_list_present.append(1)
            else:
                is_item_in_list_present.append(0)
        return is_item_in_list_present
    
    print("failed val all tests:", col_name, value)
    return False

def normalizeData(X, Y):
    X_array = np.array(X, dtype=np.float32)

    # Normalize features in X using Min-Max scaling
    X_min = X_array.min(axis=0)
    X_max = X_array.max(axis=0)
    X_normalized = (X_array - X_min) / (X_max - X_min + 1e-9)  # Adding a small constant to avoid division by zero

    return X_normalized, Y

# def evaluate_classification_metrics(model, X_val, Y_val, batch_size=64):
#     model.eval()
#     all_preds = []
#     all_labels = []
#     val_dataset = TensorDataset(X_val, Y_val)
#     dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

#     with torch.no_grad():
#         for X, Y in dataloader:
#             logits = model(X)                    # raw scores
#             preds = logits.argmax(dim=1)         # predicted class index
#             labels = Y                           # ground truth indices

#             all_preds.append(preds.cpu())
#             all_labels.append(labels.cpu())

#     all_preds = torch.cat(all_preds).numpy()
#     all_labels = torch.cat(all_labels).numpy()

#     # Confusion Matrix
#     cm = confusion_matrix(all_labels, all_preds)

#     # Classification Report
#     report = classification_report(
#         all_labels, 
#         all_preds, 
#         target_names=["Good", "Standard", "Poor"]
#     )

#     return cm, report


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    torch.set_printoptions(edgeitems=7)
    df = dataCreation(TRAIN_DATA)
    XFiltered_unorm, YFiltered_unorm = filterData(df)
    XFiltered, YFiltered = normalizeData(XFiltered_unorm, YFiltered_unorm)

    num_good = YFiltered.count([1,0,0])
    num_standard = YFiltered.count([0,1,0])
    num_poor = YFiltered.count([0,0,1])
    print("Class distribution - Good:", num_good, "Standard:", num_standard, "Poor:", num_poor)


    # class_counts = [num_good, num_standard, num_poor]
    # class_weights = 1.0 / class_counts  # inverse frequency
    # class_weights = class_weights / class_weights.sum()

    X_train, X_val, y_train, y_val = train_test_split(
        XFiltered, YFiltered, test_size=0.2, stratify=YFiltered, random_state=42
    )

    model = RandomForestClassifier(
        class_weight="balanced",  # fixes your class imbalance
        n_jobs=-1,                # use all CPU cores
        random_state=42
    )

    param_grid = {
        'n_estimators': [100, 250, 500, 1000],
        'max_depth': [None, 10, 50, 100],
        'min_samples_split': [5, 10, 20, 50],
        'min_samples_leaf': [2, 5, 10, 20],
        'bootstrap': [True, False]
    }

    gs=GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=3)
    gs = gs.fit(X_train, y_train)
    print("Best parameters found: ", gs.best_params_, "with score: ", gs.best_score_)

    # # -----------------------------------------------------------
    # # 5. TRAIN
    # # -----------------------------------------------------------
    # print("Training Random Forest...")

    # # -----------------------------------------------------------
    # # 6. VALIDATION PREDICTIONS
    # # -----------------------------------------------------------
    # preds = model.predict(X_val)
    # # -----------------------------------------------------------
    # # 7. METRICS
    # # -----------------------------------------------------------
    # print("\nCONFUSION MATRIX:")
    # y_val_idx = np.argmax(y_val, axis=1)
    # preds_idx = np.argmax(preds, axis=1)
    # print(confusion_matrix(y_val_idx, preds_idx))

    # print("\nCLASSIFICATION REPORT:")
    # print(
    #     classification_report(
    #         y_val,
    #         preds,
    #         target_names=["Good", "Standard", "Poor"]
    #     )
    # )

    # # -----------------------------------------------------------
    # # 8. OPTIONAL: FEATURE IMPORTANCES (USEFUL!)
    # # -----------------------------------------------------------
    # importances = model.feature_importances_
    # sorted_idx = np.argsort(importances)[::-1]

    # correct = 0
    # print("\nTOP FEATURE IMPORTANCES:")
    # for idx in sorted_idx[:10]:
    #     print(f"Feature {idx}: importance={importances[idx]:.4f}")

    # for i, idx in enumerate(y_val_idx):
    #     if idx == preds_idx[i]:
    #         correct += 1
    # accuracy = correct / len(y_val_idx)
    # print(f"In the end, Validation Accuracy: {accuracy*100:.2f}%")


