import torch
from torch import nn
import seaborn
import sys
import re
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier

TRAIN_DATA = 'train.csv'
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
        "values": ["Payday Loan", "Debt Consolidation Loan",  "Personal Loan", "Auto Loan", "Credit-Builder Loan", "Student Loan", "Home Equity Loan", "Mortgage Loan", "Not Specified", ],
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
        "values": ["Bad", "Standard", "Good"],
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
        "values": ["No", "NM","Yes"],
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
        "values": ["Low_spent_Small_value_payments", "Low_spent_Medium_value_payments", "Low_spent_Large_value_payments", "High_spent_Small_value_payments", "High_spent_Medium_value_payments", "High_spent_Large_value_payments"],
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
        "type": "list",
        "regex": None,
        "values": ["Poor", "Standard", "Good"],
        "cleaning": "Categorical encoding via bag-of-words or label mapping.",
        "notes": "Indicates categorization of credit score."
    }
}


def dataCreation(filePath: str) -> pd.DataFrame:
    """
    Load and preprocess credit data from a CSV file.
    
    Constructs the absolute path to the data file relative to the script location,
    loads only the columns defined in COLUMNS configuration, and removes any rows
    containing null values.
    
    Args:
        filePath: Relative path to the CSV file within the data directory
        
    Returns:
        pd.DataFrame: Cleaned dataframe with specified columns and no null values
        
    Note:
        Expected dataset size: approximately 100,000 rows
    """
    # Construct absolute path to data file relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    data_path = os.path.join(data_dir, filePath)
    
    # Load only relevant columns from CSV
    # Note: nrows parameter can be uncommented to limit dataset size for testing
    df = pd.read_csv(data_path, usecols=list(COLUMNS.keys()), encoding="utf-8")
    
    # Remove rows with any null values
    df = df.dropna()
    
    return df


def filterData(df: pd.DataFrame) -> tuple[list, list]:
    """
    Validate and transform dataframe rows into feature and target arrays.
    
    Processes each row through column-specific validation and transformation rules.
    Rows that pass all validations are split into features (X) and targets (Y)
    based on a predefined credit score index position.
    
    Args:
        df: Input dataframe with raw credit data
        
    Returns:
        tuple: (XFiltered, YFiltered) where:
            - XFiltered: List of feature vectors (all columns before credit score)
            - YFiltered: List of target vectors (credit score and following columns)
            
    Note:
        - Prints progress every 1000 rows
        - Prints summary statistics upon completion
        - Credit score index is calculated as (total_columns - 3)
    """
    XFiltered = []
    YFiltered = []

    # Process each row in the dataframe
    for x in df.index:
        row_passed = True
        filtered_row = []

        # Validate and transform each column value
        for col_name in COLUMNS.keys():
            val = validateColumnValue(col_name, df.loc[x, col_name])
            
            # If validation fails, skip this entire row
            if val is False:
                row_passed = False
                break
            # Handle multi-value columns (e.g., one-hot encoded or binary encoded)
            elif isinstance(val, list):
                filtered_row.extend(val)
            # Handle single-value columns
            else:
                filtered_row.append(val)

        # Split validated row into features and targets
        if row_passed:
            # Credit score and subsequent columns are targets (last 3 elements)
            credit_score_index = len(filtered_row) - 3
            XFiltered.append(filtered_row[:credit_score_index])
            YFiltered.append(filtered_row[credit_score_index:])
        
        # Progress logging every 1000 rows
        if x % 1000 == 0 and x > 0:
            print(f"Processed {x} rows, {len(XFiltered)} passed validation...")
    
    # Print final summary statistics
    print(f"Filtered data size: {len(XFiltered)} rows, "
          f"{len(XFiltered[0]) if XFiltered else 0} columns, "
          f"first entry: {XFiltered[0] if XFiltered else 'N/A'}, "
          f"outputs: {YFiltered[0] if YFiltered else 'N/A'}")

    return XFiltered, YFiltered


def validateColumnValue(col_name: str, value):
    """
    Validate and transform a single column value according to predefined rules.
    
    Handles multiple data types with specific validation and transformation logic:
    - Numeric (float/int): Regex extraction and range validation
    - String: Categorical value lookup and index encoding
    - List: Comma-separated parsing and binary encoding
    - Time: Duration parsing (years and months)
    - Output: One-hot encoding for target variables
    
    Args:
        col_name: Name of the column being validated
        value: Raw value from the dataframe
        
    Returns:
        Transformed value on success:
            - Numeric types: float value
            - String types: integer index (position in valid values list)
            - List types: integer representing binary cumulation
            - Time types: float (total years with fractional months)
            - Output types: list of binary indicators (one-hot encoded)
        False on validation failure
        
    Note:
        Prints debug messages when validation fails
    """
    # Verify column exists in configuration
    if col_name not in COLUMNS.keys():
        print(f"Failed validation - unknown column: {col_name}, value: {value}")
        return False

    col_info = COLUMNS[col_name]
    
    # ===== Numeric Type Handling (float/int) =====
    if col_info.get("type") in ["float", "int"]:
        min_val, max_val = col_info["valid_range"]
        try:
            # Extract numeric value using regex pattern
            match = re.findall(col_info["regex"], str(value).strip())

            if len(match):
                num = float(match[0])
            else:
                # Log regex match failure
                print(f"Failed regex validation: {col_name}, value: {value}, "
                      f"regex: {col_info['regex']}, stripped: {str(value).strip()}")
                return False

            # Validate against range constraints (None means unbounded)
            if ((min_val is None or num >= min_val) and
                (max_val is None or num <= max_val)):
                return num

        except ValueError:
            # Silently fail and fall through to final failure case
            pass

    # ===== Categorical String Type Handling =====
    elif col_info.get("type") == "str":
        if col_info.get("values") and value in col_info["values"]:
            # Return index position for categorical encoding
            return col_info["values"].index(value)

    # ===== List Type Handling (comma-separated values) =====
    elif col_info.get("type") == "list":
        # Parse comma-separated string into list
        list_items = [item.strip() for item in str(value).split(',')]

        # Clean "and" prefix from last item
        if list_items:
            list_items[-1] = list_items[-1].removeprefix("and ").strip()

        # one hot assignment for every possible value
        is_item_in_list_present = []
        for item in col_info["values"]:
            if item in list_items:
                is_item_in_list_present.append(1)
            else:
                is_item_in_list_present.append(0)
        return is_item_in_list_present

    # ===== Time Duration Type Handling =====
    elif col_info.get("type") == "time":
        # Extract years and months from time duration string
        match = re.findall(col_info["regex"], str(value).strip())
        if match and len(match) >= 1:
            years = int(match[0])
            months = int(match[1]) if len(match) > 1 else 0
            
            # Convert to total years (months as fraction)
            total_years = years + months / 12.0
            
            if total_years >= 0:
                return total_years
    
    # All validation attempts failed
    print(f"Failed validation - all tests: {col_name}, value: {value}")
    return False


def normalizeDataTrain(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply min-max normalization to training data and store scaling parameters.
    
    Normalizes features to [0, 1] range using per-feature min and max values.
    These parameters should be saved and reused for normalizing validation/test data.
    
    Args:
        X: Training tensor of shape (n_samples, n_features)
        
    Returns:
        tuple: (X_scaled, X_min, X_max) where:
            - X_scaled: Normalized tensor with values in [0, 1]
            - X_min: Per-feature minimum values (for later use)
            - X_max: Per-feature maximum values (for later use)
            
    Note:
        Features with zero range are assigned denominator of 1e-9 to avoid division by zero
    """
    # Compute per-feature min and max
    X_min = X.min(dim=0)[0]
    X_max = X.max(dim=0)[0]
    
    # Compute denominator with zero-range protection
    denom = X_max - X_min
    denom[denom == 0] = 1e-9
    
    # Apply min-max scaling
    X_scaled = (X - X_min) / denom
    
    return X_scaled, X_min, X_max


def normalizeDataWithMinMax(X: torch.Tensor, X_min: torch.Tensor, X_max: torch.Tensor) -> torch.Tensor:
    """
    Normalize data using pre-computed min and max values from training set.
    
    Apply the same min-max scaling transformation used on training data to
    validation or test data. This ensures consistent feature scaling across datasets.
    
    Args:
        X: Tensor to normalize of shape (n_samples, n_features)
        X_min: Per-feature minimum values from training data
        X_max: Per-feature maximum values from training data
        
    Returns:
        torch.Tensor: Normalized tensor with same shape as input
        
    Note:
        Features with zero range in training data are assigned denominator of 1e-9
    """
    # Compute denominator with zero-range protection
    denom = X_max - X_min
    denom[denom == 0] = 1e-9
    
    # Apply min-max scaling using training parameters
    X_scaled = (X - X_min) / denom
    
    return X_scaled

if __name__ == "__main__":
    # Configure print options for debugging (show full arrays)
    np.set_printoptions(threshold=sys.maxsize)
    torch.set_printoptions(edgeitems=7)
    
    # Load raw data from CSV file
    df = dataCreation(TRAIN_DATA)
    
    # Validate and transform data according to column-specific rules
    XFiltered_unorm, YFiltered_unorm = filterData(df)
    X_arr = np.array(XFiltered_unorm, dtype=np.float32)
    Y_arr = np.array(YFiltered_unorm, dtype=np.float32)

    # Count samples in each credit rating class (one-hot encoded: [1,0,0]=Good, [0,1,0]=Standard, [0,0,1]=Poor)
    num_poor = YFiltered_unorm.count([1, 0, 0])
    num_standard = YFiltered_unorm.count([0, 1, 0])
    num_good = YFiltered_unorm.count([0, 0, 1])
    print("Class distribution - Good:", num_good, "Standard:", num_standard, "Poor:", num_poor)

    Y_indices = np.argmax(Y_arr, axis=1)

    # # Split data into training (90%) and validation (10%) sets with stratification
    # X_train, X_val, y_train, y_val = train_test_split(
    #     XFiltered, YFiltered, test_size=0.1, stratify=YFiltered, random_state=42
    # )



    K = 10
    kf = KFold(n_splits=K, shuffle=True, random_state=42)

    fold_results = []

    fold_number = 1
    print(f"\nStarting {K}-fold cross-validation...\n")
    # Train the Random Forest model on training data
    for train_index, val_index in kf.split(X_arr):
        print(f"\n===== Fold {fold_number} / {K} =====")
    
        # Initialize Random Forest with 5000 trees and balanced class weights
        model = RandomForestClassifier(
            bootstrap=True,
            max_depth=None,
            min_samples_leaf=1,
            min_samples_split=2,
            n_estimators=5000,
            class_weight="balanced",  # Handles class imbalance automatically
            n_jobs=-1,                # Use all CPU cores for parallel training
            random_state=42
        )

        # Split
        X_train_unorm = torch.tensor(X_arr[train_index], dtype=torch.float32)
        Y_train = torch.tensor(Y_indices[train_index], dtype=torch.long)

        X_val_unorm = torch.tensor(X_arr[val_index], dtype=torch.float32)
        Y_val = torch.tensor(Y_indices[val_index], dtype=torch.long)
        # Normalize using training fold statistics
        X_train_norm, train_min, train_max = normalizeDataTrain(X_train_unorm)
        X_val_norm = normalizeDataWithMinMax(X_val_unorm, train_min, train_max)

        model.fit(X_train_norm, Y_train)
        
        # Generate predictions on validation set
        preds = model.predict(X_val_norm)
        
        # Convert one-hot encoded labels to class indices for evaluation
        y_val_idx = np.array(Y_val.numpy())
        preds_idx = np.array(preds)

        cm = confusion_matrix(Y_val.numpy(), preds_idx)
        fold_accuracy = (cm[0,0] + cm[1,1] + cm[2,2]) / cm.sum()
        print(f"Fold {fold_number} accuracy: {fold_accuracy:.4f}")
        print(f"Confusion matrix: \n{cm}")

        fold_results.append({
            "fold": fold_number,
            "confusion_matrix": cm,
            "accuracy": fold_accuracy
        })
        
        fold_number += 1


    # Plot confusion matrix
    total_cm = sum(res["confusion_matrix"] for res in fold_results)
    plt.figure(figsize=(8, 6))
    seaborn.heatmap(total_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Poor', 'Standard', 'Good'],
                yticklabels=['Poor', 'Standard', 'Good'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix for Random Forest')
    plt.show()

    # Print the confusion matrix
    print("Confusion Matrix:")
    print(cm)

    # Optional: Calculate accuracy per class
    print("\nPer-class accuracy:")
    for i, label in enumerate(['Poor', 'Standard', 'Good']):
        accuracy = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"{label}: {accuracy:.2%}")

    # Extract and display top 10 most important features
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    print("\nTOP FEATURE IMPORTANCES:")
    for idx in sorted_idx[:10]:
        print(f"Feature {idx}: importance={importances[idx]:.4f}")

    # Calculate and display final validation accuracy
    print(f"\nFinal Accuracy: {sum(res["accuracy"] for res in fold_results)*100:.2f}%")