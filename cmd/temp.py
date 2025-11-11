import re
import pandas as pd
import numpy as np 
import sys
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

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
        "values": ["Auto Loan", "Credit-Builder Loan", "Debt Consolidation Loan", "Home Equity Loan", "Mortgage Loan", "Not Specified", "Payday Loan", "Personal Loan", "Student Loan"],
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

    "Credit_History_Age": { # need a better handling for this regex because its a string that returns 2 numbers (Year & month)
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
        "type": "str",
        "regex": None,
        "values": ["Good", "Standard", "Poor"],
        "cleaning": "Categorical encoding via bag-of-words or label mapping.",
        "notes": "Indicates categorization of credit score."
    }
}

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

        if all(item in col_info["values"] for item in list_items):
            return [col_info["values"].index(item) for item in list_items]  # return the index corresponding to item in col value
            
    elif col_info.get("type") == "time":
        match = re.findall(col_info["regex"], str(value).strip())
        if match and len(match) >= 1:
            years = int(match[0])
            months = int(match[1]) if len(match) > 1 else 0
            total_years = years + months / 12.0
            if total_years >= 0:
                return total_years
    
    print("failed val all tests:", col_name, value)
    return False


def dataCreation(filePath: str):
    ##init fp
    ##expecting Z x 100,000 Pandas DS
    # Get the script's directory and build path to data folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    data_path = os.path.join(data_dir, TRAIN_DATA)
    df = pd.read_csv(data_path, usecols=list(COLUMNS.keys()), encoding="utf-8")
    df = df.dropna()  # Filter out rows with NULL values
    return df

# future improvement: make the search for validating faster using search algo 
def filterData(df: pd.DataFrame):
    XFiltered = pd.DataFrame(columns=COLUMNS.keys())
    
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
            else:
                filtered_row.append(val)

        # Append only if all columns passed validation
        if row_passed:
            XFiltered.loc[x] = filtered_row
        # Print progress every 1000 rows
        if x % 1000 == 0 and x > 0:
            print(f"Processed {x} rows, {len(XFiltered)} passed validation...")
    print(f"Filtered data size: {XFiltered.shape[0]} rows, {XFiltered.shape[1]} columns")

    # separate the credit score column from the rest of the dataframe
    # Todo: change the logic to work with test data with no Credit_Score column
    YFiltered = XFiltered["Credit_Score"].copy()
    XFiltered = XFiltered.drop(columns=["Credit_Score"])
    return XFiltered, YFiltered

class mySvm():
    def __init__(self,):
        self.scaler = StandardScaler()
        self.model = None
        self.X_scaled = None
        self.y_labels = None
        self.feature_names = None
        self.feature_breakdown = {}
        self.X_encoded = None

    # convertToBinaryFeatures() function: Convert all relevant features to binary/one-hot encoded features, fire?
    def convertToBinaryFeatures(self, X):
        X_binary = pd.DataFrame(index=X.index)
        
        # loop through each column (feature) in the dataframe
        for col_name in X.columns:
            col_info = COLUMNS[col_name]
            col_data = X[col_name]
            
            # Handle string types, convert to one-hot encoding
            # (TODO: implement ________ as other in column name)
            if col_info.get("type") == "str":
                values = col_info.get("values", [])
                for idx, value in enumerate(values):
                    X_binary[f"{col_name}_{value}"] = (col_data == idx).astype(int)

            # Handle list types (Type_of_Loan), convert to one-hot encoding
            elif col_info.get("type") == "list":
                values = col_info.get("values", [])
                loan_columns = {
                    loan_type: pd.Series(0, index=X.index)
                    for loan_type in values
                }

                for row_idx, entry in col_data.items():
                    if isinstance(entry, list):
                        for loan_idx in entry:
                            if 0 <= loan_idx < len(values):
                                loan_type = values[loan_idx]
                                loan_columns[loan_type].loc[row_idx] = 1
                    elif isinstance(entry, (int, float, np.integer)):
                        loan_idx = int(entry)
                        if 0 <= loan_idx < len(values):
                            loan_type = values[loan_idx]
                            loan_columns[loan_type].loc[row_idx] = 1

                for loan_type, series in loan_columns.items():
                    X_binary[f"{col_name}_{loan_type}"] = series

            # Handle numeric types (int, float, time) - keep as numeric
            elif col_info.get("type") in ["int", "float", "time"]:
                # using coerce to as safety for values that are not numeric to be filtered out with mask in preprocess func
                X_binary[col_name] = pd.to_numeric(col_data, errors='coerce')
        
        self.X_encoded = X_binary
        return X_binary

        

    # preprocess() function:
    def preprocess(self, X, y):

        # Convert categorical/list features to binary form
        X_encoded = self.X_encoded

        # Capture feature breakdown for reporting
        breakdown = {}
        for feature_name in COLUMNS.keys():
            if feature_name == "Credit_Score":
                continue
            # find all columns that start with the feature name
            matching_cols = [col for col in X_encoded.columns if col.startswith(f"{feature_name}_")]
            if matching_cols:
                # count the number of columns within same feature for reporting
                breakdown[feature_name] = len(matching_cols)
        breakdown["numeric_features"] = len([col for col in X_encoded.columns if col in X.columns])
        self.feature_breakdown = breakdown

        # Convert to numeric and drop rows with invalid feature or target values in one pass
        X_numeric = X_encoded.apply(pd.to_numeric, errors='coerce')
        valid_rows = X_numeric.notnull().all(axis=1) & y.notnull()
        X_processed = X_numeric[valid_rows].copy()
        y_processed = y[valid_rows].copy()

        # Normalize the data
        X_scaled = self.scaler.fit_transform(X_processed)
        X_scaled = pd.DataFrame(X_scaled, columns=X_processed.columns, index=X_processed.index)
        
        # Convert target to numeric labels (Good=0, Standard=1, Poor=2)
        if y_processed.dtype == 'object' or y_processed.dtype.name == 'category':
            label_map = {"Good": 0, "Standard": 1, "Poor": 2}
            y_labels = y_processed.map(label_map)
        else:
            # Already numeric from validation (indices: Good=0, Standard=1, Poor=2)
            y_labels = pd.to_numeric(y_processed, errors='coerce')
        
        # Remove any rows where label mapping failed
        valid_labels = y_labels.notnull()
        X_scaled = X_scaled[valid_labels]
        y_labels = y_labels[valid_labels]
        
        self.X_scaled = X_scaled
        self.y_labels = y_labels
        self.feature_names = X_scaled.columns.tolist()
        
       
        return X_scaled, y_labels

    # cross_validation() function splits the data into train and test splits,
    def cross_validation(self, X, y, k=10):

        kf = KFold(n_splits=k, shuffle=True, random_state=67)
        tss_scores = []
        accuracy_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # call training function
            self.training(X_train, y_train)
            
            # Predict
            y_pred = self.model.predict(X_test)
            
            # call tss function
            tss = self.tss(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)
            tss_scores.append(tss)
            accuracy_scores.append(acc)
        
        
        return np.mean(tss_scores), tss_scores, np.mean(accuracy_scores), accuracy_scores

    #training() function trains a SVM classification model on input features and corresponding target
    def training(self, X_train, y_train):

        #for now lets use linear kernel standard scaler, we will likely change this later
        self.model = SVC(kernel='linear', random_state=67)
        self.model.fit(X_train, y_train)
        
        return self.model

    # tss() function computes the accuracy of predicted outputs (i.e target prediction on test set)
    def tss(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        
        tss_scores = []
        for i in range(len(cm)):
            # For class i
            TP = cm[i, i]
            FN = np.sum(cm[i, :]) - TP
            FP = np.sum(cm[:, i]) - TP
            TN = np.sum(cm) - TP - FN - FP
            
            if (TP + FN) > 0 and (TN + FP) > 0:
                tss = TP / (TP + FN) - FP / (FP + TN)
                tss_scores.append(tss)
        
        # Return average TSS across all classes
        return np.mean(tss_scores) if tss_scores else 0.0

def visualize_results(svm_model, tss_scores, accuracy_scores, feature_names, baseline_accuracy=None, baseline_label=None):
   
    mean_tss = np.mean(tss_scores)
    std_tss = np.std(tss_scores)
    mean_accuracy = np.mean(accuracy_scores)
    std_accuracy = np.std(accuracy_scores)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def _plot_metric(ax, scores, ylabel, title, average):
        ax.bar(range(len(scores)), scores, color='steelblue', alpha=0.7)
        ax.set_xlabel('Fold')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(range(len(scores)))
        ax.set_xticklabels([f'Fold {i+1}' for i in range(len(scores))], rotation=45)
        ax.axhline(y=average, color='r', linestyle='--', linewidth=2, label=f'Average: {average:.4f}')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    _plot_metric(axes[0], tss_scores, 'TSS Score', 'TSS by Fold', mean_tss)
    axes[0].set_facecolor('#f7f7f7')

    axes[1].bar(range(len(accuracy_scores)), accuracy_scores, color='forestgreen', alpha=0.7)
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy by Fold')
    axes[1].set_xticks(range(len(accuracy_scores)))
    axes[1].set_xticklabels([f'Fold {i+1}' for i in range(len(accuracy_scores))], rotation=45)
    axes[1].axhline(y=mean_accuracy, color='r', linestyle='--', linewidth=2, label=f'Average: {mean_accuracy:.4f}')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_facecolor('#f7f7f7')

    plt.tight_layout()
    plt.show()

    print("\n" + "="*80)
    print("SVM MODEL RESULTS - ALL FEATURES WITH BINARY ENCODING")
    print("="*80)
    print(f"Total Features Used: {len(feature_names)}")
    binary_features = [f for f in feature_names if '_' in f and f.split('_')[0] in COLUMNS.keys()]
    print(f"  - Binary/One-hot Features: {len(binary_features)}")
    print(f"  - Numeric Features: {len(feature_names) - len(binary_features)}")
    print(f"\nAverage TSS Score: {mean_tss:.6f} (± {std_tss:.6f})")
    print(f"Average Accuracy: {mean_accuracy:.6f} (± {std_accuracy:.6f})")
    if baseline_accuracy is not None:
        print(f"Majority-class Baseline Accuracy ({baseline_label}): {baseline_accuracy:.6f}")
        print(f"Random Baseline Accuracy: 0.333333")
        print(f"Improvement over Majority Baseline (Accuracy - Majority Baseline Accuracy): {mean_accuracy - baseline_accuracy:.6f}")
        print(f"Improvement over Random Baseline (Accuracy - Random Baseline Accuracy): {mean_accuracy - 0.333333:.6f}")

    print("\nPer-fold TSS:")
    for i, score in enumerate(tss_scores, 1):
        print(f"  Fold {i}: {score:.6f}")

    print("\nPer-fold Accuracy:")
    for i, score in enumerate(accuracy_scores, 1):
        print(f"  Fold {i}: {score:.6f}")

    print("="*80)

if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    
    print("="*80)
    print("SVM Credit Classifier - All Features with Binary Encoding")
    print("="*80)
    
    print("\n1. Loading and processing data...")
    df = dataCreation(TRAIN_DATA)
    XFiltered, YFiltered = filterData(df)
    
    # Limit to 1000 datapoints for faster testing
    print(f"\n2. Limiting dataset to 1000 datapoints for faster testing...")
    if len(XFiltered) > 1000:
        XFiltered = XFiltered.head(1000)
        YFiltered = YFiltered.head(1000)
    print(f"   Using {len(XFiltered)} datapoints")
    
    print("\n3. Initializing and preprocessing data for SVM (including categorical encoding)...")
    svm_model = mySvm()
    svm_model.convertToBinaryFeatures(XFiltered)
    X_processed, y_processed = svm_model.preprocess(XFiltered, YFiltered)
    print(f"   Processed data shape: X={X_processed.shape}, y={y_processed.shape}")
    if svm_model.feature_breakdown:
        print("   Feature breakdown (binary columns created):")
        for feat, count in sorted(svm_model.feature_breakdown.items()):
            if feat == "numeric_features":
                continue
            print(f"     - {feat}: {count} columns")
        print(f"   Numeric feature count: {svm_model.feature_breakdown.get('numeric_features', 0)}")
    
    majority_label = y_processed.mode()[0]
    baseline_accuracy = (y_processed == majority_label).mean()
    label_map_inv = {0: "Good", 1: "Standard", 2: "Poor"}
    majority_label_name = label_map_inv.get(majority_label, str(majority_label))
    print(f"\nBaseline (majority class = {majority_label_name}): {baseline_accuracy:.6f} accuracy")
    
    print("\n4. Performing 10-fold cross-validation...")
    mean_tss, tss_scores, mean_acc, accuracy_scores = svm_model.cross_validation(
        X_processed.values, y_processed.values, k=10
    )
    
    print("\n5. Reporting cross-validation statistics...")
    visualize_results(
        svm_model,
        tss_scores,
        accuracy_scores,
        svm_model.feature_names,
        baseline_accuracy=baseline_accuracy,
        baseline_label=majority_label_name,
    )
