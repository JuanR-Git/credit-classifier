import re
import pandas as pd
import numpy as np 
import sys

TRAIN_DATA = 'train.csv'
COLUMNS = {
    "Age": {
        "type": "int",
        "valid_range": (18, 122),
        "regex": r"\d+(?:\.0+)?",
        "cleaning": "Ensure numeric, no extra characters.",
        "notes": "Reject if outside human age range."
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
        "notes": "Use ‘_______’ to mark ‘no other’. Assign numeric encoding after NLP preprocessing."
    },

    "Annual_Income": {
        "type": "float",
        "valid_range": (None, None),
        "regex": r"\d+(?:\.\d+)?",
        "cleaning": "Remove underscores and non-numeric symbols before conversion.",
        "notes": "Check if real number."
    },

    "Monthly_Inhand_Salary": {
        "type": "float",
        "valid_range": (None, None),
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
        "notes": "May include negatives if early payments are encoded as negative delays."
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
        "valid_range": (None, None),
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
        "valid_range": (None, None),
        "values": ["No", "NM", "Yes"],
        "cleaning": "Ensure Yes, No, or NM (not much).",
        "notes": "Collects Yes, No, or NM (not much) into a small bag of words."
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
        "valid_range": (None, None),
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

def validate_column_value(col_name, value):
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
            if len(match):
                num = float(match[0])
            else:
                print("failed val regex:", col_name, value, col_info["regex"], str(value).strip())
                return False
            if ((min_val is None or num >= min_val) and
                (max_val is None or num <= max_val)):
                # print("successful num val:", num)
                return num  # return numeric value
        except ValueError:
            pass

    # String type: return value if valid
    elif col_info.get("type") == "str":
        if col_info.get("values") and value in col_info["values"]:
            # print("successful str val:", value)
            return col_info["values"].index(value)

    # List type: return list if all valid
    elif col_info.get("type") == "list":
        if col_info.get("values"):
            list_items = [item.strip() for item in str(value).split(',')]
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
    df = pd.read_csv('data/' + TRAIN_DATA, usecols=list(COLUMNS.keys()), encoding="utf-8")
    df = df.dropna()  # Filter out rows with NULL values
    return df

def filterData(df: pd.DataFrame):
    XFiltered = pd.DataFrame(columns=COLUMNS.keys())
    for x in df.index:
        row_passed = True
        filtered_row = []

        for col_name, col_body in COLUMNS.items():
            val = validate_column_value(col_name, df.loc[x, col_name])
            
            # If validation fails
            if val is False:
                row_passed = False
                break
            else:
                filtered_row.append(val)

        # Append only if all columns passed validation
        print("Row", x, "passed:", row_passed)
        if row_passed:
            XFiltered.loc[x] = filtered_row
    print("Filtered data size:", XFiltered.shape,'\n', XFiltered)
    YFiltered = XFiltered["Credit_Score"]
    return XFiltered, YFiltered

if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    df = dataCreation(TRAIN_DATA)
    XFiltered, YFiltered = filterData(df)
    # print("XFiltered:", XFiltered, XFiltered.size)
    # print("YFiltered:", YFiltered, YFiltered.size)
