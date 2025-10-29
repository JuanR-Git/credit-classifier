import re
import pandas as pd
import numpy as np 

TRAIN_DATA = 'train.csv'
FEATURE_NAMES = ["ID","Age","Occupation","Annual_Income","Monthly_Inhand_Salary","Num_Bank_Accounts","Num_Credit_Card","Interest_Rate","Num_of_Loan","Type_of_Loan","Delay_from_due_date","Num_of_Delayed_Payment","Changed_Credit_Limit","Num_Credit_Inquiries","Credit_Mix","Outstanding_Debt","Credit_Utilization_Ratio","Credit_History_Age","Payment_of_Min_Amount","Total_EMI_per_month","Amount_invested_monthly","Payment_Behaviour","Monthly_Balance","Credit_Score"]


COLUMNS = {
    "Age": {
        "type": "int",
        "valid_range": (18, 122),
        "regex": r"^\d+$",
        "cleaning": "Ensure numeric, no extra characters.",
        "notes": "Reject if outside human age range."
    },

    "Number of Credit Cards": {
        "type": "int",
        "valid_range": (0, 11),
        "regex": r"^\d+$",
        "cleaning": "Ensure integer only.",
        "notes": "Reject non-integer or real numbers."
    },

    "Number of Bank Accounts": {
        "type": "int",
        "valid_range": (0, 11),
        "regex": r"^\d+$",
        "cleaning": "Ensure integer only.",
        "notes": "Reject if outside 0–11 range."
    },

    "Occupation": {
        "type": "str",
        "regex": None,
        "cleaning": "Bag-of-words classification. Normalize capitalization and whitespace.",
        "notes": "Use ‘__________’ to mark ‘no other’. Assign numeric encoding after NLP preprocessing."
    },

    "Annual Income": {
        "type": "float",
        "valid_range": None,
        "regex": r"\d+(?:\.\d+)?",
        "cleaning": "Remove underscores and non-numeric symbols before conversion.",
        "notes": "Check if real number."
    },

    "Monthly In hand Salary": {
        "type": "float",
        "valid_range": None,
        "regex": r"\d+(?:\.\d+)?",
        "cleaning": "Remove underscores and non-numeric characters.",
        "notes": "Tentative column—verify its interpretation."
    },

    "Interest Rate": {
        "type": "float",
        "valid_range": (0, 100),
        "regex": r"\d+(?:\.\d+)?",
        "cleaning": "Ensure percentage format without % sign.",
        "notes": "Reject >100."
    },

    "Number of Loans": {
        "type": "int",
        "valid_range": (0, None),
        "regex": r"^\d+$",
        "cleaning": "Ensure integer only.",
        "notes": "Should match length of 'Type of Loan' list."
    },

    "Type of Loan": {
        "type": "str",
        "regex": None,
        "cleaning": "Tokenize categories, possibly split on commas.",
        "notes": "Correlates to 'Number of Loans'. Verify consistency."
    },

    "Delay from Due Date": {
        "type": "int",
        "valid_range": (0, None),
        "regex": r"^-?\d+$",
        "cleaning": "Raw numeric values only.",
        "notes": "May include negatives if early payments are encoded as negative delays."
    },

    "Num of Delayed Payments": {
        "type": "int",
        "valid_range": (0, None),
        "regex": r"^\d+$",
        "cleaning": "Remove blanks and negatives.",
        "notes": "Must be non-negative integer."
    },

    "Changed Credit Limit": {
        "type": "float",
        "regex": r"[+-]?\d+(?:\.\d+)?",
        "cleaning": "Ensure numeric string, use regex validation.",
        "notes": "Could be positive or negative depending on direction of change."
    },

    "Number of Credit Inquiries": {
        "type": "int",
        "valid_range": (0, None),
        "regex": r"^\d+$",
        "cleaning": "Ensure integer only.",
        "notes": "Usually small whole number count."
    },

    "Credit Mix": {
        "type": "str",
        "regex": None,
        "cleaning": "Categorical encoding via bag-of-words or label mapping.",
        "notes": "Indicates mix of secured/unsecured credit types."
    },

    "Outstanding Debt": {
        "type": "float",
        "regex": r"\d+(?:\.\d+)?",
        "cleaning": "Ensure numeric, remove formatting symbols.",
        "notes": "Should be non-negative."
    },

    "Credit Utilization Ratio": {
        "type": "float",
        "valid_range": (0, 1),
        "regex": r"\d+(?:\.\d+)?",
        "cleaning": "Convert percentage values (e.g., '45%') to 0.45.",
        "notes": "Should be expressed as a fraction between 0 and 1."
    },

    "Credit History Age": {
        "type": "float",
        "valid_range": (0, None),
        "regex": r"\d+(?:\.\d+)?",
        "cleaning": "Ensure numeric representation (years).",
        "notes": "Derived metric; must be non-negative."
    },

    "Payment Min Amount": {
        "type": "float",
        "regex": r"\d+(?:\.\d+)?",
        "cleaning": "Ensure numeric.",
        "notes": "Check if zero values indicate missing data."
    },

    "Total EMI per Month": {
        "type": "float",
        "regex": r"\d+(?:\.\d+)?",
        "cleaning": "Ensure numeric, remove underscores/commas.",
        "notes": "Non-negative; may be related to number of loans."
    },

    "Amount Invested Monthly": {
        "type": "float",
        "regex": r"\d+(?:\.\d+)?",
        "cleaning": "Ensure numeric.",
        "notes": "Non-negative; financial variable."
    },

    "Payment Behaviours": {
        "type": "str",
        "regex": None,
        "cleaning": "Categorical encoding. NLP or one-hot encoding likely required.",
        "notes": "Represents qualitative behavior descriptors."
    },

    "Monthly Balance": {
        "type": "float",
        "regex": r"\d+(?:\.\d+)?",
        "cleaning": "Ensure numeric, remove formatting symbols.",
        "notes": "May be negative if overdrawn."
    }
}


# def validate_column_value(col_name, value):
#     info = COLUMNS[col_name]
#     if info["regex"] and not re.match(info["regex"], str(value).strip()):
#         return False
#     if(info["type"] == float or info["type"] == int):
#         if info["valid_range"]:
#             min_val, max_val = info["valid_range"]
#             num = float(value)
#             if (min_val is not None and num < min_val) or (max_val is not None and num > max_val):
#                 return False
#     elif(info["type"] == str):
        
#     return True

def dataCreation(str filePath):
    ##init fp
    ##expecting Z x 100,000 Pandas DS
    df = pd.read_csv('../data/' + TRAIN_DATA, usecols=FEATURE_NAMES)
    df = df.dropna()  # Filter out rows with NULL values
    return df

def filterData(DataFrame df):
    #need to add regex shit
    #need to figure out str arbitrary numerical assignments
    #need to add lists of possible values for strings to each appicable dictrionarcies 
    #seperate X & Y
    #tentatively consider not killing ourselves with Type_of_Loan
    #maybe only do this for 1000 points for this point in time for testing 



    
    for x in df.index:
        for y, key in COLUMNS:
            

            if y["type"] == "int" or y["type"] == "float":
                if df.loc[x, key] > y["valid_range"][1] or df.loc[x,key] < y["valid_range"][0]:
                df.drop(x, inplace = True)
    return XFiltered, YFiltered