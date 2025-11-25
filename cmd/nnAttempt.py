import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import re
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

FIRST_LAYER_NEURONS_DROPOUT = 32
SECOND_LAYER_NEURONS_DROPOUT = 16
OUTPUT_LAYER_NEURONS_DROPOUT = 3

TRAIN_DATA = 'train.csv'
LEARNING_RATE = 1e-4
NUMBER_OF_DATAPOINTS = 30000
NUMBER_OF_TOTAL_COLUMNS = 30
DROPOUT_RATE = 0.3
BATCH_SIZE = 32
L2_CONSTANT = 1e-4
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
        "type": "list",
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
    df = pd.read_csv(data_path, usecols=list(COLUMNS.keys()), encoding="utf-8")#, nrows=#NUMBER_OF_DATAPOINTS)
    df = df.dropna()  # Filter out rows with NULL values
    return df

# future improvement: make the search for validating faster using search algo 
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

        is_item_in_list_present = []
        for item in col_info["values"]:
            if item in list_items:
                is_item_in_list_present.append(1)
            else:
                is_item_in_list_present.append(0)
        return is_item_in_list_present

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

def normalizeData(X, Y):
    X_array = np.array(X, dtype=np.float32)

    # Normalize features in X using Min-Max scaling
    X_min = X_array.min(axis=0)
    X_max = X_array.max(axis=0)
    X_normalized = (X_array - X_min) / (X_max - X_min + 1e-9)  # Adding a small constant to avoid division by zero

    return X_normalized, Y

class CreditClassifierNN(nn.Module):
    """
    TODO: Complete this neural network class to include dropout layers.

    The dropout_rate parameter should control the dropout probability.
    When dropout_rate=0.0, no dropout is applied.
    """

    def __init__(self, input_size, dropout_rate=0.0):
        super(CreditClassifierNN, self).__init__()
        # Define connected layers
        self.fc1 = nn.Linear(input_size, FIRST_LAYER_NEURONS_DROPOUT)
        self.fc2 = nn.Linear(FIRST_LAYER_NEURONS_DROPOUT, SECOND_LAYER_NEURONS_DROPOUT)
        self.fc3 = nn.Linear(SECOND_LAYER_NEURONS_DROPOUT, OUTPUT_LAYER_NEURONS_DROPOUT)
        # Define activation functions and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        """
        Forward Pass with Dropout Layers.
        Apply dropout after each hidden layer if dropout_rate > 0.0.
        Return the final output.
        """
        # First hidden layer with ReLU and optional dropout
        x = self.relu(self.fc1(x))
        if self.dropout_rate > 0.0:
            x = self.dropout(x)
        
        # Second hidden layer with ReLU and optional dropout
        x = self.relu(self.fc2(x))
        if self.dropout_rate > 0.0:
            x = self.dropout(x)
        
        # Output layer
        return self.fc3(x)

def calculate_full_loss(model, criterion, X, Y):
    """Helper function to calculate loss over an entire dataset."""
    model.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculation
        outputs = model(X)
        loss = criterion(outputs, Y)
    model.train() # Set model back to train mode
    return loss.item()

def calculate_accuracy(model, X, Y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        correct = (logits.argmax(dim=1) == Y.argmax(dim=1)).sum().item()
        return correct / len(Y)

def train_with_dropout(model, criterion, optimizer, X_train, Y_train, X_val, Y_val,
                                 num_iterations, batch_size, check_every):
    """
    TODO: Complete this training function to support dropout.
    """
    # CODE HERE: Use need to fill like using miniSGD in part 2
    train_dataset = TensorDataset(X_train, Y_train)

    train_losses = []
    val_losses = []
    iterations = []
    train_accs = []
    val_accs = []
    iteration = 0
    loss = None

    print(f"Training for {num_iterations} iterations with batch size {batch_size} (check every {check_every})")
            
    model.train()
    # Keep looping until total iterations are reached
    while iteration < num_iterations:
        # Create new dataloader for each completed examination of dataset to shuffle data
        data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Loop through batches
        for batch_X, batch_Y in data_loader:
            if iteration > num_iterations:
                break

            optimizer.zero_grad()
            outputs = model(batch_X)

            # Compute loss and backpropagate
            # outputs_index = torch.argmax(outputs, dim=1)

            # print("batch_Y:", batch_Y, "outputs:", outputs)#, "outputs_index:", outputs_index, )
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()

            # Check and record losses periodically
            if iteration % check_every == 0:
                # Calculate full losses and accuracies
                train_loss = calculate_full_loss(model, criterion, X_train, Y_train)
                val_loss = calculate_full_loss(model, criterion, X_val, Y_val)
                train_acc = calculate_accuracy(model, X_train, Y_train)
                val_acc = calculate_accuracy(model, X_val, Y_val)
                
                # Append metrics
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                iterations.append(iteration)
                train_accs.append(train_acc)
                val_accs.append(val_acc)
                print(f"Check #{iteration/check_every}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
            # increase iteration count
            iteration += 1
    return train_losses, val_losses, train_accs, val_accs, iterations, model
    
if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    torch.set_printoptions(edgeitems=7)
    df = dataCreation(TRAIN_DATA)
    XFiltered_unorm, YFiltered_unorm = filterData(df)
    XFiltered, YFiltered = normalizeData(XFiltered_unorm, YFiltered_unorm)
    # split XFiltered, YFiltered into training and validation sets
    split_index = int(0.8 * len(XFiltered))
    X_train_t = torch.tensor(XFiltered[:split_index], dtype=torch.float32)
    Y_train_t = torch.tensor(YFiltered[:split_index], dtype=torch.float32)
    X_val_t = torch.tensor(XFiltered[split_index:], dtype=torch.float32)
    Y_val_t = torch.tensor(YFiltered[split_index:], dtype=torch.float32)
    print("Training data size:", X_train_t.shape, Y_train_t.shape)
    print("Validation data size:", X_val_t.shape, Y_val_t.shape)

    EPOCH_LEN = len(X_train_t)  # Number of training samples
    credit_nn = CreditClassifierNN(input_size=NUMBER_OF_TOTAL_COLUMNS, dropout_rate=DROPOUT_RATE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(credit_nn.parameters(), lr=LEARNING_RATE, weight_decay=L2_CONSTANT)
    print("X_train_t shape:", X_train_t.shape, "Y_train_t shape:", Y_train_t.shape, "X_val_t shape:", X_val_t.shape, "Y_val_t shape:", Y_val_t.shape)

    train_losses, val_losses, train_accs, val_accs, iterations, model = train_with_dropout(credit_nn, criterion, optimizer, X_train_t, Y_train_t, X_val_t, Y_val_t, EPOCH_LEN*5, BATCH_SIZE, EPOCH_LEN//10)

    # --- Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Loss
    ax1.plot(iterations, train_losses, label='Minibatch - Train Loss', linestyle=':', color='green', marker='o')
    ax1.plot(iterations, val_losses, label='Minibatch - Validation Loss', linestyle='-', color='green', marker='x')
    ax1.set_title('Minibatch FeedForward NN Training Loss with Dropout')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss (BCELoss)')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Accuracy
    ax2.plot(iterations, train_accs, label='Minibatch - Train Accuracy', linestyle=':', color='blue', marker='o')
    ax2.plot(iterations, val_accs, label='Minibatch - Validation Accuracy', linestyle='-', color='blue', marker='x')
    ax2.set_title('Minibatch FeedForward NN Training Accuracy with Dropout')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()