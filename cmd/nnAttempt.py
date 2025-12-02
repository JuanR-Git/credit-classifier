import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import re
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold


TRAIN_DATA = 'train.csv'
LEARNING_RATE = 1e-5
BATCH_SIZE = 256
L2_CONSTANT = 3e-5
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
    
    Args:
        filePath: Relative path to the CSV file within the data directory
        
    Returns:
        pd.DataFrame: Cleaned dataframe with specified columns and no null values
        
    Note:
        Expected dataset size: Z x 100,000 rows
    """
    # Construct absolute path to data file relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    data_path = os.path.join(data_dir, filePath)
    
    # Load only relevant columns from CSV
    df = pd.read_csv(data_path, usecols=list(COLUMNS.keys()), encoding="utf-8")
    
    # Remove rows with any null values
    df = df.dropna()
    
    return df


def filterData(df: pd.DataFrame) -> tuple[list, list]:
    """
    Validate and transform dataframe rows into feature and target arrays.
    
    Processes each row through column-specific validation rules. Rows that pass
    all validations are split into features (X) and targets (Y) based on the
    credit score index position.
    
    Args:
        df: Input dataframe with raw credit data
        
    Returns:
        tuple: (XFiltered, YFiltered) where:
            - XFiltered: List of feature vectors (all columns before credit score)
            - YFiltered: List of target vectors (credit score and following columns)
            
    Note:
        Prints progress every 1000 rows and summary statistics at completion
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
            
            # If validation fails, skip this row entirely
            if val is False:
                row_passed = False
                break
            # Handle multi-value columns (e.g., binary encoded lists)
            elif isinstance(val, list):
                filtered_row.extend(val)
            # Handle single-value columns
            else:
                filtered_row.append(val)

        # Split validated row into features and targets
        if row_passed:
            credit_score_index = len(filtered_row) - 3
            XFiltered.append(filtered_row[:credit_score_index])
            YFiltered.append(filtered_row[credit_score_index:])
        
        # Progress logging
        if x % 1000 == 0 and x > 0:
            print(f"Processed {x} rows, {len(XFiltered)} passed validation...")
    
    # Print final statistics
    print(f"Filtered data size: {len(XFiltered)} rows, "
          f"{len(XFiltered[0])} columns, "
          f"first entry: {XFiltered[0] if XFiltered else 'N/A'}, "
          f"outputs: {YFiltered[0] if YFiltered else 'N/A'}")

    return XFiltered, YFiltered


def validateColumnValue(col_name: str, value):
    """
    Validate and transform a single column value according to predefined rules.
    
    Handles multiple data types with specific validation logic:
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
            - String types: integer index
            - List types: binary cumulation integer
            - Time types: float (years)
            - Output types: list of binary indicators
        False on validation failure
    """
    # Verify column exists in configuration
    if col_name not in COLUMNS.keys():
        print(f"Failed validation - unknown column: {col_name}, value: {value}")
        return False

    col_info = COLUMNS[col_name]
    
    # Handle numeric types (float/int)
    if col_info.get("type") in ["float", "int"]:
        min_val, max_val = col_info["valid_range"]
        try:
            # Extract numeric value using regex
            match = re.findall(col_info["regex"], str(value).strip())

            if len(match):
                num = float(match[0])
            else:
                print(f"Failed regex validation: {col_name}, value: {value}, "
                      f"regex: {col_info['regex']}")
                return False

            # Validate range (None means unbounded)
            if ((min_val is None or num >= min_val) and
                (max_val is None or num <= max_val)):
                return num

        except ValueError:
            pass  # Fall through to failure

    # Handle categorical string types
    elif col_info.get("type") == "str":
        if col_info.get("values") and value in col_info["values"]:
            # Return index position for encoding
            return col_info["values"].index(value)

    # Handle list types (comma-separated values)
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

    # Handle time duration types
    elif col_info.get("type") == "time":
        match = re.findall(col_info["regex"], str(value).strip())
        if match and len(match) >= 1:
            years = int(match[0])
            months = int(match[1]) if len(match) > 1 else 0
            total_years = years + months / 12.0
            if total_years >= 0:
                return total_years
    
    # Validation failed for all type handlers
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

FIRST_LAYER_NEURONS = 1024
SECOND_LAYER_NEURONS = 1024
THIRD_LAYER_NEURONS = 512
FOURTH_LAYER_NEURONS = 512
FIFTH_LAYER_NEURONS = 64
OUTPUT_LAYER_NEURONS = 3

class CreditClassifierNN(nn.Module):
    """
    Neural network classifier for credit risk assessment.
    
    Architecture:
        - 5 fully connected layers with decreasing width
        - Batch normalization after first two layers
        - ReLU activation functions
        - Dropout regularization at two points in the network
    
    Args:
        input_size: Number of input features
    """

    def __init__(self, input_size):
        super(CreditClassifierNN, self).__init__()
        
        # Define fully connected layers with decreasing dimensions
        self.fc1 = nn.Linear(input_size, FIRST_LAYER_NEURONS)
        self.fc2 = nn.Linear(FIRST_LAYER_NEURONS, SECOND_LAYER_NEURONS)
        self.fc3 = nn.Linear(SECOND_LAYER_NEURONS, THIRD_LAYER_NEURONS)
        self.fc4 = nn.Linear(THIRD_LAYER_NEURONS, FOURTH_LAYER_NEURONS)
        self.fc5 = nn.Linear(FOURTH_LAYER_NEURONS, FIFTH_LAYER_NEURONS)
        self.fc6 = nn.Linear(FIFTH_LAYER_NEURONS, OUTPUT_LAYER_NEURONS)
        
        # Batch normalization layers for training stability
        self.batchNorm1 = nn.BatchNorm1d(FIRST_LAYER_NEURONS)
        self.batchNorm2 = nn.BatchNorm1d(SECOND_LAYER_NEURONS)
        self.batchNorm3 = nn.BatchNorm1d(THIRD_LAYER_NEURONS)
        self.batchNorm4 = nn.BatchNorm1d(FOURTH_LAYER_NEURONS)
        self.batchNorm5 = nn.BatchNorm1d(FIFTH_LAYER_NEURONS)

        # Activation function
        self.relu = nn.ReLU()
        
        # Dropout layers for regularization
        self.dropout1 = nn.Dropout(0.35)  # Higher dropout after first layer
        self.dropout2 = nn.Dropout(0.1)   # Lower dropout in deeper layers
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            torch.Tensor: Raw logits of shape (batch_size, num_classes)
        """
        # First block: FC -> BatchNorm -> ReLU -> Dropout
        x = self.relu(self.batchNorm1(self.fc1(x)))
        x = self.dropout1(x)
        
        # Second block: FC -> BatchNorm -> ReLU
        x = self.relu(self.batchNorm2(self.fc2(x)))
        x = self.dropout1(x)
        
        # Third block: FC -> BatchNorm -> ReLU ->  Dropout
        x = self.relu(self.batchNorm3(self.fc3(x)))
        x = self.dropout1(x)
        
        # Fourth block: FC -> BatchNorm -> ReLU -> Dropout
        x = self.relu(self.batchNorm4(self.fc4(x)))
        x = self.dropout2(x)

        # Fifth block: FC -> BatchNorm -> ReLU -> Dropout
        x = self.relu(self.batchNorm5(self.fc5(x)))
        x = self.dropout2(x)

        # Output layer (no activation, raw logits for CrossEntropyLoss)
        return self.fc6(x)


def calculate_full_loss(model, criterion, X, Y):
    """
    Calculate loss on entire dataset without gradient computation.
    
    Args:
        model: Neural network model
        criterion: Loss function
        X: Input features tensor
        Y: Target labels tensor
        
    Returns:
        float: Loss value as a Python scalar
    """
    model.eval()  # Set model to evaluation mode (disables dropout/batchnorm training behavior)
    
    with torch.no_grad():  # Disable gradient calculation for efficiency
        outputs = model(X)
        loss = criterion(outputs, Y)
    
    model.train()  # Restore model to training mode
    
    return loss.item()


def calculate_accuracy(model, X, Y):
    """
    Calculate classification accuracy on a dataset.
    
    Args:
        model: Neural network model
        X: Input features tensor of shape (n_samples, n_features)
        Y: Target label indices tensor of shape (n_samples,)
        
    Returns:
        float: Accuracy as a fraction (0.0 to 1.0)
    """
    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():
        outputs = model(X)
        # Get predicted class by taking argmax of logits
        correct = (outputs.argmax(dim=1) == Y).sum().item()
        return correct / len(Y)


def train_credit_classifier(model, criterion, optimizer, X_train, Y_train, X_val, Y_val,
                           num_iterations, batch_size, check_every):
    """
    Train the credit classifier with mini-batch gradient descent and L1 regularization.
    
    Features:
        - Mini-batch training with shuffling
        - L1 regularization on fc2 layer weights
        - Periodic validation and early stopping
        - Tracks loss and accuracy metrics
    
    Args:
        model: CreditClassifierNN instance
        criterion: Loss function (e.g., CrossEntropyLoss)
        optimizer: Optimizer (e.g., Adam)
        X_train: Training features tensor
        Y_train: Training label indices tensor
        X_val: Validation features tensor
        Y_val: Validation label indices tensor
        num_iterations: Total number of training iterations
        batch_size: Number of samples per mini-batch
        check_every: Frequency of validation checks (in iterations)
        
    Returns:
        tuple: (train_losses, val_losses, train_accs, val_accs, epochs, model)
            - train_losses: List of training losses at checkpoints
            - val_losses: List of validation losses at checkpoints
            - train_accs: List of training accuracies at checkpoints
            - val_accs: List of validation accuracies at checkpoints
            - epochs: List of epoch numbers corresponding to checkpoints
            - model: Trained model
    """
    # Create dataset for DataLoader
    train_dataset = TensorDataset(X_train, Y_train)

    # Initialize metric tracking lists
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    epochs = []
    
    iteration = 0
    patience = 3

    print(f"Training for {num_iterations} iterations with batch size {batch_size} "
          f"(check every {check_every})")

    model.train()  # Set model to training mode
    
    # Training loop: continue until reaching target iterations
    while iteration < num_iterations:
        # Create new DataLoader each epoch to reshuffle data
        data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Process each mini-batch
        for batch_X, batch_Y in data_loader:
            if iteration >= num_iterations:
                break

            # Standard training step
            optimizer.zero_grad()  # Clear gradients from previous step
            outputs = model(batch_X)  # Forward pass
            
            # Total loss = cross entropy
            loss = criterion(outputs, batch_Y)
            
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            # Periodic evaluation and metric logging
            if iteration % check_every == 0:
                # Calculate metrics on full train and validation sets
                train_loss = calculate_full_loss(model, criterion, X_train, Y_train)
                val_loss = calculate_full_loss(model, criterion, X_val, Y_val)
                train_acc = calculate_accuracy(model, X_train, Y_train)
                val_acc = calculate_accuracy(model, X_val, Y_val)
                
                # Early stopping: check for minimal improvement in validation accuracy
                if len(val_accs) > 0 and ((val_acc - val_accs[-1]) < 1e-3 or (val_loss - val_losses[-1]) > 1e-3):
                    if patience <= 0:
                        print("Early stopping due to minimal validation accuracy improvement.")
                        return train_losses, val_losses, train_accs, val_accs, epochs, model
                    else:
                        patience = patience - 1
                
                # Record metrics
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_accs.append(train_acc)
                val_accs.append(val_acc)
                epochs.append(iteration)
                
                # Print progress
                print(f"Check {iteration//check_every}/{num_iterations//check_every}: "
                      f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                      f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
            
            iteration += 1

    return train_losses, val_losses, train_accs, val_accs, epochs, model


def evaluate_classification_metrics(model, X_val, Y_val, batch_size=64):
    """
    Evaluate model performance using confusion matrix and classification report.
    
    Args:
        model: Trained CreditClassifierNN model
        X_val: Validation features tensor
        Y_val: Validation label indices tensor
        batch_size: Batch size for evaluation (default: 64)
        
    Returns:
        tuple: (cm, report) where:
            - cm: Confusion matrix (numpy array)
            - report: Classification report string with precision, recall, F1-score
            
    Note:
        Assumes 3 classes: "Good", "Standard", "Poor"
    """
    model.eval()  # Set model to evaluation mode
    
    all_preds = []
    all_labels = []
    
    # Create DataLoader for batch processing
    val_dataset = TensorDataset(X_val, Y_val)
    dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Collect predictions without gradient computation
    with torch.no_grad():
        for X, Y in dataloader:
            logits = model(X)              # Raw model outputs
            preds = logits.argmax(dim=1)   # Convert to predicted class indices
            labels = Y                      # Ground truth class indices

            # Move to CPU and accumulate
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all batches and convert to numpy
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Generate detailed classification report
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=["Good", "Standard", "Poor"]
    )

    return cm, report


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    torch.set_printoptions(edgeitems=7)

    print("Loading and filtering training data...")
    df = dataCreation(TRAIN_DATA)
    XFiltered_unorm, YFiltered_unorm = filterData(df)

    X_arr = np.array(XFiltered_unorm, dtype=np.float32)
    Y_arr = np.array(YFiltered_unorm, dtype=np.float32)

    # Convert labels to class indices
    Y_indices = np.argmax(Y_arr, axis=1)

    # Compute class weights once
    num_good = YFiltered_unorm.count([1, 0, 0])
    num_standard = YFiltered_unorm.count([0, 1, 0])
    num_poor = YFiltered_unorm.count([0, 0, 1])
    class_counts = torch.tensor([num_good, num_standard, num_poor], dtype=torch.float32)
    class_weights = (1.0 / class_counts)
    class_weights = class_weights / class_weights.sum()

    print(f"Class distribution - Good: {num_good}, Standard: {num_standard}, Poor: {num_poor}")
    print(f"Class weights: {class_weights}")

    # Set up K-fold cross validation
    K = 10
    kf = KFold(n_splits=K, shuffle=True, random_state=42)

    fold_results = []
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    fold_histories = []

    fold_number = 1
    print(f"\nStarting {K}-fold cross-validation...\n")

    for train_index, val_index in kf.split(X_arr):
        print(f"\n===== Fold {fold_number} / {K} =====")

        # Split
        X_train_unorm = torch.tensor(X_arr[train_index], dtype=torch.float32)
        Y_train = torch.tensor(Y_indices[train_index], dtype=torch.long)

        X_val_unorm = torch.tensor(X_arr[val_index], dtype=torch.float32)
        Y_val = torch.tensor(Y_indices[val_index], dtype=torch.long)

        # Normalize using training fold statistics
        X_train_norm, train_min, train_max = normalizeDataTrain(X_train_unorm)
        X_val_norm = normalizeDataWithMinMax(X_val_unorm, train_min, train_max)

        # Init model, loss, optimizer
        credit_nn = CreditClassifierNN(input_size=X_train_norm.shape[1])
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(
            credit_nn.parameters(),
            lr=LEARNING_RATE,
            weight_decay=L2_CONSTANT
        )

        # Train
        EPOCH_LEN = len(X_train_norm)
        train_losses_one_fold, val_losses_one_fold, train_accs_one_fold, val_accs_one_fold, epochs, model = train_credit_classifier(
            credit_nn,
            criterion,
            optimizer,
            X_train_norm,
            Y_train,
            X_val_norm,
            Y_val,
            EPOCH_LEN * 50,
            BATCH_SIZE,
            EPOCH_LEN // 10,
        )

        # Predictions
        preds = model(X_val_norm)
        pred_classes = torch.argmax(preds, dim=1)

        cm = confusion_matrix(Y_val.numpy(), pred_classes.numpy())
        fold_accuracy = (cm[0,0] + cm[1,1] + cm[2,2]) / cm.sum()
        print(f"Fold {fold_number} accuracy: {fold_accuracy:.4f}")

        fold_results.append({
            "fold": fold_number,
            "confusion_matrix": cm,
            "accuracy": fold_accuracy
        })

        fold_histories.append({
            "fold": fold_number,
            "train_losses": train_losses_one_fold,
            "val_losses": val_losses_one_fold,
            "train_accs": train_accs_one_fold,
            "val_accs": val_accs_one_fold,
            "epochs": epochs
        })

        fold_number += 1
        

    # Summary of all folds
    print("\n===== Cross-validation Summary =====")
    for res in fold_results:
        print(f"Fold {res['fold']}: accuracy = {res['accuracy']:.4f}")

    avg_acc = np.mean([r["accuracy"] for r in fold_results])
    print(f"\nAverage accuracy across {K} folds: {avg_acc:.4f}\n")

    print("Cross-validation complete.")

    # Optional: Calculate accuracy per class
    print("\nPer-class accuracy:")
    for i, label in enumerate(['Poor', 'Standard', 'Good']):
        accuracy = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"{label}: {accuracy:.2%}")

    # Visualize training progress
    print("\nGenerating training plots...")

    total_cm = sum(res["confusion_matrix"] for res in fold_results)
    plt.figure(figsize=(6,5))
    sns.heatmap(
        total_cm,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=['Poor', 'Standard', 'Good'],
        yticklabels=['Poor', 'Standard', 'Good']
    )
    plt.title("Average Confusion Matrix Across Folds")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # -------------------------
    # Left Plot: Loss
    # -------------------------
    for hist in fold_histories:
        ax1.plot(hist["epochs"], hist["train_losses"], linestyle=':', marker='o', label=f'Fold {hist["fold"]} Train')
        ax1.plot(hist["epochs"], hist["val_losses"], linestyle='-', marker='x', label=f'Fold {hist["fold"]} Val')

    ax1.set_title('Training and Validation Loss Across Folds')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # -------------------------
    # Right Plot: Accuracy
    # -------------------------
    for hist in fold_histories:
        ax2.plot(hist["epochs"], hist["train_accs"], linestyle=':', marker='o', label=f'Fold {hist["fold"]} Train Acc')
        ax2.plot(hist["epochs"], hist["val_accs"], linestyle='-', marker='x', label=f'Fold {hist["fold"]} Val Acc')

    ax2.set_title('Training and Validation Accuracy Across Folds')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
    
    print("\nTraining complete!")
    avg_acc = np.mean([r["accuracy"] for r in fold_results])
    print("\nFinal Accuracy: ", avg_acc)