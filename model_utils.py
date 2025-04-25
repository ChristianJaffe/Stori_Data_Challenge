import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

# Function that reads a dataset from a file and returns a pandas DataFrame
# It supports CSV, Excel, and JSON formats.
# It also handles errors such as file not found or unsupported file types.
def read_dataset(file_path, file_type='csv', **kwargs):
    """
    Parameters:
        file_path (str): Path to the dataset file.
        file_type (str): Type of the file ('csv', 'excel', 'json'). Default is 'csv'.
        **kwargs: Additional keyword arguments to pass directly to the
                  corresponding pandas read function (e.g., read_csv, read_excel).

    Returns:
        pd.DataFrame or None: Loaded dataset as a pandas DataFrame if successful,
                              otherwise None if an error occurs.
    """
    try:
        # Read based on specified file type
        if file_type == 'csv':
            df = pd.read_csv(file_path, **kwargs)
        elif file_type == 'excel':
            df = pd.read_excel(file_path, **kwargs)
        elif file_type == 'json':
            df = pd.read_json(file_path, **kwargs)
        else:
            # Raise error for unsupported types
            raise ValueError(f"Unsupported file type: {file_type}. Please use 'csv', 'excel', or 'json'.")

        print(f"Dataset loaded successfully from '{file_path}' with shape: {df.shape}")
        return df

    except FileNotFoundError:
        # Handle case where the file does not exist
        print(f"Error: File not found at {file_path}")
        return None # Return None to indicate failure
    except ValueError as ve:
        # Handle specific errors like unsupported file type
        print(f"Error: {ve}")
        return None
    except Exception as e:
        # Handle any other unexpected errors during file reading
        print(f"An unexpected error occurred while reading the file: {e}")
        return None


def create_preprocessor():
    """
    Creates and returns a standard preprocessing pipeline.

    This pipeline includes:
    1. SimpleImputer: Fills missing numerical values using the median strategy.
    2. StandardScaler: Scales features to have zero mean and unit variance.

    Returns:
        Pipeline: A scikit-learn Pipeline object ready to be used for preprocessing.
    """
    return Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Fill NaNs with median
        ('scaler', StandardScaler())                  # Scale features
    ])

class FraudModelTrainer:
    """
    Encapsulates the workflow for training, evaluating, and analyzing
    a scikit-learn compatible classification model, specifically tailored
    for tasks like fraud detection.

    It combines preprocessing, model training, prediction, evaluation metric
    calculation, confusion matrix plotting, and feature importance extraction.
    """
    def __init__(self, model, model_name, preprocessor):
        """
        Initializes the FraudModelTrainer instance.

        Args:
            model: A scikit-learn compatible classifier object (e.g., LogisticRegression(),
                   XGBClassifier()). This model should be instantiated with desired
                   parameters (like balancing weights) *before* being passed here.
            model_name (str): A user-friendly name for the model (e.g., "Logistic Regression", "XGBoost")
                             used for printing and labeling.
            preprocessor (Pipeline): A scikit-learn Pipeline object that handles
                                     preprocessing steps (e.g., imputation, scaling)
                                     before the model sees the data. Typically created
                                     using the `create_preprocessor` function.
        """
        if model is None:
            raise ValueError("A valid model object must be provided.")
        self.model_base = model # The base classifier (LR, XGB, etc.)
        self.model_name = model_name # Name for logging/plotting
        self.preprocessor = preprocessor # Preprocessing steps
        self.pipeline = None # The final pipeline (preprocessor + classifier) built in .train()
        self.feature_names_in_ = None # Stores feature names if input is a DataFrame

    def train(self, X_train, y_train):
        """
        Trains the complete pipeline (preprocessing + model) on the provided training data.

        Args:
            X_train (pd.DataFrame or np.ndarray): Training features. If DataFrame, column names are stored.
            y_train (pd.Series or np.ndarray): Training target variable.
        """
        print(f"\n--- Training Model: {self.model_name} ---")
        # Create the full pipeline combining the preprocessor and the base model
        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', self.model_base)
        ])
        try:
            # Store feature names if input X_train is a pandas DataFrame
            if isinstance(X_train, pd.DataFrame):
                self.feature_names_in_ = X_train.columns.tolist()
            else:
                 print("Warning: X_train is not a DataFrame, specific feature names were not saved.")
                 # Optionally handle numpy array case if feature names are needed later

            # Fit the entire pipeline to the training data
            self.pipeline.fit(X_train, y_train)
            print(f"Training completed for {self.model_name}.")
        except Exception as e:
            # Catch potential errors during fitting
            print(f"Error during training of {self.model_name}: {e}")
            self.pipeline = None # Ensure pipeline is None if training failed

    def evaluate(self, X_test, y_test):
        """
        Evaluates the trained pipeline on the test data.

        Calculates and prints standard classification metrics, plots the
        confusion matrix, and returns a dictionary of key results.

        Args:
            X_test (pd.DataFrame or np.ndarray): Test features.
            y_test (pd.Series or np.ndarray): True target values for the test set.

        Returns:
            dict or None: A dictionary containing key evaluation metrics ('AUC',
                          'Avg Precision', 'Classification Report') if successful,
                          otherwise None.
        """
        # Check if the model was trained successfully
        if self.pipeline is None:
            print(f"ERROR: The model {self.model_name} has not been trained yet (pipeline is None).")
            return None

        print(f"\n--- Evaluating Model: {self.model_name} ---")
        try:
            # Make predictions on the test set
            y_pred = self.pipeline.predict(X_test)

            # Predict probabilities for AUC/AP calculation, if possible
            y_pred_proba = None
            if hasattr(self.pipeline, "predict_proba"):
                 # Probability of the positive class (usually class 1)
                 y_pred_proba = self.pipeline.predict_proba(X_test)[:, 1]
            else:
                 print(f"Info ({self.model_name}): Model does not support predict_proba.")

            # Print standard classification report (Precision, Recall, F1)
            print("\nClassification Report:")
            # Use target_names for better readability if needed, e.g., target_names=['No Fraud', 'Fraud']
            print(classification_report(y_test, y_pred))

            # Plot the confusion matrix
            print("Confusion Matrix:")
            self._plot_confusion_matrix(y_test, y_pred)

            # Calculate AUC and Average Precision if probabilities are available
            auc_score = 'N/A'
            ap_score = 'N/A'
            if y_pred_proba is not None:
                try:
                    auc_score = roc_auc_score(y_test, y_pred_proba)
                    ap_score = average_precision_score(y_test, y_pred_proba)
                    print(f"\nAUC-ROC Score: {auc_score:.4f}")
                    print(f"Average Precision Score (PR AUC): {ap_score:.4f}")
                except ValueError as ve_met:
                     # Handle cases where metric calculation fails (e.g., only one class in y_test)
                     print(f"Warning: Could not calculate AUC/AP scores. Error: {ve_met}")
                     auc_score = 'Calculation Error'
                     ap_score = 'Calculation Error'
            else:
                print("\nAUC-ROC Score: N/A (predict_proba not available)")
                print("Average Precision Score (PR AUC): N/A (predict_proba not available)")


            # Return key metrics in a dictionary
            return {
                'model_name': self.model_name,
                'AUC': auc_score,
                'Avg Precision': ap_score,
                'Classification Report': classification_report(y_test, y_pred, output_dict=True) # Return dict for parsing
            }
        except Exception as e:
            # Catch potential errors during prediction or evaluation
            print(f"Error during evaluation of {self.model_name}: {e}")
            return None

    def _plot_confusion_matrix(self, y_true, y_pred):
        """
        Private helper method to plot the confusion matrix using seaborn.

        Args:
            y_true (pd.Series or np.ndarray): True target values.
            y_pred (pd.Series or np.ndarray): Predicted target values.
        """
        try:
            cm = confusion_matrix(y_true, y_pred)
            # Use seaborn for a visually appealing heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['No Fraud', 'Fraud'], yticklabels=['No Fraud', 'Fraud'])
            plt.title(f'Confusion Matrix - {self.model_name}')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.show()
        except Exception as e:
            print(f"Error plotting confusion matrix for {self.model_name}: {e}")


    def get_feature_importance(self):
        """
        Calculates and returns feature importance from the trained model.

        Handles both tree-based models (using `.feature_importances_`) and
        linear models (using `.coef_`). Does not print the results directly.

        Returns:
            pd.DataFrame or None: A DataFrame containing 'Feature' and 'Importance'
                                  (or 'Coefficient') sorted by importance/magnitude,
                                  or None if importance cannot be determined or an error occurs.
        """
        # Check if model is trained and feature names are available
        if self.pipeline is None or 'classifier' not in self.pipeline.named_steps:
            print(f"ERROR ({self.model_name}): Model not trained or classifier step is missing.")
            return None
        if self.feature_names_in_ is None:
             print(f"ERROR ({self.model_name}): Feature names were not recorded during training.")
             return None

        model = self.pipeline.named_steps['classifier'] # Get the actual classifier model
        feature_names = self.feature_names_in_ # Get the stored feature names
        importance_df = None

        try:
            # Check for tree-based model importance attribute
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                # Sort by importance value, descending
                importance_df = importance_df.sort_values(by='Importance', ascending=False)

            # Check for linear model coefficient attribute
            elif hasattr(model, 'coef_'):
                # coef_ typically has shape (n_classes-1, n_features) for binary or (n_classes, n_features) for multi-class
                # For binary, we usually take the first row (index 0)
                if model.coef_.shape[0] == 1:
                     coefficients = model.coef_[0]
                else:
                     # Handle multi-class case if needed, or assume binary focus
                     print(f"Warning ({self.model_name}): Model coefficients have shape {model.coef_.shape}. Using coefficients for the positive class (index 1 assumed).")
                     # Adjust index if needed based on model output or stick to index 0/1
                     coefficients = model.coef_[1] if model.classes_[1] == 1 else model.coef_[0]


                # Verify dimensions match
                if len(coefficients) != len(feature_names):
                     print(f"Warning ({self.model_name}): Number of coefficients ({len(coefficients)}) does not match number of features ({len(feature_names)}). Cannot reliably assign feature names.")
                     return None

                importance_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
                # Add absolute coefficient for sorting by magnitude
                importance_df['Abs_Coefficient'] = np.abs(importance_df['Coefficient'])
                # Sort by absolute magnitude, descending
                importance_df = importance_df.sort_values(by='Abs_Coefficient', ascending=False)

            else:
                # Model type doesn't have standard importance attributes
                print(f"INFO ({self.model_name}): This model type does not provide standard '.feature_importances_' or '.coef_' attributes.")
                return None

            # Return the resulting DataFrame
            return importance_df

        except Exception as e:
            # Catch any errors during importance calculation
            print(f"Error calculating feature importance for {self.model_name}: {e}")
            return None
