# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# Redefine the StrokePrediction class with separate train and evaluate methods
class StrokePrediction(BaseEstimator, TransformerMixin):
    """
    StrokePrediction class handles preprocessing, undersampling, model training, model evaluation, and prediction.
    """

    def __init__(self, models, n_features=None, balance_ratio=3):
        """
        Initializes the StrokePrediction object.

        Parameters:
        models (list): A list of tuples. Each tuple contains a string (the name of the model) and
        an instance of a sklearn estimator.
        n_features (int): The number of features to select for training the model. Default is None.
        balance_ratio (int): The ratio to use for undersampling. Default is 3.
        """
        self.n_features = n_features
        self.balance_ratio = balance_ratio
        self.models = models

        if self.n_features is not None:
            self.sfm = SelectFromModel(
                RandomForestClassifier(random_state=42),
                threshold=-np.inf,
                max_features=self.n_features,
            )

    def preprocess(self, X, y=None):
        """
        Preprocesses the data by applying the appropriate pipeline to numerical and categorical columns.

        Parameters:
        X (DataFrame): The features to preprocess.
        y (Series): The target variable. If given, the method will fit the feature selection model.

        Returns:
        X_processed (np.array): The preprocessed features.
        """
        print("Preprocessing data...")
        # Convert NumPy arrays to Pandas DataFrames
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.columns_)
        # Define the preprocessing steps
        num_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("std_scaler", StandardScaler()),
            ]
        )
        cat_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        # Get the list of numerical and categorical columns
        self.num_columns_ = (
            X.select_dtypes(include=[np.number]).columns
            if y is not None
            else self.num_columns_
        )
        self.cat_columns_ = (
            X.select_dtypes(include=[np.object]).columns
            if y is not None
            else self.cat_columns_
        )

        # Apply each pipeline
        if y is not None:  # If y is given (training phase), fit the pipelines
            X_num = num_pipeline.fit_transform(X[self.num_columns_])
            X_cat = cat_pipeline.fit_transform(
                X[self.cat_columns_]
            ).toarray()  # Convert sparse matrix to numpy array
            self.num_pipeline_ = num_pipeline
            self.cat_pipeline_ = cat_pipeline
        else:  # If y is not given (testing phase), transform the data using the fitted pipelines
            X_num = self.num_pipeline_.transform(X[self.num_columns_])
            X_cat = self.cat_pipeline_.transform(X[self.cat_columns_]).toarray()

        # Combine the numerical and categorical data
        X_processed = np.hstack((X_num, X_cat))

        # If y is given (training phase), fit the feature selection model
        if y is not None and self.n_features is not None:
            print("Fitting feature selection model...")
            self.sfm.fit(X_processed, y)
        # Transform the data using the fitted feature selection model
        if self.n_features is not None:
            X_processed = self.sfm.transform(X_processed)

        return X_processed

    def undersample(self, X, y):
        """
        Undersamples the data.

        Parameters:
        X (np.array): The features to undersample.
        y (np.array): The target variable.

        Returns:
        X_undersampled (np.array), y_undersampled (np.array): The undersampled features and target variable.
        """

        print("Undersampling data...")
        # Separate the 'No Stroke' and 'Stroke' instances
        X_no_stroke = X[y == 0]
        X_stroke = X[y == 1]

        # Compute the nearest 'Stroke' neighbor for each 'No Stroke' instance
        neighbors = NearestNeighbors(n_neighbors=1).fit(X_stroke)
        distances, indices = neighbors.kneighbors(X_no_stroke)

        # Keep a 1:self.balance_ratio ratio of 'Stroke':'No Stroke' instances
        num_no_stroke = len(X_stroke) * self.balance_ratio
        idx_closest = np.argsort(distances.ravel())[:num_no_stroke]
        X_no_stroke_closest = X_no_stroke[idx_closest]

        # Combine the 'No Stroke' and 'Stroke' instances to get the undersampled data
        X_undersampled = np.vstack((X_no_stroke_closest, X_stroke))
        y_undersampled = np.hstack((np.zeros(num_no_stroke), np.ones(len(X_stroke))))
        return X_undersampled, y_undersampled

    def train_model(self, X_train, y_train):
        """
        Trains the models.

        Parameters:
        X_train (np.array): The features to train the model on.
        y_train (np.array): The target variable for training.

        Returns:
        self.voting_clf (VotingClassifier): The trained model.
        """

        print("Training models...")
        for name, model in self.models:
            print(f"Training {name}...")
            model.fit(X_train, y_train)

        # Create a new Voting Classifier
        self.voting_clf = VotingClassifier(
            estimators=self.models,
            voting="soft",
        )

        # Train the Voting Classifier
        self.voting_clf.fit(X_train, y_train)

        return self.voting_clf

    def train(self, X, y, oversample=True):
        """
        Trains the model. This includes preprocessing the data, undersampling (if needed), and training the model.

        Parameters:
        X (DataFrame): The features to train the model on.
        y (Series): The target variable for training.
        oversample (bool): Whether to oversample the data. Default is True.

        Returns:
        self (StrokePrediction): The trained model.
        """

        self.columns_ = X.columns
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train_processed = self.preprocess(X_train, y_train)
        X_test_processed = self.preprocess(X_test)
        print(len(X_train_processed), len(X_test_processed))
        if oversample:
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(
                X_train_processed, y_train
            )  # apply SMOTE only on the training data
        else:
            X_train_res, y_train_res = self.undersample(X_train_processed, y_train)
        self.model_ = self.train_model(X_train_res, y_train_res)
        self.X_test = X_test  # Save the original test data
        self.X_test_processed = X_test_processed  # Save the preprocessed test data
        self.y_test = y_test
        return self

    def evaluate(self):
        """
        Evaluates the model. This includes making predictions on the test data and printing the confusion matrix and
        classification report.

        Returns:
        self (StrokePrediction): The evaluated model.
        """
        print("Evaluating the model...")
        y_pred = self.model_.predict(self.X_test_processed)

        # Convert y_pred to int
        y_pred = y_pred.astype(int)

        # Compute the confusion matrix and the classification report
        cm = confusion_matrix(self.y_test, y_pred)
        cr = classification_report(
            self.y_test, y_pred, target_names=["No Stroke", "Stroke"]
        )

        # Print the confusion matrix and the classification report
        print("Confusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(cr)

        # Create a DataFrame with the IDs, true labels, and predicted labels
        predictions_df = pd.DataFrame({
            'ID': self.X_test.index,
            'y_true': self.y_test,
            'y_pred': y_pred
        })

        # Save the DataFrame to a CSV file
        predictions_df.to_csv('../predictions/stroke_classifier_eval_preds.csv', index=False)

        return self


    def predict(self, X):
        """
        Makes predictions on the given data.

        Parameters:
        X (DataFrame or np.array): The data to make predictions on.

        Returns:
        y_pred (np.array): The predicted classes.
        """
        print("Predicting classes...")
        try:
            getattr(self, "model_")
        except AttributeError:
            raise RuntimeError("You must train the model before predicting.")
        X_processed = self.preprocess(X)
        y_pred = self.model_.predict(X_processed)

        return y_pred


if __name__ == "__main__":
    # Load the dataset
    data = pd.read_csv("../data/healthcare-dataset-stroke-data.csv")

    # Separate the features and the target
    X = data.drop(["id", "stroke"], axis=1)
    y = data["stroke"]

    models = [
        ("lr", LogisticRegression(class_weight="balanced", random_state=42)),
        ("rf", RandomForestClassifier(class_weight="balanced", random_state=42)),
        ("xgb", XGBClassifier(use_label_encoder=False, random_state=42)),
    ]

    # Create an instance of the StrokePrediction class
    stroke_pred = StrokePrediction(models=models, n_features=10)

    # Train the model and make predictions
    stroke_pred.train(X, y, oversample=False)

    # Evaluate the model on the test data
    stroke_pred.evaluate()

    # Predict the classes for the test data
    y_test_pred = stroke_pred.predict(stroke_pred.X_test)

    # Print the predicted classes
    print(y_test_pred)
