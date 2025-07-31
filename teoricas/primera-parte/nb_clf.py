import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator


# Custom Transformer class for Label Encoding categorical features
class myLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}  # Dictionary to store LabelEncoders for each column

    def fit(self, X, y=None):
        """
        Fit LabelEncoders on the input data.

        Parameters:
            X (DataFrame): Input data to fit the LabelEncoders.
            y : Ignored. Present for compatibility with scikit-learn.

        Returns:
            self : Returns the instance of the transformer.
        """
        for c in X.columns:
            self.encoders[c] = LabelEncoder()  # Create a new LabelEncoder for each column
            self.encoders[c].fit(X[c])  # Fit the LabelEncoder on the column data
        return self

    def transform(self, X, y=None):
        """
        Transform the input data using the fitted LabelEncoders.

        Parameters:
            X (DataFrame): Input data to be transformed.
            y : Ignored. Present for compatibility with scikit-learn.

        Returns:
            X_out (DataFrame): Transformed data with categorical columns replaced by numeric values.
        """
        X_out = pd.DataFrame()  # Create an empty DataFrame to store the transformed data
        for c in X.columns:
            X_out[c] = self.encoders[c].transform(X[c])  # Transform each column using the respective LabelEncoder
        return X_out


def load_data(filepath):
    """
    Load data from a file and preprocess it.

    Parameters:
        filepath (str): Path to the data file.

    Returns:
        X_tr (DataFrame): Training features DataFrame.
        y_tr (Series): Training target Series.
        X_ts (DataFrame): Test features DataFrame.
    """
    # Read data from the specified file and set column names to lowercase with underscores
    df = pd.read_csv(filepath, sep="\t", header=0)
    df.columns = [e.lower().replace(" ", "_") for e in df.columns]

    # Separate training data and test data based on the "mamífero" column
    X_tr = df.loc[df["mamífero"].notna(), df.columns != "mamífero"]  # Training features
    y_tr = df.loc[df["mamífero"].notna(), "mamífero"]  # Training target
    X_ts = df.loc[df["mamífero"].isna(), df.columns != "mamífero"]  # Test features

    return X_tr, y_tr, X_ts


def learn_model(X_tr, y_tr, clf):
    """
    Train the provided classifier on the given training data.

    Parameters:
        X_tr (DataFrame): Training features DataFrame.
        y_tr (Series): Training target Series.
        clf : The classifier to be trained.

    Returns:
        clf : The trained classifier.
    """
    # Fit the classifier on the training data
    clf.fit(X_tr.loc[:, X_tr.columns != "animal"], y_tr)
    return clf


def apply_model(X_ts, clf):
    """
    Make predictions using the trained classifier on the test data.

    Parameters:
        X_ts (DataFrame): Test features DataFrame.
        clf : The trained classifier.

    Returns:
        preds (DataFrame): Predicted probabilities for each class as a DataFrame.
    """
    # Predict probabilities for each class using the trained classifier
    preds = pd.DataFrame(clf.predict_proba(X_ts.loc[:, X_ts.columns != "animal"]))

    # Set column names to match the class labels
    preds.columns = clf.classes_
    return preds


def main():
    # Load the data
    training_set_X, training_set_y, test_set_X = load_data("mamíferos.txt")

    # Create a pipeline for label encoding and Naive Bayes classifier
    learning_algorithm = Pipeline(steps=[
        ('preprocessor', myLabelEncoder()),  # Label encoding transformer
        ('classifier', CategoricalNB())  # Categorical Naive Bayes classifier
    ])

    # Train the model
    trained_model = learn_model(training_set_X, training_set_y, learning_algorithm)

    # Apply the trained model to make predictions
    preds = apply_model(test_set_X, trained_model)

    # Print the predicted probabilities
    print(preds)


if __name__ == "__main__":
    main()
