# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from ml.data import process_data, load_data
from ml.model import train_model, compute_model_metrics, inference
import joblib

data_path = "../data/clean_census.csv"

# Add code to load in the data.
data = load_data(data_path)

# Even if we use used the K-fold cross validation
# we keep a test set to evaluate the model later
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Preprocess the entire dataset for k-fold validation.
X_train, y_train, encoder, lb = process_data(train,
                                             categorical_features=cat_features,
                                             label="salary", training=True)


X_test, y_test, _, _ = process_data(test,
                                    categorical_features=cat_features,
                                    label="salary", training=False, encoder=encoder, lb=lb)


# Train and save a model.
model = train_model(X_train, y_train, params_tuning=True)

# Model performance evaluation on categorical features
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

print(f"Precision:{precision}, Recall:{recall}, Fbeta score:{fbeta}")

# joblib.dump(model, 'best_model.joblib') 