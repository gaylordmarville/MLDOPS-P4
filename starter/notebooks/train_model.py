# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from ml.data import process_data

# Add code to load in the data.
data = pd.read_csv("../data/clean_census.csv")

# Optional enhancement, use K-fold cross validation instead
# of a train-test split.
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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(test,
                                             categorical_features=cat_features,
                                             label="salary",
                                             training=False, encoder=encoder)

print(y_train)     

# Train and save a model.
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X_train, y_train, encoder, lb = process_data(data,
                                             categorical_features=cat_features,
                                             label="salary", training=True)

# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [100, 110, 120],
    'max_features': [5, 6, 7, 8],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [14, 16, 18, 20],
    'n_estimators': [300, 400, 500, 600]
}
# Create a based model
clf = RandomForestClassifier()
# Instantiate the grid search model

l_cv = [4,]

for cv in l_cv:

    grid_search = GridSearchCV(estimator = clf, param_grid = param_grid, 
                            cv = cv, n_jobs = -1, verbose = 10)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    import pickle as pkl

    pkl.dump(grid_search.best_params_, open(f"./best_params_cv{cv}_v4.pkl", "wb"))
