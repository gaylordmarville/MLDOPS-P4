from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def find_model_and_params(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    # Create the parameter grid based on the results of random search
    param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }

    # Create a based model
    clf = RandomForestClassifier()

    l_cv = [4, ]

    for cv in l_cv:

        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid,
                                   cv=cv, n_jobs=-1, verbose=2)

        # Fit the grid search to the data
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_

    return RandomForestClassifier, best_params


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train, params_tuning=False):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    if params_tuning:
        model, best_parameters = find_model_and_params(X_train, y_train)
        print(best_parameters)
        clf = model(**best_parameters)
        clf.fit(X_train, y_train)
    else:
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)

    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision,
    recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    y_pred = model.predict(X)

    return y_pred
