# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Gaylord Marville created the model. It is random forest classifier using the grid search cross validation in scikit-learn 1.0

## Intended Use

Predict whether income exceeds $50K/yr based on census data. Also known as "Adult" dataset.

## Training Data

The data was obtained from the publicly available Census Bureau data (https://archive.ics.uci.edu/ml/datasets/census+income).

The original data set has 48842 rows and 14 attributes, and a 80-20 split was used to break this into a train and test set. No stratification was done. To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.

## Evaluation Data

The training set was used in the grid search cross validation in scikit-learn. The model was evaluated on the testing set.

## Metrics

The model was evaluated using F1 score. The value is around 0.68.

## Ethical Considerations

## Caveats and Recommendations
