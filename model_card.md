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

The model was evaluated using F1 score with average parameter set as "weighted".
The results on the test set are:
| Precision   | Recall      | F1 score    |
| ----------- | ----------- | ----------- |
| 0.8599      | 0.8654      | 0.8603      |

## Ethical Considerations

### Social Data: Biases, Methodological Pitfalls, and Ethical Boundaries
#### REVIEW article
#### Front. Big Data, 11 July 2019 | https://doi.org/10.3389/fdata.2019.00013

Previous sections can be seen as covering what are ultimately ethical issues that Mittelstadt et al. (2016) calls epistemic concerns (sections 3–8), such as using evidence that is inconclusive or misguided. In contrast, this section deals with normative concerns, related mostly to the consequences of research.

Research on human subjects is regulated by law in many jurisdictions; and given that data elements in social datasets represent people or groups of people (Varshney, 2015; Diaz, 2016), research on social data is, arguably, human subjects research (Metcalf and Crawford, 2016). The fact that social data is often publicly accessible does not mean research done on it is ethical (Zimmer, 2010; boyd and Crawford, 2012). As a result, both scientists (Dwork and Mulligan, 2013; Barocas and Selbst, 2016) and journalists (Hill, 2014; Kirchner, 2015) have pressed for greater scrutiny of the use of social data against possible ethical pitfalls, such as breaching users privacy (Goroff, 2015), or enabling racial, socioeconomic or gender-based profiling (Barocas and Selbst, 2016).

Such ethical issues have been further highlighted by recent cases, including the Facebook contagion experiment (performed in early 2012 and published in late 2014), where researchers manipulated users' social feeds to include more or less of certain kinds of content based on the expressed emotions (Kramer et al., 2014). The experiment was criticized as an intervention that affected the emotional state of unsuspecting users, who had not given consent to participate in the study (Hutton and Henderson, 2015a). Another example is the Encore research project and how it measured web censorship around the world by instructing web browsers to attempt downloads of sensitive web content without users' knowledge or consent (Burnett and Feamster, 2015), potentially putting people in some countries at risk of harm due to these attempted accesses. In an unprecedented move, the Program Committee (PC) of SIGCOMM 2015 decided to accept the Encore research paper on the condition of placing a prominent note at the top of the paper highlight the PC's ethical concerns (Narayanan and Zevenbergen, 2015).23

## Caveats and Recommendations
