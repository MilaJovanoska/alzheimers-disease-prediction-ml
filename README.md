# Alzheimer’s Disease Classification with Machine Learning

This project compares machine-learning models for classifying Alzheimer’s diagnosis labels from demographic, lifestyle, clinical, and cognitive-assessment features. It explores feature selection, class-imbalance handling, and the trade-off between identifying positive labels and producing false positives.

Developed for a **Machine Learning course**, it asks:

> How do different classifiers balance positive-class recall and precision on this dataset?

**Recorded finding:** The Random Forest configuration has the highest saved accuracy (**approximately 95%**) and ROC-AUC (**0.9423**). These exploratory scores are affected by the evaluation issues documented below.

[Explore the analysis notebook](notebooks/Predicting_Alzheimer’s_Disease_Using_Machine_Learning_A_Classification_Approach.ipynb)

**Project status:** Notebook-based educational analysis. The task classifies existing labels; future disease prediction and clinical use have not been validated.

## Contents

- [Dataset and Prediction Task](#dataset-and-prediction-task)
- [Analysis Workflow](#analysis-workflow)
- [Models and Training](#models-and-training)
- [Recorded Results](#recorded-results)
- [Repository and Technologies](#repository-and-technologies)
- [Explore or Run the Project](#explore-or-run-the-project)
- [Implementation Issues and Study Limitations](#implementation-issues-and-study-limitations)
- [Future Improvements](#future-improvements)
- [Author and References](#author-and-references)

## Dataset and Prediction Task

The notebook expects `alzheimers_disease_data.csv`. These details come from its **saved inspection outputs**; the CSV has not been independently checked.

| Attribute | Recorded value |
|---|---|
| Records | 2,149 |
| Columns | 35 |
| Target | `Diagnosis` |
| Negative label | `0`: 1,389 records, approximately 64.63% |
| Positive label | `1`: 760 records, approximately 35.37% |
| Missing values | None reported |
| Initial candidate predictors | 32, after excluding the target and two metadata columns |
| Selected predictors | 20 in the saved run |
| Final training/test split | Stratified 80% / 20%, with `random_state=42` |
| Final test set | 430 records: 278 negative and 152 positive labels |

Predictors cover demographics, lifestyle, medical history, blood pressure, cholesterol, cognitive and functional assessments, and symptoms. Examples include age, BMI, MMSE, activities of daily living (ADL), and memory complaints. `PatientID` and `DoctorInCharge` are excluded; `Diagnosis` is the target.

**Data availability:** The CSV is not tracked in the repository. The notebook links to an unrelated stroke dataset, so the original source, version, and collection or generation process need confirmation. Do not assume these are independently verified clinical records.

## Analysis Workflow

1. Inspect the data and explore distributions, class balance, and correlations.
2. Combine the top 15 mutual-information features, top 15 Random Forest importance features, and 10 features selected by recursive feature elimination (RFE). Their union contains 20 predictors in the saved run.
3. Create the final stratified 80/20 split.
4. Train classifiers, applying scaling and imbalance handling in selected experiments.
5. Tune selected models using five-fold `GridSearchCV` with positive-class recall as the objective.
6. Evaluate predictions using classification reports, confusion matrices, and ROC-AUC.

This is the existing execution order. Selecting features before the final split introduces leakage risk, as explained under [Implementation Issues and Study Limitations](#implementation-issues-and-study-limitations).

## Models and Training

| Model | Implemented configuration |
|---|---|
| Linear Discriminant Analysis (LDA) | Grid search over solver and shrinkage; separate adjusted-prior experiment |
| Quadratic Discriminant Analysis (QDA) | Baseline classifier |
| Gaussian Naive Bayes | Baseline and an adjusted-prior variant using `[0.3, 0.7]` |
| Logistic Regression | Grid search on standardized training features; separate `class_weight="balanced"` model |
| Support Vector Machine (SVM) | Linear kernel, balanced class weights, and probability estimates |
| k-Nearest Neighbours (kNN) | Standardization, SMOTE, and grid search over neighbours, weighting, and distance metric |
| Random Forest | Balanced class weights and grid search over tree count, depth, and minimum split size |
| Neural network | TensorFlow/Keras dense network with standardization and SMOTE |

The neural network uses **64- and 32-unit ReLU layers**, dropout **0.3**, and a sigmoid output. Training uses Adam, binary cross-entropy, batch size 16, up to 50 epochs, and early stopping on validation loss with patience 5 and best-weight restoration.

SMOTE generates synthetic minority-class training examples for kNN and the neural network. Other experiments use class weights or adjusted priors.

## Recorded Results

**Evidence:** Results below are transcribed from the saved notebook outputs, not a fresh training run. The original CSV and end-to-end reproducibility remain unverified.

Precision measures how often predicted positives are correct; recall measures how many positive labels are identified. F1 combines precision and recall. Accuracy measures overall correct classification, while ROC-AUC measures how well scores rank positive labels above negatives.

Precision, recall, and F1 refer to **class 1**. Values retain the saved reporting precision. Configurations use different preprocessing and imbalance treatments, so this is not a controlled comparison of algorithms alone. Feature-selection leakage also limits every configuration’s evaluation.

| Model/configuration | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| LDA | 0.81 | 0.74 | 0.73 | 0.74 | 0.8826 |
| LDA, adjusted priors | 0.76 | 0.61 | 0.90 | 0.73 | — |
| QDA | 0.81 | 0.71 | 0.77 | 0.74 | 0.8689 |
| Gaussian Naive Bayes | 0.77 | 0.68 | 0.69 | 0.68 | 0.8505 |
| Naive Bayes, adjusted priors | 0.73 | 0.58 | 0.90 | 0.70 | 0.8505 |
| Logistic Regression, balanced weights¹ | 0.81 | 0.69 | 0.86 | 0.76 | 0.8837 |
| Linear SVM, balanced weights | 0.81 | 0.69 | 0.84 | 0.76 | 0.8800 |
| kNN with SMOTE | 0.75 | 0.60 | 0.82 | 0.69 | 0.8302 |
| Neural network with SMOTE | 0.88 | 0.82 | 0.84 | 0.83 | 0.9155 |
| **Random Forest, balanced weights** | **0.95** | **0.95** | **0.91** | **0.93** | **0.9423** |

“—” means an ROC-AUC output was not recorded for that configuration.

¹ The balanced Logistic Regression fit reports a convergence warning.

The separate grid-searched Logistic Regression experiment reports accuracy **0.68** and ROC-AUC **0.6759**, but is excluded from the table because it evaluates an estimator trained on scaled features using unscaled test features. Those scores are not a valid assessment of a consistently preprocessed model.

### What the results show

An always-negative prediction would achieve **64.65% accuracy** on the recorded test set (278 / 430), but **zero positive-class recall**. This baseline is calculated from the saved class counts, not a separately trained model, and shows why accuracy alone is insufficient.

Random Forest has the strongest recorded scores. Its selected parameters are `n_estimators=200`, `max_depth=10`, and `min_samples_split=10`. Adjusted priors improve recall but reduce precision for LDA and Naive Bayes, illustrating the trade-off between detecting more positives and generating more false positives.

## Repository and Technologies

| Path | Purpose |
|---|---|
| `README.md` | Project explanation and execution guide |
| `notebooks/Predicting_Alzheimer’s_Disease_Using_Machine_Learning_A_Classification_Approach.ipynb` | Analysis code, explanations, and saved outputs |
| `.gitignore` | Git exclusion rules |

The project uses **Python, pandas, NumPy, Matplotlib, seaborn, scikit-learn, imbalanced-learn, and TensorFlow/Keras**.

The notebook includes feature-distribution plots, a correlation heatmap, feature-importance plots, confusion matrices, an LDA projection, and neural-network learning curves. Much of its text and some figure labels are in Macedonian.

The repository does not include a dependency lockfile, exported model, or deployed application.

## Explore or Run the Project

### View the existing analysis

Open the [notebook on GitHub](notebooks/Predicting_Alzheimer’s_Disease_Using_Machine_Learning_A_Classification_Approach.ipynb) to read its code and saved plots without installing software. Re-running the analysis requires the original CSV, which is not bundled.

### 1. Clone the repository

```bash
git clone https://github.com/MilaJovanoska/alzheimers-disease-prediction-ml.git
cd alzheimers-disease-prediction-ml
```

### 2. Create an environment

Saved logs indicate Python 3.11. On Linux/macOS with Python 3.11 installed:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell with Python 3.11 and the Python launcher installed:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Core versions below are recorded in saved logs. Other packages remain unpinned, and this setup has not been tested end to end.

```bash
python -m pip install jupyterlab ipykernel pandas matplotlib seaborn numpy==2.0.2 scipy==1.15.3 scikit-learn==1.6.1 imbalanced-learn==0.13.0 tensorflow==2.18.0
```

Install these before running the first cell: some imports precede the notebook’s embedded installation commands.

### 4. Supply the data and launch JupyterLab

Place the original file at `notebooks/alzheimers_disease_data.csv`, preserving its column names. Then run:

```bash
cd notebooks
python -m jupyterlab
```

Select the installed environment’s kernel and execute the notebook in order. Starting in `notebooks/` matches its relative CSV path. A successful execution would still retain the methodological issues below.

## Implementation Issues and Study Limitations

### Implementation issues

- **Feature-selection leakage:** Mutual information uses the full dataset. Random Forest importance and RFE use an earlier split, allowing final test observations to influence selection.
- **Validation contamination:** kNN scaling and SMOTE precede cross-validation rather than occurring within each fold. Neural-network validation uses a split of already resampled data.
- **Logistic Regression preprocessing:** The grid-searched model receives unscaled test features after training on scaled features. The balanced variant also reports non-convergence.
- **Reproducibility:** Some stochastic steps lack fixed seeds, and using a set to combine features does not guarantee stable ordering. Some narrative comments disagree with saved outputs.

### Study limitations
- Numerical treatment of categorical codes and ordinary SMOTE on mixed feature types may produce inappropriate representations.
- Cognitive assessments and symptoms are predictors, so the experiment does not demonstrate early prediction before symptoms.
- There is no external validation, subgroup evaluation, probability calibration, or reported uncertainty interval for the model scores. The results do not establish clinical performance or causal risk factors.

## Future Improvements

1. Confirm the data source, version, provenance, and access conditions.
2. Split before supervised feature selection and use pipelines for fold-specific selection, scaling, and resampling.
3. Correct Logistic Regression preprocessing and convergence; create an independent neural-network validation set before resampling.
4. Use appropriate categorical representations, stable feature ordering, fixed seeds, and pinned dependencies.
5. Re-run evaluation on an untouched test set, assess uncertainty and calibration, and align notebook explanations with the outputs.

These improvements are proposed, not implemented.

## Author and References

**Mila Jovanoska** — Machine Learning course project.

- [GitHub profile](https://github.com/MilaJovanoska)
- [Project notebook](notebooks/Predicting_Alzheimer’s_Disease_Using_Machine_Learning_A_Classification_Approach.ipynb) — source for implementation details and reported results.
- [scikit-learn: Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html) — preprocessing consistency and data leakage.
- [imbalanced-learn: Common pitfalls](https://imbalanced-learn.org/stable/common_pitfalls.html) — resampling and validation leakage.

**Dataset reference:** Pending confirmation of the original source.

