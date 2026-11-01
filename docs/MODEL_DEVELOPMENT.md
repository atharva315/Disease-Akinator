# Disease Akinator — Model Development Record

This page is intentionally limited to the two model-development paths: datasets, documented preprocessing, splits, model families, exported inference artifacts, and the accuracy figures reported for the project. It describes the historical models that are currently served by the production application; the Doctor Portal database migration is documented separately.

The record distinguishes facts visible in the available training-code snapshot and application from project-summary figures. Where the original training run cannot be reconstructed from the archived material, the gap is stated rather than filled with guessed parameters.

## 1. Symptom → Disease model

### Dataset

| Item | Project record |
|---|---|
| Dataset file used by the Colab-exported training script | `Final_Augmented_dataset_Diseases_and_Symptoms.csv` |
| Target column | `diseases` |
| Input columns | All columns other than `diseases` |
| Approximate size reported in project summary | ~240,000 records |
| Approximate feature count reported in project summary | 370+ symptom features |
| Approximate label count reported in project summary | 700+ disease labels |
| Exact source URL/version/licence | Not retained in the original Akinator repository |

The available training script reads the CSV from the Colab workspace and separates the `diseases` column from the symptom features. The project summary describes the data as roughly 240K patient/symptom records. Exact row, feature, and class counts for the specific file used to produce the deployed `final_ensemble.pkl` were not saved alongside that artifact.

### Preprocessing and input representation

The available training-code snapshot documents these steps:

1. Read the CSV with pandas.
2. Use `diseases` as the target label and all remaining columns as model inputs.
3. Split the rows into training and test sets.
4. Fit a scikit-learn `SimpleImputer(strategy="constant", fill_value=0)` on the training features only.
5. Apply that fitted imputer to both training and test features.
6. Train and evaluate the candidate classifiers using the resulting feature matrices.

The project README describes the broader work as data cleaning, symptom vectorisation, and feature engineering, but the exact cleaning/vectorisation scripts and their parameters are not in the current public repository. Do not infer additional transformations from the serialized model alone.

At inference time, the Streamlit application uses the saved feature order in `feature_names.pkl`. Answers are converted to the input values used by the application:

| User answer | Numeric value |
|---|---:|
| Yes | 1.0 |
| Probably | 0.8 |
| Probably Not | 0.2 |
| No | 0.0 |

The application starts with a zero-valued feature vector and fills positions as questions are answered. This is inference-time input construction; it should not be confused with a complete record of the original dataset-cleaning pipeline.

### Train/test split and validation

The available Colab-exported training script uses:

- `train_test_split`
- `test_size=0.25` (75% training / 25% test)
- `random_state=42`
- `stratify=None` (the saved code does not stratify the split)

The helper function also calls three-fold `cross_val_score` on up to the first 8,000 training rows when cross-validation is enabled. The script does not create a separate fixed validation partition. The exact counts in the resulting train and test partitions were not saved in the repository.

### Models and ensemble

The available Colab-exported script defines and evaluates:

- **Gaussian Naive Bayes** (`GaussianNB`)
- **Logistic Regression** (`LogisticRegression(max_iter=1500, random_state=42)`)
- **K-Nearest Neighbours** (`KNeighborsClassifier(n_neighbors=75)`)

It then defines a scikit-learn `VotingClassifier` using **soft voting** over those three estimators and fits that ensemble on the training partition. The standalone KNN evaluation in the script uses a subset; the final ensemble is subsequently fitted through the `VotingClassifier` training call.

**Model-count reconciliation:** an older project résumé describes a five-model ensemble consisting of Naive Bayes, Logistic Regression, Gradient Boosting, KNN, and XGBoost. The Colab-exported training script available for this documentation shows a three-estimator soft-voting ensemble (Naive Bayes, Logistic Regression, and KNN); it does not contain the Gradient Boosting or XGBoost estimators. The binary `final_ensemble.pkl` has not been deserialized for this documentation update, so the exact membership of the deployed binary is not independently confirmed. The documentation therefore records both source descriptions instead of presenting the five-model list as verified by the available code.

### Reported result

**Approximately 91% accuracy** is the historical project-reported symptom-model result (README/project summary). The exact evaluation run, test-set count, class-wise report, confusion matrix, and association between this figure and a particular saved training-script revision are not present in the repository. It is therefore reported as a project result, not independently reproduced test performance.

## 2. Skin image → Condition model

### Dataset

The historical project documentation identifies the **Kaggle 20 Skin Diseases Dataset** as the image-data source. The project résumé describes a curated dermatology dataset with **15+ skin-disease classes**.

The current application labels its loaded image model **“ResNet50 (DermNet NZ)”**. The exact Kaggle dataset URL/revision, licence, class list after filtering, and link between that dataset description and the current model label are not recorded in a training manifest. These labels are preserved as the references found in project materials; the repository does not establish whether they refer to the same collection, a curated subset, or a later model revision.

### Preprocessing and class preparation

The historical project notes describe:

- Filtering noisy classes and classes with too few examples.
- Using image augmentation to help balance the retained classes.

The exact filtering thresholds, final class counts, augmentation operations, augmentation parameters, and order of operations were not retained in the public repository.

The production application's **inference-time** image preprocessing is explicit in `app.py`:

1. Convert the uploaded image to RGB.
2. Resize to `160 × 160` pixels.
3. Convert to a `float32` NumPy array.
4. Divide pixel values by `255.0` (scale to `[0, 1]`).
5. Add a batch dimension before model inference.

This documents the serving input contract. It does not establish every transform used during training.

### Split and model

The project describes the image model as a **CNN using transfer learning**. The current UI labels the loaded artifact as **ResNet50 (DermNet NZ)**. The exact backbone construction code, pretrained-weight revision, trainable-layer selection, optimizer, loss, learning rate, batch size, epoch count, callbacks, and checkpoint-selection rule are not included in the Akinator repository.

The project reports an **82% validation accuracy** for the image classifier. The exact train/validation/test proportions, image counts per split, random seed, and split manifest are not available. This figure must remain labelled as validation accuracy; it must not be restated as test accuracy.

## 3. Exported artifacts used by production

All five model/metadata artifacts are tracked through Git LFS. The following byte sizes are recorded in the original artifact inventory.

| File | Format / role | Recorded size |
|---|---|---:|
| `final_ensemble.pkl` | Joblib/scikit-learn serialized symptom ensemble | 664,926,619 bytes |
| `feature_names.pkl` | Joblib-serialized ordered symptom feature list | 7,489 bytes |
| `imputer.pkl` | Joblib-serialized scikit-learn imputer | 11,231 bytes |
| `skin_model.h5` | TensorFlow/Keras HDF5 image-model artifact | 23,804,864 bytes |
| `class_names.pkl` | Joblib-serialized image output-label mapping | 625 bytes |

### How production uses these files

- Symptom inference loads `final_ensemble.pkl` and `feature_names.pkl`.
- The `imputer.pkl` artifact is retained in the repository, but the current `app.py` symptom loader does not load or call it. The application's interactive input vector is already populated with numeric values (unanswered positions remain zero).
- Image inference loads `skin_model.h5` with TensorFlow/Keras and maps its output indices with `class_names.pkl`.
- The hosted runtime uses CPU-only TensorFlow `2.20.0`. This is the deployment runtime pin; the exact TensorFlow/Keras version used to train and export the original HDF5 file was not preserved.

## 4. Accuracy summary

| Model | Project-reported result | Correct interpretation |
|---|---:|---|
| Symptom → Disease | ~91% | Historical accuracy figure; exact evaluation split/report not archived |
| Skin Image → Condition | ~82% | Historical validation-accuracy figure; exact split/report not archived |

These are project-reported model-development results. They are separate from the 31 automated application/database tests, which validate software behaviour rather than predictive accuracy. Neither metric constitutes clinical validation.

## 5. What is included for model handoff

A clean clone of the production repository contains the exported inference artifacts and the metadata required by the live application. Git LFS must be installed and the LFS objects pulled before model loading.

The repository does not include the raw medical datasets or the full image/symptom training workspaces. The documentation above is the model-development record supported by the available Colab-exported code, historical project summary, application inference code, and model-artifact inventory. It does not fabricate missing dataset versions, image split counts, augmentation settings, hyperparameters, training curves, or per-class metrics.
