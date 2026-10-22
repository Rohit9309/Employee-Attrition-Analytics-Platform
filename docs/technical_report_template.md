# Technical Report: Employee Attrition Analytics Platform

## Abstract
Summarize the attrition problem, dataset, model comparison, selected model, dashboard and deployment architecture after running the project.

## 1. Problem statement
Define employee attrition and why HR teams may want aggregate, responsible analytics.

## 2. Dataset and data dictionary
Describe source, record count, target definition, feature types, missingness, duplicates, class distribution, privacy and representativeness limitations.

## 3. Methodology
- Data cleaning and identifier exclusion
- Train/test split with stratification
- Imputation, scaling and categorical encoding
- Classical classifiers, neural network, Naive Bayes and discriminant analysis
- Five-fold cross-validation on training data
- Held-out test metrics

## 4. Evaluation
Insert actual values from `outputs/model_metrics.csv` and `outputs/cross_validation.csv`. Include confusion matrix and ROC curve. Do not invent or copy scores.

## 5. Dashboard and API
Describe KPI calculations, filtering, prediction form, risk bands, API schema and limitations.

## 6. IBM Watson deployment plan
Describe the actual assets, runtime, endpoint, job schedule and deployment version only after these have been configured in IBM Cloud. Clearly label planned vs completed work.

## 7. Business interpretation
Discuss predictive associations cautiously. Avoid claiming causal effects from feature importance.

## 8. Responsible AI, privacy and fairness
Discuss access controls, data minimization, employee transparency, group-wise performance checks, human oversight and the prohibition on automated consequential employment decisions.

## 9. Limitations and future work
Include drift monitoring, calibration, fairness assessment, threshold validation, secure deployment and retraining governance.

## 10. Conclusion
Summarize actual results and next steps.
