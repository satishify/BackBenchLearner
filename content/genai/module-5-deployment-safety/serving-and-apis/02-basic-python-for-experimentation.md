---
title: "Basic Python for Experimentation"
description: "Basic Python for Experimentation: concepts, examples, and practical notes for learners on BackbenchLearner."
---

Python is ideal for quickly testing prompts, APIs, retrieval pipelines, and evaluation scripts.

## Useful beginner stack

- `requests` for API calls.
- `pandas` for data and evaluation tables.
- `jupyter` for quick experiments.
- `pytest` for repeatable checks.

## Starter workflow

1. Write a script to call model API.
2. Store prompt/output pairs.
3. Run simple quality checks.
4. Iterate prompt and settings.

## ML experimentation mini-flow (from session concepts)

1. Load dataset.
2. Split into train and test sets (for example, 80/20).
3. Scale features when required.
4. Train baseline model.
5. Evaluate with confusion matrix, precision, recall, and F1.

## Useful packages for this flow

- `scikit-learn` for models and evaluation
- `numpy` for arrays and math
- `matplotlib` for plots

## Sklearn starter snippet

```
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```
