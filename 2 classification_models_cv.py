# DIABETES PREDICTION MODEL

# CLASSIFICATION MODELS - CROSS VALIDATION

'''
Steps to follow:
    - Load the saved dataset
    - General View
    - Modeling
        - Base models: LogisticRegression, GaussianNB, KNeighborsClassifier, SVC, MLPClassifier, DecisionTreeClassifier,
                       BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,
                       XGBClassifier, LGBMClassifier, CatBoostClassifier, NGBClassifier
    - Model Evaluation
    - Model Tuning
    - Model Results
'''

# Import dependencies
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from ngboost import NGBClassifier

import os
import pickle

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import ClassPredictionError

import warnings
warnings.simplefilter(action="ignore")
pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# Load the preprocessed dataset
diabetes_preprocessed = pd.read_csv(r'C:\Users\yakup\PycharmProjects\dsmlbc\projects\diabetes_classification\diabetes_prepared.csv')
df = diabetes_preprocessed.copy()


## GENERAL VIEW

df.head()
df.shape
df.info()
df.columns
df.index
df.describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T

# Check for missing values
df.isnull().values.any()
df.isnull().sum().sort_values(ascending=False)


# MODELING


# Define dependent and independent variables
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)


# Evaluate each model in turn by looking at cross validation scores
def evaluate_classification_model_cross_validation(models, X, y):
    # Define lists to track names and results for models
    names = []
    train_accuracy_results = []
    cross_validation_scores = []

    print('################ Accuracy scores (cross-validation=10) for test set for the models: ################\n')
    for name, model in models:
        model.fit(X, y)
        y_pred = model.predict(X)

        train_accuracy_result = accuracy_score(y, y_pred)
        cross_validation_score = cross_val_score(model, X, y, cv=10).mean()
        train_accuracy_results.append(train_accuracy_result)
        cross_validation_scores.append(cross_validation_score)

        names.append(name)
        msg = "%s: %f" % (name, cross_validation_score)
        print(msg)

    print('\n################ Train and test results for the model: ################\n')
    data_result = pd.DataFrame({'models': names,
                                'accuracy_train': train_accuracy_results,
                                'cross_val_score': cross_validation_scores})
    print(data_result)

    # Plot the results
    plt.figure(figsize=(15, 12))
    sns.barplot(x='cross_val_score', y='models', data=data_result.sort_values(by="cross_val_score", ascending=False), color="r")
    plt.xlabel('Accuracy Scores')
    plt.ylabel('Models')
    plt.title('Accuracy Scores For Test Set')
    plt.show()

    # # Boxplot algorithm comparison
    # fig = plt.figure(figsize=(15, 10))
    # fig.suptitle('Algorithm Comparison')
    # ax = fig.add_subplot(111)
    # plt.boxplot(cross_validation_scores)
    # ax.set_xticklabels(names)
    # plt.show()


# Define a function to plot feature_importances
def plot_feature_importances(tuned_model):
    feature_imp = pd.Series(tuned_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    sns.barplot(x=feature_imp, y=feature_imp.index)
    plt.xlabel('Significance Score Of Variables')
    plt.ylabel('Variables')
    plt.title("Feature Importances")
    plt.show()


# Function to plot confusion_matrix
def plot_confusion_matrix(model, X_test, y_test, normalize=True):
    plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues, normalize=normalize)
    plt.show()


# Funtion to plot ROC-AUC Curve
def plot_roc_auc_curve(model):
    model_roc_auc = roc_auc_score(y, model.predict(X))
    fpr, tpr, thresholds = roc_curve(y, model.predict_proba(X)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % model_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


# See the results for base models
base_models = [('LogisticRegression', LogisticRegression()),
               ('Naive Bayes', GaussianNB()),
               ('KNN', KNeighborsClassifier()),
               ('SVM', SVC()),
               ('ANN', MLPClassifier()),
               ('CART', DecisionTreeClassifier()),
               ('BaggedTrees', BaggingClassifier()),
               ('RF', RandomForestClassifier()),
               ('AdaBoost', AdaBoostClassifier()),
               ('GBM', GradientBoostingClassifier()),
               ("XGBoost", XGBClassifier()),
               ("LightGBM", LGBMClassifier()),
               ("CatBoost", CatBoostClassifier(verbose=False)),
               ("NGBoost", NGBClassifier(verbose=False))]

evaluate_classification_model_cross_validation(base_models, X, y)
# LogisticRegression: 0.773462
# Naive Bayes: 0.585800
# KNN: 0.851538
# SVM: 0.847556
# ANN: 0.837218
# CART: 0.845147
# BaggedTrees: 0.881562
# RF: 0.880229
# AdaBoost: 0.864610
# GBM: 0.886791
# XGBoost: 0.877683
# LightGBM: 0.877700

# For some distance-based models, we need to scale the features in order to
# increase the speed and the performance/accuracy level of the model.
# We can also see the effect of scaling type by changing the sclaer function.
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

# LogisticRegression with make_pipeline
logreg = make_pipeline(MinMaxScaler(), LogisticRegression())
evaluate_classification_model_cross_validation([('LogisticRegression', logreg)], X, y) # 0.776 instead of 0.773462

# KNN with make_pipeline
knn = make_pipeline(RobustScaler(), KNeighborsClassifier())
evaluate_classification_model_cross_validation([('KNN', knn)], X, y) # 0.818 instead of 0.851538

# SVC with make_pipeline
svc = make_pipeline(RobustScaler(), SVC())
evaluate_classification_model_cross_validation([('SVM', svc)], X, y) # 0.852 instead of 0.847556

# ANN with make_pipeline
ann = make_pipeline(StandardScaler(), MLPClassifier())
evaluate_classification_model_cross_validation([('ANN', ann)], X, y) # 0.845 instead of 0.837218

# NGBClassifier with make_pipeline
ngb = make_pipeline(RobustScaler(), NGBClassifier())
evaluate_classification_model_cross_validation([('NGBoost', ngb)], X, y) #


# MODEL TUNING

'''
Models to be tuned:
    - LogisticRegression
    - RandomForestClassifier
    - LightGBMClassifier
    - XGBClassifier
'''

# LogisticRegression # 0.773462

logreg_model = LogisticRegression(random_state=12345)
logreg_params = {'penalty': ['l1', 'l2'],
                 'C': [0.001, 0.009, 0.01, 0.09, 1, 5, 10, 25]}

logreg_cv_model = GridSearchCV(logreg_model, logreg_params, cv=10, n_jobs=-1, verbose=2).fit(X, y)
logreg_cv_model.best_params_ # {'C': 25, 'penalty': 'l2'}

# Final Model
logreg_tuned = LogisticRegression(**logreg_cv_model.best_params_).fit(X,y)
cross_val_score(logreg_tuned, X, y, cv=10).mean() # 0.7786739576213261

# Visualization of Results --> Feature Importances
confusion_matrix(y, logreg_tuned.predict(X))
plot_roc_auc_curve(logreg_tuned)


# RandomForestClassifier # 0.880229

rf_model = RandomForestClassifier(random_state=12345)
rf_params = {"n_estimators": [100, 200, 500, 1000],
             "max_features": [3, 5, 7],
             "min_samples_split": [2, 5, 10, 30],
            "max_depth": [3, 5, 8, None]}

rf_cv_model = GridSearchCV(rf_model, rf_params, cv=10, n_jobs=-1, verbose=2).fit(X, y)
rf_cv_model.best_params_ # {'max_depth': None, 'max_features': 5, 'min_samples_split': 2, 'n_estimators': 500}

# Final Model
rf_tuned = RandomForestClassifier(**rf_cv_model.best_params_).fit(X,y)
cross_val_score(rf_tuned, X, y, cv=10).mean() # 0.886790840738209

# Visualization of Results --> Feature Importances
plot_feature_importances(rf_tuned)
confusion_matrix(y, rf_tuned.predict(X))
plot_roc_auc_curve(rf_tuned)


# XGBClassifier # 0.877683

xgb_model = XGBClassifier()
xgb_params = {"learning_rate": [0.1, 0.01, 1],
             "max_depth": [2, 5, 8],
             "n_estimators": [100, 500, 1000],
             "colsample_bytree": [0.3, 0.6, 1]}

xgb_cv_model = GridSearchCV(xgb_model, xgb_params, cv=10, n_jobs=-1, verbose=2).fit(X, y)
xgb_cv_model.best_params_ # {'colsample_bytree': 0.6, 'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 100}

# Final Model
xgb_tuned = XGBClassifier(**xgb_cv_model.best_params_).fit(X,y)
cross_val_score(xgb_tuned, X, y, cv=10).mean() # 0.895933014354067

# Visualization of Results --> Feature Importances
plot_feature_importances(xgb_tuned)
confusion_matrix(y, xgb_tuned.predict(X))
plot_roc_auc_curve(xgb_tuned)


# LightGBMClassifier # 0.877700

lgbm_model = LGBMClassifier()
lgbm_params = {"learning_rate": [0.01, 0.05, 0.1],
              "n_estimators": [200, 500, 1000],
              "max_depth":[5,8,10],
              "colsample_bytree": [1,0.5,0.3]}

lgbm_cv_model = GridSearchCV(lgbm_model, lgbm_params, cv=10, n_jobs=-1, verbose=2).fit(X, y)
lgbm_cv_model.best_params_ # {'colsample_bytree': 0.05, 'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 800}

# Final Model
lgbm_tuned = LGBMClassifier(**lgbm_cv_model.best_params_).fit(X, y)
cross_val_score(lgbm_tuned, X, y, cv=10).mean() # 0.8971804511278195

# Visualization of Results --> Feature Importances
plot_feature_importances(lgbm_tuned)
confusion_matrix(y, lgbm_tuned.predict(X))
plot_roc_auc_curve(lgbm_tuned)


# Comparison of tuned models

tuned_models = [('LogisticRegression', logreg_tuned),
                ('RF', rf_tuned),
                ('XGBoost', xgb_tuned),
                ('LightGBM', lgbm_tuned)]

# evaluate each model in turn
results = []
names = []

for name, model in tuned_models:
    kfold = KFold(n_splits=10, random_state=12345)
    cv_results = cross_val_score(model, X, y, cv=10, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# LogisticRegression: 0.778674 (0.038794)
# RF: 0.884176 (0.033919)
# XGBoost: 0.895933 (0.027015)
# LightGBM: 0.897180 (0.029169)


# boxplot algorithm comparison
fig = plt.figure(figsize=(15, 10))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


evaluate_classification_model_cross_validation(tuned_models, X, y)


