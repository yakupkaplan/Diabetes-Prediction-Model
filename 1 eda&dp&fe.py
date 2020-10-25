# DIABETES PREDICTION MODEL

# EXPLORATORY DATA ANALYSIS, DATA PREPROCESSING AND FEATURE ENGINEERING

'''
In this project we will try to predict if the person has diabetes has or not.

Steps for Exploratory Data Analysis, Data Visualization, Data Preprocessing and Feature Engineering:

    - GENERAL / GENERAL OVERVIEW / GENERAL PICTURE
    - NUMERICAL VARIABLE ANALYSIS
        describe with quantiles to see whether there are extraordinary values or not
        Basic visualization by using histograms
    - TARGET ANALYSIS
        Target analysis according to categorical variables --> target_summary_with_cats()
        Target analysis according to numerical variables --> target_summary_with_nums()
    - ANALYSIS OF NUMERCIAL VARIABLES IN COMPARISON WITH EACH OTHER
        scatterplot
        lmplot
        correlation
    - Outlier Analysis
    - Missing Values Analysis
    - New Features Creation
    - Label and One Hot Encoding
    - Standardization
    - Saving the Dataset
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

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

import warnings
warnings.simplefilter(action="ignore")
pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# Load the dataset
diabetes = pd.read_csv(r'C:\Users\yakup\PycharmProjects\dsmlbc\datasets\diabetes.csv')
df = diabetes.copy()
df.head()

# Beacuse we do not have many variables, we can follow the same procedure for each variable.
# Note that all the variables are numerical.

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


## NUMERICAL VARIABLES ANALYSIS

# Plot histograms for the dataset
df.hist(bins=20, figsize=(15, 15), color='r')
plt.show()


# Function to plot histograms for numerical variables
def hist_for_nums(data, numeric_cols):
    col_counter = 0
    data = data.copy()
    for col in numeric_cols:
        data[col].hist(bins=20)
        plt.xlabel(col)
        plt.title(col)
        plt.show()
        col_counter += 1
    print(col_counter, "variables have been plotted")


hist_for_nums(df, df.columns)


## TARGET ANALYSIS

df["Outcome"].value_counts()

# See how many 0 and 1 values in the dataset and if there is imbalance
sns.countplot(x='Outcome', data=df)
plt.show()

# Look at the mean and meadian for each variable groupped by Outcome
for col in df.columns:
    print(df.groupby("Outcome").agg({col: ["mean", "median"]}))


## ANALYSIS OF NUMERCIAL VARIABLES IN COMPARISON WITH EACH OTHER

# Show the scatterplots for each variable and add teh dimension for Outcome, so we can differentiate between classes.
# sns.pairplot(df, hue='Outcome');
# plt.show()

# Show the correlation matrix
plt.subplots(figsize=(15,12))
correlation_matrix = df.corr().round(2)
sns.heatmap(data=correlation_matrix, cmap='coolwarm', annot=True, square=True);
plt.show()


## OUTLIER ANALYSIS


# Function to calculate outlier thresholds
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.05)
    quartile3 = dataframe[variable].quantile(0.95)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


# Function to report variables with outliers and return the names of the variables with outliers with a list
def has_outliers(dataframe, num_col_names, plot=False):
    variable_names = []
    for col in num_col_names:
        low_limit, up_limit = outlier_thresholds(dataframe, col)
        if dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis=None):
            number_of_outliers = dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].shape[0]
            print(col, ":", number_of_outliers)
            variable_names.append(col)
            if plot:
                sns.boxplot(x=dataframe[col])
                plt.show()
    return variable_names


has_outliers(df, df.columns)


# Function to reassign up/low limits to the ones above/below up/low limits by using apply and lambda method
def replace_with_thresholds_with_lambda(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe[variable] = dataframe[variable].apply(lambda x: up_limit if x > up_limit else (low_limit if x < low_limit else x))


# Assign outliers thresholds values for all the numerical variables
for col in df.columns:
    replace_with_thresholds_with_lambda(df, col)

# Check for outliers, again
has_outliers(df, df.columns)


## MISSING VALUES ANALYSIS

# Check for missing values
df.isnull().values.any()
df.isnull().sum().sort_values(ascending=False)

# It seems, that there are no missing values in our dataset. (That makes me suspicious)
# However, after delving deep into the dataset, we realized that there are actually missing values, but they are written as '0'.

'''
On these columns, a value of zero does not make sense and thus indicates missing value.
Following columns or variables have an invalid zero value:
    - Glucose
    - BloodPressure
    - SkinThickness
    - Insulin
    - BMI
'''

# Let's correct the errors in the dataset.
variables_with_na = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in variables_with_na:
    df[col] = df[col].apply(lambda x: np.NaN if x == 0 else x)
# Check for missing values, again.
df.isnull().sum().sort_values(ascending=False)

# Visualize missing variables
sns.heatmap(df.isnull(), cbar=False, cmap='magma')
plt.show()
# Missing values overall view
msno.bar(df)
plt.show()
# Now, we can see the relationship between missing values
msno.matrix(df)
plt.show()
# Nullity correlation visualization
msno.heatmap(df)
plt.show()

# Impute median values for missing values for numeral variables with respect to their class
# df = df.apply(lambda x: x.fillna(x.median()), axis=0)
for col in variables_with_na:
    df[col] = df[col].fillna(df.groupby("Outcome")[col].transform("median"))

# Check for missing values, again and control
df.isnull().sum()
df.isnull().sum().sum() # 0


## FEATURE CREATION

# Create BMI ranges
df['BMIRanges'] = pd.cut(x=df['BMI'], bins=[0, 18.5, 25, 30, 100], labels=["Underweight", "Healthy", "Overweight", "Obese"])
df['BMIRanges'] = df['BMIRanges'].astype(str)
df.head()

# See the results for the new feature
df.groupby(["Outcome", "BMIRanges"]).describe()
df[['BMIRanges']].value_counts()
df.groupby(["BMIRanges"]).agg({"Outcome": [np.mean, np.count_nonzero, np.size]}) # Super!

# See the counts for each class that we created
sns.countplot(x='BMIRanges', hue='Outcome', data=df)
plt.show()

# Create Age ranges --> young, mid_aged, old
df['Age'].describe()
df['AgeRanges'] = pd.cut(x=df['Age'], bins=[15, 25, 65, 81], labels=["Young", "Mid_Aged", "Senior"])
df['AgeRanges'] = df['AgeRanges'].astype(str)
df.head()
# See the results for the new feature
df.groupby(["Outcome", "AgeRanges"]).describe()
df[['AgeRanges']].value_counts()
df.groupby(["AgeRanges"]).agg({"Outcome": [np.mean, np.count_nonzero, np.size]}) # Super!

# Create Insulin/Glucose ranges --> low, normla, secret, high
df['Glucose'].describe()
df['GlucoseLevels'] = pd.cut(x=df['Glucose'], bins=[0, 70, 99, 126, 200], labels=["Low", "Normal", "Secret", "High"])
df['GlucoseLevels'] = df['GlucoseLevels'].astype(str)
df.head()

# See the results for the new feature
df.groupby(["Outcome", "GlucoseLevels"]).describe()
df[['GlucoseLevels']].value_counts()
df.groupby(["GlucoseLevels"]).agg({"Outcome": [np.mean, np.count_nonzero, np.size]}) # Super!

# Create a feature that shows the ratio of pregnancies/age
df['Pregnancies/Age'] = df['Pregnancies'] / df['Age']
df['Pregnancies/Age'].describe()

# See the results for the new feature
sns.boxplot(x='Outcome', y='Pregnancies', data=df)
plt.show()

df.info()


## LABEL AND ONE HOT ENCODING


# Catch numerical variables
cat_cols = [col for col in df.columns if df[col].dtypes == 'O']
len(cat_cols)


# Define a function to apply one hot encoding to categorical variables.
def one_hot_encoder(dataframe, categorical_cols, nan_as_category=True):
    original_columns = list(dataframe.columns)
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, dummy_na=False, drop_first=True)
    new_columns = [c for c in dataframe.columns if c not in original_columns]
    return dataframe, new_columns


df, new_cols_ohe = one_hot_encoder(df, cat_cols)
df.head()
len(new_cols_ohe)

df.info()

# Export the dataset for later use by modeling
df.to_csv(r'C:\Users\yakup\PycharmProjects\dsmlbc\projects\diabetes_classification\diabetes_prepared.csv', index=False)


# STANDARDIZATION

df.head()

# Catch numerical variables
num_cols = [col for col in df.columns if df[col].dtypes != 'O' and col not in ["Outcome"]]
len(num_cols)

# MinMaxScaler

df_minmax_scaled = df.copy()

from sklearn.preprocessing import MinMaxScaler
transformer = MinMaxScaler()
df_minmax_scaled[num_cols] = transformer.fit_transform(df_minmax_scaled[num_cols])  # default value is between 0 and 1

df_minmax_scaled[num_cols].describe().T
len(num_cols)

# StandardScaler

df_std_scaled = df.copy()

from sklearn.preprocessing import StandardScaler
transformer = StandardScaler()
df_std_scaled[num_cols] = transformer.fit_transform(df_std_scaled[num_cols])

df_std_scaled[num_cols].describe().T
len(num_cols)

# RobustScaler

df_robust_scaled = df.copy()

from sklearn.preprocessing import RobustScaler
transformer = RobustScaler()
df_robust_scaled[num_cols] = transformer.fit_transform(df_robust_scaled[num_cols])

df_robust_scaled[num_cols].describe().T
len(num_cols)

# Check before modeling for missing values and outliers in the dataset
# df.isnull().sum().sum()
# has_outliers(df, num_cols)
df_minmax_scaled.isnull().sum().sum()
has_outliers(df_minmax_scaled, num_cols)

df_std_scaled.isnull().sum().sum()
has_outliers(df_std_scaled, num_cols)

df_robust_scaled.isnull().sum().sum()
has_outliers(df_robust_scaled, num_cols)

# Last look at the dataset
df.head()
df.info()

# Export the dataset for later use by modeling
#df.to_csv(r'C:\Users\yakup\PycharmProjects\dsmlbc\projects\diabetes_classification\diabetes_prepared.csv')
df_minmax_scaled.to_csv(r'C:\Users\yakup\PycharmProjects\dsmlbc\projects\diabetes_classification\diabetes_prepared_minmaxscaled.csv', index=False)
df_std_scaled.to_csv(r'C:\Users\yakup\PycharmProjects\dsmlbc\projects\diabetes_classification\diabetes_prepared_stdscaled.csv', index=False)
df_robust_scaled.to_csv(r'C:\Users\yakup\PycharmProjects\dsmlbc\projects\diabetes_classification\diabetes_prepared_robustscaled.csv', index=False)



