# The program generates Decision Trees after cleaning the information
# and using Model Evaluation to check the correctness of the information

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

df = pd.read_csv("votersdata.csv")

# Q1
RSEED = 123
# print(df.head())
# Q2- Knowing the data
# Q2.a- stacked bar plots for the categorical variables
my_crosstab_status = pd.crosstab(df["status"], df["vote"])
my_crosstab_status.plot.bar(stacked=True, title="status")

my_crosstab_passtime = pd.crosstab(df["passtime"], df["vote"])
my_crosstab_passtime.plot.bar(stacked=True, title="passtime")

my_crosstab_sex = pd.crosstab(df["sex"], df["vote"])
my_crosstab_sex.plot.bar(stacked=True, title="sex")

# Q2.b- boxplot multiva for numeric variables.
df.boxplot(column=["age"], by="vote", grid=False)
df.boxplot(column=["salary"], by="vote", grid=False)
df.boxplot(column=["volunteering"], by="vote", grid=False)
# plt.show()

# Q3- Correcting the data
# print(df.isnull().sum())
# Age- We assume that the minimum age of vote is 16,So we changed the age under 16 and the missing values to the mean .
df["age"] = df["age"].mask(df["age"] < 16, np.nan)
df["age"] = df["age"].replace(to_replace=np.nan, value=df.age.mean())
# print(df.isnull().sum())
# salary column
# we replaced the extreme value to null, and changed the missing values with the average number
extreme_val = df['salary'].max()
# print(df.salary.describe())

df['salary'] = df['salary'].mask(df['salary'] == extreme_val, np.NaN)
df['salary'] = df['salary'].replace(to_replace=np.nan, value=df.salary.mean())
# print(df.salary.describe())
# checking if the last null value was change:
# print(df.salary.tail(10))

# Passtime- We replaced the missing value to the most common value.
# print(df["passtime"].value_counts())
df["passtime"] = df["passtime"].replace(to_replace=np.nan, value="fishing")
# print(df.isnull().sum())

# Creating a new dataframe for normlize
dfnum = df.copy(deep=True)
# encode the categorical variables
le = LabelEncoder()
le.fit(dfnum['sex'])
dfnum['new_sex'] = le.transform(df['sex'])

le.fit(dfnum['passtime'])
dfnum['new_passtime'] = le.transform(df['passtime'])

le.fit(dfnum['status'])
dfnum['new_status'] = le.transform(df['status'])

# Normalize
zscores = stats.zscore(df["salary"])
dfnum["salary"] = zscores
zscores = stats.zscore(df["volunteering"])
dfnum["volunteering"] = zscores
zscores = stats.zscore(df["age"])
dfnum["age"] = zscores
zscores = stats.zscore(dfnum["new_sex"])
dfnum["new_sex"] = zscores
zscores = stats.zscore(dfnum["new_passtime"])
dfnum["new_passtime"] = zscores
zscores = stats.zscore(dfnum["new_status"])
dfnum["new_status"] = zscores
print(dfnum.head())

#print(df.head())

# Q4- Randomly dividing the data into 70% training set and 30% test set using RSEED
dfnum = dfnum.drop(["passtime", "sex"], axis=1)
x = dfnum.drop(["vote", "status"], axis=1)
y = dfnum["vote"]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=RSEED)

# Q5- Building a tree model using set train to predict the value of the variable vote
clf = DecisionTreeClassifier( random_state=RSEED)
clf = clf.fit(X_train, y_train)
# The construction of the tree
plt.figure(figsize=(14, 10), dpi=200)
plot_tree(clf, filled=True, feature_names=X_train.columns, class_names=["Democrat","Republican" ])
# plt.show()

# Q6- Building Confusion matrix
y_test_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))

# Accuracy
ac = pd.crosstab(y_test, y_test_pred, colnames=["pred"], margins=True)
print(ac)
tp=ac.iloc[0,0]
fp=ac.iloc[0,1]
fn=ac.iloc[1,0]
tn=ac.iloc[1,1]
Accuracy=(tp+tn)/(tp+fp+fn+tn)

# Recall
Recall = tp/(tp+fn)

# Precision
Precision= tp/(tp+fp)
print("test set:")
print(" Accuracy:", Accuracy)
print("Recall:",Recall)
print("Precision:",Precision)

# Q7
# checking the train set
y_train_pred = clf.predict(X_train)
ac = pd.crosstab(y_train, y_train_pred , colnames=["pred"], margins=True)
print(ac)
tp=ac.iloc[0,0]
fp=ac.iloc[0,1]
fn=ac.iloc[1,0]
tn=ac.iloc[1,1]
Accuracy=(tp+tn)/(tp+fp+fn+tn)

# Recall
Recall = tp/(tp+fn)

# Precision
Precision= tp/(tp+fp)
print("training set:")
print(" Accuracy:", Accuracy)
print("Recall:",Recall)
print("Precision:",Precision)

# There is an overwriting because the indices of the training set are all 100% -
# it is meen that the machine learned exactly the training set and wont recognize other similar data.


# Q8
# limit the tree to avoid overflow
clf = DecisionTreeClassifier(max_depth=5, min_samples_split =40, random_state=RSEED)
clf = clf.fit(X_train, y_train)
plt.figure(figsize=(14, 10), dpi=200)
plot_tree(clf, filled=True, feature_names=X_train.columns, class_names=["Democrat","Republican" ], fontsize=4)
# plt.show()


# Q9- Prediction on the train set and the limited test set
y_test_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))

# Decision tree - Multiclass -10
# Normalize vote
le.fit(dfnum['vote'])
dfnum['new_vote'] = le.transform(df['vote'])
zscores = stats.zscore(dfnum["new_vote"])
dfnum["new_vote"] = zscores

dfnum = dfnum.drop(["vote"], axis=1)
x = dfnum.drop(["new_status", "status"], axis=1)
y = dfnum["status"]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=RSEED)

# building the tree
clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=40,random_state=RSEED)
clf = clf.fit(X_train, y_train)
plt.figure(figsize=(18, 15), dpi=100)
plot_tree(clf, filled=True, feature_names=X_train.columns, class_names=[ "family","couple", "single"], fontsize=4)

# confusion matrix
y_test_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))

# Accuracy
ac = pd.crosstab(y_test, y_test_pred, colnames=["pred"], margins=True)
tp = ac.iloc[0,0]
fp = ac.iloc[0,1]
fn = ac.iloc[1,0]
tn = ac.iloc[1,1]
Accuracy=(tp+tn)/(tp+fp+fn+tn)
print(Accuracy)



