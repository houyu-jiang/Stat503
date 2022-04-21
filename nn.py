import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, ThresholdedReLU
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score


# read_data
train = pd.read_csv("hr_train.csv")
train = pd.DataFrame(train)
test = pd.read_csv("hr_test.csv")
test = pd.DataFrame(test)

lst1 = list(train["enrollee_id"])
lst2 = list(test["enrollee_id"])

data = pd.concat([train, test], ignore_index = True)

data.groupby(["city"]).count()["enrollee_id"].sort_values(ascending = False)[0:20]

# Feature Engineering
temp = train[train.city.isin(list(["city_103", "city_21", "city_16", "city_114", "city_160", "city_136", "city_67", "city_75", "city_102"]))]
temp = dict(Counter(temp["city"]))
temp = pd.DataFrame([[key, val] for key, val in temp.items()], columns=list(["city", "count"]))
temp = temp.sort_values(by = "count", ascending = False)

ax = sns.barplot(x="city", y="count", data=temp)
plt.savefig("city_dist.png")

# Reduce the level of categorical variables
city_list = ["city_103", "city_21", "city_16", "city_114", "city_160", "city_136", "city_67"]
for i in range(data.shape[0]):
    if data.iloc[i, 1] not in city_list:
        data.iloc[i, 1] = "Other"
    temp_list = ["Masters", "Phd"]
    if data.iloc[i, 6] in temp_list:
        data.iloc[i, 6] = "Above_master"
    elif data.iloc[i, 6] == "Graduate":
        pass
    else:
        data.iloc[i, 6] = "Other".

data = pd.get_dummies(data, columns=['gender'])
data = pd.get_dummies(data, columns=['relevent_experience'])
data = pd.get_dummies(data, columns=['enrolled_university'])
data = pd.get_dummies(data, columns=['education_level'])
data = pd.get_dummies(data, columns=['major_discipline'])
data = pd.get_dummies(data, columns=['company_size'])
data = pd.get_dummies(data, columns=['company_type'])
data = pd.get_dummies(data, columns=['city'])
data = pd.get_dummies(data, columns=['last_new_job'])

data["city_development_index"] = (data["city_development_index"] - np.mean(data["city_development_index"]))/\
    np.std(data["city_development_index"])
data["experience"] = data["experience"] / max(data["experience"])
data["training_hours"] = data["training_hours"] / max(data["training_hours"])

train = data[data.enrollee_id.isin(lst1)]
test = data[data.enrollee_id.isin(lst2)]
train_y = train["target"]
train_x = train.drop(["target", "enrollee_id"], axis = 1)
test_y = test["target"]
test_x = test.drop(["target", "enrollee_id"], axis = 1)

model = Sequential()
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.35, input_shape=(256,)))
model.add(Dense(32, activation='softmax'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=300)

result = list(model.predict(test_x))
result = [1 if item > 0.5 else 0 for item in result]
np.mean(result)

# Compute metrics
def compute_metrics(raw_result, threshold, label):
    result = [1 if item > threshold else 0 for item in raw_result]
    err = 1 - accuracy_score(label, result)
    recall = recall_score(label, result)
    precision = precision_score(label, result)
    f1 = (2 * recall * precision)/ (recall + precision)
    return err, recall, precision, f1

train_acc, train_recall, train_precision, train_f1 = compute_metrics(list(model.predict(train_x)), 0.5, train_y)
print(train_acc) # err
print(train_recall)
print(train_precision)
print(train_f1)


test_acc, test_recall, test_precision, test_f1 = compute_metrics(list(model.predict(test_x)), 0.5, test_y)
print(test_acc) # err
print(test_recall)
print(test_precision)
print(test_f1)

test_pred_result = list(model.predict(test_x))
test_pred_result = [1 if item > 0.5 else 0 for item in test_pred_result]

def acc_by_class(pred, true, group):
    if len(pred) != len(true):
        return False
    elif group == 1:
        total = np.sum(true)
        err = 0
        for i in range(len(pred)):
            if true[i] == 1 and pred[i] == 0:
                err+=1
        return ((err)/total)
    else:
        total = len(pred) - np.sum(true)
        err = 0
        for i in range(len(pred)):
            if true[i] == 0 and pred[i] == 1:
                err+=1
        return ((err)/total)

# Accuracy for positive class on the test set
acc_by_class(test_pred_result, list(test_y), 1)

# Accuracy for negative class on the test set
acc_by_class(test_pred_result, list(test_y), 0)


## Use SMOTE for re-sampling

import imblearn
from imblearn.over_sampling import SMOTE

oversample = SMOTE()
train_x, train_y = oversample.fit_resample(train_x, train_y)