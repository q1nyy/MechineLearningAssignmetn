#coding=utf-8
import pandas as pd

data = pd.read_excel(r'Data\dataSet1\train.xlsx')
data.drop_duplicates(inplace=True)
data.reset_index(drop=True, inplace=True)

print("")
def filter_nan(data, narate = 0.2):
    n_samples = data.shape[0]
    list_nan_cols = []
    
    for col in data.columns:
        if data[col].isna().sum() / n_samples >= (narate):
            list_nan_cols.append(col)
    
    return list_nan_cols

list_nullfactor_todrop = filter_nan(data, narate=0.2)

data_select = data.drop(list_nullfactor_todrop, axis = 1).copy()


from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

numerical_cols = data_select.select_dtypes(include=['float64', 'int64']).columns

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_select[numerical_cols])

knn_imputer = KNNImputer(n_neighbors=10)
imputed_data = knn_imputer.fit_transform(scaled_data)

imputed_data = scaler.inverse_transform(imputed_data)

data_select[numerical_cols] = imputed_data


data_select['Patient Code'] = range(len(data))
data_select['Group '] = data_select['Group '].apply(lambda x: 0 if x == 'Control' else 1)

from sklearn.model_selection import train_test_split
X = data_select.drop(columns=['Group ', 'Patient Code'])
y = data_select['Group ']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix


decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)

y_pred = decision_tree_model.predict(X_test)

print("分类报告:\n", classification_report(y_test, y_pred))
print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))