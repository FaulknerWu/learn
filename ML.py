import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def column_eliminator(df, thr=0.9):
    columns = np.full((df.shape[0],), True, dtype=bool)
    for i in range(df.shape[0]):
        for j in range(i + 1, df.shape[0]):
            if df.iloc[i, j] >= thr:
                if columns[j]:
                    columns[j] = False
    return columns


df_resting = pd.read_csv("D:\\EEG_128channels_resting_lanzhou_2015\\csv\\resting.csv")
df_demog = pd.read_excel(
    "D:\\EEG_128channels_resting_lanzhou_2015\\subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx",
    sheet_name='Sheet1')

df_resting['patient_identifier'] = [i[:8] for i in df_resting['patient_identifier']]
df_resting['patient_identifier'] = df_resting['patient_identifier'].astype(int)

df = pd.merge(df_resting, df_demog, left_on='patient_identifier', right_on='subject id', how='inner')

df.set_index('patient_identifier', inplace=True)
df.drop(columns=['Unnamed: 11', 'Unnamed: 12', 'subject id'], inplace=True)

df.type.replace({'MDD': 1, 'HC': 0}, inplace=True)
df.drop(columns='gender', inplace=True)

columns_to_consider = list(df.columns[column_eliminator(df.corr(), thr=0.8)])
df = df[columns_to_consider]

features = df.columns[:-1]
x = df.loc[:, features].values

x = StandardScaler().fit_transform(x)

pca = PCA(n_components=15)
principal_components = pca.fit_transform(x)

principal_df = pd.DataFrame(data=principal_components, columns=[f'principal component {i+1}' for i in range(15)])

final_df = pd.concat([principal_df, df[['type']].reset_index(drop=True)], axis=1)

X = final_df.drop(columns=['type'])
y = final_df['type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)

print(f'随机森林分类器准确率: {rf_accuracy:.2f}')
print(f'随机森林分类器精确率: {rf_precision:.2f}')
print(f'随机森林分类器召回率: {rf_recall:.2f}')
print(f'随机森林分类器F1分数: {rf_f1:.2f}')

svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_clf.fit(X_train, y_train)
y_pred_svm = svm_clf.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_precision = precision_score(y_test, y_pred_svm)
svm_recall = recall_score(y_test, y_pred_svm)
svm_f1 = f1_score(y_test, y_pred_svm)

print(f'SVM分类器准确率: {svm_accuracy:.2f}')
print(f'SVM分类器精确率: {svm_precision:.2f}')
print(f'SVM分类器召回率: {svm_recall:.2f}')
print(f'SVM分类器F1分数: {svm_f1:.2f}')
