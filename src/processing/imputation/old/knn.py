import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

plt.style.use('classic')


def knn_algorithm(X, y, k):
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # feature scaling
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # training
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # evaluation
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return model, scaler


def find_best_k(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    error = []
    for i in tqdm(range(1, 10)):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != y_test))

    x = range(1, 10)
    y = error
    f2 = interp1d(x, y, kind='cubic')
    xnew = np.linspace(1, 9, 36)

    plt.figure(figsize=(12, 6))
    plt.plot(x, y, marker='o', markersize=5)
    plt.plot(xnew, f2(xnew))
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    plt.legend(['data', 'interpolation'], loc='best')

    plt.savefig('elbow_knn.png', bbox_inches='tight')


"""
def size_code_imputation(df):
    aux = df[['size_code', 'air_time']].dropna()
    # 0 - NB; 1 - WB
    aux['size_code'] = aux['size_code'].astype(
        'category').cat.codes  # convert categorical into numerical
    aux.corr()  # 86%
    X = aux.iloc[:, 1].values.reshape(-1, 1)  # flight time
    y = aux.iloc[:, 0].values  # size code

    # knn.find_best_k(X, y)  # k = 4 seems to be the best fit

    model, scaler = knn.knn_algorithm(X, y, 4)
    nansize_airtimes = df[df['size_code'].isna()]['air_time']
    nansize_airtimes = scaler.transform(nansize_airtimes.values.reshape(-1, 1))
    size_pred = model.predict(nansize_airtimes)
    df['size_code'] = df['size_code'].fillna(
        pd.Series(size_pred, index=df[df['size_code'].isna()].index))
    df.loc[df['size_code'] == 1.0, 'size_code'] = 'WB'
    df.loc[df['size_code'] == 0.0, 'size_code'] = 'NB'

    return df
"""
