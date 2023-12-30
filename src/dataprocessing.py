import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

def process_data():
    iris = load_iris()
    df = pd.DataFrame(np.concatenate((iris.data, np.array([iris.target]).T), axis=1),
                      columns=iris.feature_names + ['target'])

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
