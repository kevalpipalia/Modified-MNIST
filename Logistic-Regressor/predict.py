
import numpy as np
import pandas as pd
from train import softmax

TEST_DATA_LOC = './data/test.csv'

def load_and_preprocess(test_data_loc = TEST_DATA_LOC):
    X_test = pd.read_csv(test_data_loc)
    X_test = X_test.iloc[:, :-1]
    X_test = X_test.to_numpy()
    X_test = np.insert(X_test, 0, 1, axis=1)
    return X_test


def predict():
    # Loading trained weights for prediction
    with open('./saved models/Logistic-Regressor/parameters.npy', 'rb') as f:
        w = np.load(f)

    X_test = load_and_preprocess(TEST_DATA_LOC)
    
    N = X_test.shape[0]

    predictions = []

    for i in range(N):
        ex = X_test[i]
        y_pred = softmax(w @ ex)
        k = np.argmax(y_pred)
        predictions.append(k)

    # For submission in kaggle we need to convert results in this format
    df_y = pd.DataFrame(np.array(predictions)).reset_index()
    df_y.columns = ['Index', 'Class']
    df_y.to_csv('./results/Logistic-Regressor.csv', index=False)


if __name__ == '__main__':
    predict()