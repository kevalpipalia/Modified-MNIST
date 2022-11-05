import numpy as np
import pandas as pd
import keras

TEST_DATA_LOC = './data/test.csv'

def load_and_preprocess(test_data_loc = TEST_DATA_LOC):
    X_test = pd.read_csv(test_data_loc)
    X_test = X_test.iloc[:, :-1]
    X_test = X_test.values.reshape(-1,28,56,1)

    return X_test

def predict():

    nets = 15
    model = [0] *nets
    # Loading trained model
    for j in range(nets):
        model[j] = keras.models.load_model(f'./saved models/ensemble-c5/{j}')

    X_test = load_and_preprocess(TEST_DATA_LOC)
    final = np.zeros((nets, X_test.shape[0]))
    for j in range(nets):
        res_vector = model[j].predict(X_test)
        final[j] = np.argmax(res_vector, axis=1)
    df = pd.DataFrame(final.T)
    y_pred = np.zeros(X_test.shape[0])
    for i in range(X_test.shape[0]):
        y_pred[i] = df.iloc[i].value_counts().index.values[0]
    # from one hot to original label
    y_pred = np.argmax(y_pred, axis=1)

    # for kaggle submission we requiew predictions in a specifically formated csv
    df_y = pd.DataFrame(y_pred).reset_index()
    df_y.columns = ['Index', 'Class']
    df_y.to_csv('./results/voting-c5.csv', index=False)



if __name__ == '__main__':
    predict()