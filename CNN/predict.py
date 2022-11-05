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
    # Loading trained model
    model = keras.models.load_model('./saved models/cnn-c-5-100-iter')
    X_test = load_and_preprocess(TEST_DATA_LOC)
    y_pred = model.predict(X_test)
    # from one hot to original label
    y_pred = np.argmax(y_pred, axis=1)

    # for kaggle submission we requiew predictions in a specifically formated csv
    df_y = pd.DataFrame(y_pred).reset_index()
    df_y.columns = ['Index', 'Class']
    df_y.to_csv('./results/pd-cnn-5-submission.csv', index=False)



if __name__ == '__main__':
    predict()