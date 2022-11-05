import numpy as np
import pandas as pd


# Set location of the data in from your local data repo
TRAIN_DATA_LOC = './data/train.csv'
TRAIN_LABEL_LOC = './data/train_result.csv'


def load_and_preprocess(train_data_loc = TRAIN_DATA_LOC, train_label_loc = TRAIN_LABEL_LOC):
    # Read Data into pandas dataframe
    X_train = pd.read_csv(train_data_loc)
    y = pd.read_csv(train_label_loc, index_col = 0).reset_index(drop=True)
    
    # Remove irrelevant columns
    X_train = X_train.iloc[:, :-1]

    # Dataframe to numpy arrays transformation
    X_train  = X_train.to_numpy()
    y = y.to_numpy()
    y = y.reshape(y.shape[0])

    # One hot encoding y
    y_encoded  = pd.get_dummies(y).values

    return X_train, y_encoded


def softmax(z):
    return np.exp(z)/np.sum(np.exp(z))


def crossEntropy(p, q):
    return -np.vdot(p, np.log(q)) 


def eval_L(X, Y, w):
    N = X.shape[0]
    L = 0.0
    for i in range(N):
        ex = X[i]
        y_orig = Y[i]
        y_pred = softmax(w @ ex)
        L += crossEntropy(y_orig, y_pred)
    return L


def regressor(X_train, y_train, learning_rate, epochs):
    N, d = X_train.shape
    X_train = np.insert(X_train, 0, 1, axis=1)
    K = y_train.shape[1]
    w = np.zeros((K, d+1)) 
    
    for epoch in range(epochs):
        L = eval_L(X_train, y_train, w)
        

        print("Epoch: " + str(epoch) + " --- Loss: " + str(L))

        params = np.random.permutation(N) 
        for i in params: 
            ex = X_train[i]
            y_orig = y_train[i]
            y_pred = softmax(w @ ex)
            gradient = np.outer(y_pred - y_orig, ex) 
            
            # updating weights
            w = w - learning_rate * gradient
    return w

def train():
    x_train, y_train= load_and_preprocess(TRAIN_DATA_LOC, TRAIN_LABEL_LOC)

    # Defining Params:
    # Decided after experimenting with 80-20 split earlier, now training with all data for max accuracy
    learning_rate = 0.001
    epochs = 5

    w = regressor(x_train, y_train, learning_rate, epochs)
    
    with open('./saved models/Logistic-Regressor/parameters.npy', 'wb') as f:
        np.save(f, w)


if __name__ == "__main__":
    train()


