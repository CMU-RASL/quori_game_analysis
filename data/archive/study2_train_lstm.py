import pickle as pkl
import numpy as np
from keras import Sequential
from keras.layers import LSTM, Dense, Masking
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def train_test_split(features, labels, test_perc, special_value):
    np.random.seed(6)

    num_series = len(features)
    dimension = features[0].shape[1]
    max_seq_len = max([x.shape[0] for x in features])
    X = np.full((num_series, max_seq_len, dimension), fill_value=special_value)
    Y = np.zeros((num_series, 3))
    for ii, x in enumerate(features):
        seq_len = x.shape[0]
        X[ii, 0:seq_len, :] = x.to_numpy()
        if labels[ii]['confidence'][0] < 2:
            Y[ii, 0] = 1
        elif labels[ii]['confidence'][0] > 2:
            Y[ii, 2] = 1
        else:
            Y[ii, 1] = 1

    num_train = np.floor((1-test_perc)*num_series).astype('int')
    inds = np.arange(num_series).astype('int')
    np.random.shuffle(inds)
    
    train_X = X[inds[:num_train],:,:]
    train_Y = Y[inds[:num_train],:]
    test_X = X[inds[num_train:],:,:]
    test_Y = Y[inds[num_train:],:]
    
    return train_X, train_Y, test_X, test_Y

def create_model(train_X, train_Y, test_X, test_Y, special_value):
    (unique, counts) = np.unique(train_Y, return_counts=True)
    print(counts)
    counts = counts/np.sum(counts)
    weight = {k:v for k, v in zip(unique, counts)}
    print(counts)
    model = Sequential()
    model.add(Masking(mask_value=special_value, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    history = model.fit(train_X, train_Y, epochs=100, batch_size=10, shuffle=True, verbose=1, validation_data=(test_X, test_Y))
    
    return model, history

def plot_loss(history):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.show()

def plot_confusion(X, Y, predicted_y, label):
    true_y = np.argmax(Y, axis=1)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(true_y, predicted_y, display_labels=['Low', 'Medium', 'High'], ax=ax)
    ax.set_title(label)

def plot_prop(X, Y, predicted_y, label):
    fig, ax = plt.subplots(3, 3, sharey=True, sharex=True)
    true_y = np.argmax(Y, axis=1)
    for series_num in range(X.shape[0]):
        prediction = []
        for timestep in range(X.shape[1]):
            if X[series_num, timestep, 0] == special_value:
                pass
            else:
                features = np.copy(X[series_num, :, :])
                features[timestep+1:, :] = special_value
                features = features.reshape([1, features.shape[0], features.shape[1]])
                predicted_y = model.predict(features)
                prediction.append(predicted_y)
        prediction = np.vstack(prediction)
        colors = ['r', 'y', 'g']
        for ii in range(3): 
            ax[true_y[series_num], ii].plot(range(prediction.shape[0]), prediction[:,ii], color=colors[ii])
    
    ax[0, 0].set_ylabel('Low Confidence')
    ax[1, 0].set_ylabel('Medium Confidence')
    ax[2, 0].set_ylabel('High Confidence')
    ax[0, 0].set_title('Predicted Low Confidence')
    ax[0, 1].set_title('Predicted Medium Confidence')
    ax[0, 2].set_title('Predicted High Confidence')
    fig.suptitle(label)

def plot_length(X, Y, predicted_y, label):
    fig, ax = plt.subplots()
    true_y = np.argmax(Y, axis=1)
    
    correct = [[], []]
    incorrect = [[], []]
    for series_num in range(X.shape[0]):

        for timestep in range(X.shape[1]):
            if X[series_num, timestep, 0] == special_value:
                series_len = timestep
                break

        if predicted_y[series_num] == true_y[series_num]:
            correct[0].append(series_len)
            correct[1].append(true_y[series_num])
        else:
            incorrect[0].append(series_len)
            incorrect[1].append(true_y[series_num])
    ax.scatter(correct[0], correct[1], color='g')
    ax.scatter(incorrect[0], incorrect[1], color='r')
    ax.set_title(label)

if __name__ == '__main__':
    file = open("data/study2_data.pkl",'rb')
    data = pkl.load(file)
    file.close()

    features = data['features']
    labels = data['labels'] 
    
    #Test Train Split
    test_perc = 0.2
    special_value = -10000.0

    train_X, train_Y, test_X, test_Y = train_test_split(features, labels, test_perc, special_value)
 
    #Train Model
    model, history = create_model(train_X, train_Y, test_X, test_Y, special_value)
    predicted_train_y = model.predict(train_X)
    predicted_train_y = np.argmax(predicted_train_y, axis=1)
    predicted_test_y = model.predict(test_X)
    predicted_test_y = np.argmax(predicted_test_y, axis=1)

    #Plot Loss
    # plot_loss(history)

    # #Plot Confusion
    # plot_confusion(train_X, train_Y, predicted_train_y, 'Train Confusion - LSTM')
    # plot_confusion(test_X, test_Y, predicted_test_y, 'Test Confusion - LSTM')

    # #Plot Propagation
    # plot_prop(train_X, train_Y, predicted_train_y, 'Train Prediction Propagation - LSTM')
    # plot_prop(test_X, test_Y, predicted_test_y, 'Test Prediction Propagation - LSTM')    

    plot_length(train_X, train_Y, predicted_train_y, 'Train Series Length - LSTM')
    plot_length(test_X, test_Y, predicted_test_y, 'Test Series Length - LSTM')

    plt.show()
 

    