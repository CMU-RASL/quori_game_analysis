import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.mixture import GaussianMixture
from keras import Sequential
from keras.layers import LSTM, Dense, Masking

def train_test_split(features, labels, model_type, special_value):
    np.random.seed(6)
    num_series = len(features)
    inds = np.arange(num_series).astype('int')
    np.random.shuffle(inds)

    split_inds = np.array_split(inds, 5)

    all_train_X = []
    all_train_Y = []
    all_test_X = []
    all_test_Y = []
    for split_num in range(5):
        if model_type == 'bayes':
            train_X = []
            train_Y = []
            test_X = []
            test_Y = []
            for ii, x in enumerate(features):
                if ii in split_inds[split_num]:
                    test_X.append(x.to_numpy())
                    if labels[ii]['confidence'][0] < 2:
                        test_Y.append(0)
                    elif labels[ii]['confidence'][0] > 2:
                        test_Y.append(2)
                    else:
                        test_Y.append(1)
                else:
                    train_X.append(x.to_numpy())
                    if labels[ii]['confidence'][0] < 2:
                        train_Y.append(0)
                    elif labels[ii]['confidence'][0] > 2:
                        train_Y.append(2)
                    else:
                        train_Y.append(1)
        elif model_type == 'lstm':
            dimension = features[0].shape[1]
            max_seq_len = 40 #max([x.shape[0] for x in features])
            train_X = []
            train_Y = []
            test_X = []
            test_Y = []
            for ii, x in enumerate(features):
                seq_len = x.shape[0]
                xx = np.full((max_seq_len, dimension), fill_value=special_value)
                xx[0:seq_len, :] = x.to_numpy()
                if ii in split_inds[split_num]:
                    test_X.append(xx)
                    if labels[ii]['confidence'][0] < 2:
                        test_Y.append([1, 0, 0])
                    elif labels[ii]['confidence'][0] > 2:
                        test_Y.append([0, 0, 1])
                    else:
                        test_Y.append([0, 1, 0])
                else:
                    train_X.append(xx)
                    if labels[ii]['confidence'][0] < 2:
                        train_Y.append([1, 0, 0])
                    elif labels[ii]['confidence'][0] > 2:
                        train_Y.append([0, 0, 1])
                    else:
                        train_Y.append([0, 1, 0])

            train_X = np.stack(train_X)
            train_Y = np.vstack(train_Y)
            test_X = np.stack(test_X)
            test_Y = np.vstack(test_Y)
        elif model_type == 'lstm_regress':
            dimension = features[0].shape[1]
            max_seq_len = 40 #max([x.shape[0] for x in features])
            train_X = []
            train_Y = []
            test_X = []
            test_Y = []
            for ii, x in enumerate(features):
                seq_len = x.shape[0]
                xx = np.full((max_seq_len, dimension), fill_value=special_value)
                xx[0:seq_len, :] = x.to_numpy()
                if ii in split_inds[split_num]:
                    test_X.append(xx)
                    test_Y.append(labels[ii]['confidence'][0])
                else:
                    train_X.append(xx)
                    train_Y.append(labels[ii]['confidence'][0])
                    
            train_X = np.stack(train_X)
            train_Y = np.vstack(train_Y)
            test_X = np.stack(test_X)
            test_Y = np.vstack(test_Y)
        all_train_X.append(train_X)
        all_train_Y.append(train_Y)
        all_test_X.append(test_X)
        all_test_Y.append(test_Y)

    return all_train_X, all_train_Y, all_test_X, all_test_Y

def create_model(train_X, train_Y, test_X, test_Y, model_type, special_value=0, num_components=3, num_epochs=100, num_hidden_units=50):
    if model_type == 'lstm':
        (unique, counts) = np.unique(train_Y, return_counts=True)
        counts = counts/np.sum(counts)
        weight = {k:v for k, v in zip(unique, counts)}
        model = Sequential()
        model.add(Masking(mask_value=special_value, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(LSTM(num_hidden_units, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        history = model.fit(train_X, train_Y, epochs=num_epochs, batch_size=10, shuffle=True, verbose=1)
        class_weight = None
    elif model_type == 'lstm_regress':
        (unique, counts) = np.unique(train_Y, return_counts=True)
        counts = counts/np.sum(counts)
        weight = {k:v for k, v in zip(unique, counts)}
        model = Sequential()
        model.add(Masking(mask_value=special_value, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(LSTM(num_hidden_units, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        
        history = model.fit(train_X, train_Y, epochs=num_epochs, batch_size=10, shuffle=True, verbose=1)
        class_weight = None
    elif model_type == 'bayes':
        cur_X = [[], [], []]
        class_weight = [0, 0, 0]
        for x, y in zip(train_X, train_Y):
            cur_X[y].append(x)
            class_weight[y] += 1
        class_weight = np.array(class_weight)
        class_weight = class_weight/np.sum(class_weight)
        model = []
        for ii in range(3):
            flat_X = np.vstack(cur_X[ii])
            gm = GaussianMixture(n_components=num_components).fit(flat_X)
            model.append(gm)
        history = None
    
    return model, history, class_weight

def prob_prop(models, X, Y, class_weight):
    all_probs = []
    for xx, yy, in zip(X, Y):
        prev_prob = np.ones_like(class_weight)/class_weight.shape[0]
        probs = np.zeros((xx.shape[0], 3))
        for ii in range(xx.shape[0]):
            model_prob = np.zeros((len(models)))
            for model_ind, model in enumerate(models):
                model_prob[model_ind] = model.score_samples(xx[ii,:].reshape(1, -1))[0]
            model_prob = np.exp(model_prob)
            model_prob[model_prob < 1e-6] = 1e-6
            prev_prob[prev_prob < 1e-6] = 1e-6

            probs[ii,:] = model_prob*class_weight*prev_prob
            probs[ii,:] = probs[ii,:]/(np.sum(probs[ii,:]))

            prev_prob = probs[ii,:]
        all_probs.append(probs)
    return all_probs

def plot_length(X, Y, probs, model_type, label, special_value=0):
    fig, ax = plt.subplots()
    if model_type == 'lstm':
        true_y = np.argmax(Y, axis=1)
    elif model_type == 'bayes':
        true_y = np.copy(Y)

    correct = [[], []]
    incorrect = [[], []]
    for series_num, prob in enumerate(probs):
        series_len = 40
        if model_type == 'lstm':
            series_len = X.shape[1]
            for timestep in range(X.shape[1]):
                if X[series_num, timestep, 0] == special_value:
                    series_len = timestep
                    break
            predicted_y = np.argmax(prob)
        elif model_type == 'bayes':
            series_len = prob.shape[0]
            predicted_y = np.argmax(prob[-1,:])
        if predicted_y == true_y[series_num]:
            correct[0].append(series_len)
            correct[1].append(true_y[series_num])
        else:
            incorrect[0].append(series_len)
            incorrect[1].append(true_y[series_num])
    ax.scatter(correct[0], correct[1], color='g')
    ax.scatter(incorrect[0], incorrect[1], color='r')
    ax.set_title(label)
    ax.set_xlim([0, 40])

def get_confusion(Y, probs, model_type):
    if model_type == 'lstm':
        true_y = np.argmax(Y, axis=1)
    elif model_type == 'bayes':
        true_y = np.copy(Y)
    elif model_type == 'lstm_regress':
        true_y = np.copy(Y)

    conf_matrix = np.zeros((3,3))
    
    for series_num, prob in enumerate(probs):
        if model_type == 'lstm':
            predicted_y = np.argmax(prob)
            conf_matrix[true_y[series_num], predicted_y] += 1
        elif model_type == 'bayes':
            predicted_y = np.argmax(prob[-1,:])
            conf_matrix[true_y[series_num], predicted_y] += 1
        elif model_type == 'lstm_regress':
            predicted_y = regress_to_class(prob)
            conf_matrix[regress_to_class(true_y[series_num]), predicted_y] += 1
    return conf_matrix

def plot_acc(x_data, conf_matrices, label):
    acc = []
    for conf_mat in conf_matrices:
        acc.append(np.sum(np.diag(conf_mat)) / np.sum(conf_mat))

    fig, ax = plt.subplots()
    ax.plot(x_data, acc, marker='D', label='Accuracy')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.set_title(label)

def regress_to_class(val):
    if val < 1.5:
        return 0
    elif val > 2.5:
        return 2
    else:
        return 1
    # if val < 0.5:
    #     return 0
    # elif val < 1.5:
    #     return 1
    # elif val< 2.5:
    #     return 2
    # elif val < 3.5:
    #     return 3
    # else:
    #     return 4

def single_run(train_X, train_Y, test_X, test_Y, model_type, special_value):
    num_components = 6
    num_epochs = 100
    num_hidden_units = 110

    train_confusion = np.zeros((3,3))
    test_confusion = np.zeros((3,3))

    for split in range(5):
        model, history, class_weight = create_model(train_X[split], train_Y[split], test_X[split], test_Y[split], model_type, special_value=special_value, num_components=num_components, num_epochs=num_epochs, num_hidden_units=num_hidden_units)

        if model_type == 'lstm':
            train_probs = model.predict(train_X[split])
            test_probs = model.predict(test_X[split])

        elif model_type == 'bayes':
            train_probs = prob_prop(model, train_X[split], train_Y[split], class_weight)
            test_probs = prob_prop(model, test_X[split], test_Y[split], class_weight)
        
        elif model_type == 'lstm_regress':
            train_probs = model.predict(train_X[split])
            test_probs = model.predict(test_X[split])

        train_confusion_split = get_confusion(train_Y[split], train_probs, model_type)
        test_confusion_split = get_confusion(test_Y[split], test_probs, model_type)
        train_confusion += train_confusion_split
        test_confusion += test_confusion_split
        print('Split ', split+1)

    train_confusion = train_confusion/5
    test_confusion = test_confusion/5
    
    if model_type == 'bayes':
        plot_confusion(train_confusion, 'Train Confusion - {} - {} Components'.format(model_type, num_components))
        plot_confusion(test_confusion, 'Test Confusion - {} - {} Components'.format(model_type, num_components))
    else:
        plot_confusion(train_confusion, 'Train Confusion - {} - {} Hidden Units - {} Epochs'.format(model_type, num_hidden_units, num_epochs))
        plot_confusion(test_confusion, 'Test Confusion - {} - {} Hidden Units - {} Epochs'.format(model_type, num_hidden_units, num_epochs))
    
def plot_confusion(conf_mat, title):
    acc = np.sum(np.diag(conf_mat)) / np.sum(conf_mat)
    fig, ax = plt.subplots()
    # disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=['0', '1', '2', '3', '4'])
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=['Low', 'Medium', 'High'])
    disp.plot(ax=ax, cmap='YlGn')
    ax.set_title('{} - Acc: {:.0%}'.format(title, acc))

def components(train_X, train_Y, test_X, test_Y, model_type):
    num_components_arr = range(2, 7)
    num_epochs = 70
    num_hidden_units = 50
    special_value = -1000
    
    all_train_conf = []
    all_test_conf = []
    for num_parameter_ind in range(len(num_components_arr)):
        all_train_conf.append(np.zeros((3,3)))
        all_test_conf.append(np.zeros((3,3)))

    for split in range(5):

        for num_parameter_ind, num_components in enumerate(num_components_arr):
            model, history, class_weight = create_model(train_X[split], train_Y[split], test_X[split], test_Y[split], model_type, special_value=special_value, num_components=num_components, num_epochs=num_epochs, num_hidden_units=num_hidden_units)

            train_probs = prob_prop(model, train_X[split], train_Y[split], class_weight)
            test_probs = prob_prop(model, test_X[split], test_Y[split], class_weight)
            
            all_train_conf[num_parameter_ind] += get_confusion(train_Y[split], train_probs, model_type)
            all_test_conf[num_parameter_ind] += get_confusion(test_Y[split], test_probs, model_type)

            print('Split {}/5 - Parameter {}/{}'.format(split+1, num_parameter_ind+1, len(num_components_arr)))
    
    for num_parameter_ind in range(len(num_components_arr)):
        all_train_conf[num_parameter_ind] = all_train_conf[num_parameter_ind]/5
        all_test_conf[num_parameter_ind] = all_test_conf[num_parameter_ind]/5

    #Plot accuracies
    plot_acc(num_components_arr, all_train_conf, 'Train Data - Number of Components - ' + model_type)
    plot_acc(num_components_arr, all_test_conf, 'Test Data - Number of Components - ' + model_type)

def hidden_units(train_X, train_Y, test_X, test_Y, model_type, special_value):
    num_components = 3
    num_epochs = 80
    num_hidden_units_arr = [70, 80, 90, 100, 110, 120]
    
    all_train_conf = []
    all_test_conf = []
    for num_parameter_ind in range(len(num_hidden_units_arr)):
        all_train_conf.append(np.zeros((3,3)))
        all_test_conf.append(np.zeros((3,3)))

    for split in range(5):
        # Train Models
        train_res = []
        test_res = []
        for num_parameter_ind, num_hidden_units in enumerate(num_hidden_units_arr):
            model, history, class_weight = create_model(train_X[split], train_Y[split], test_X[split], test_Y[split], model_type, special_value=special_value, num_components=num_components, num_epochs=num_epochs, num_hidden_units=num_hidden_units)

            train_probs = model.predict(train_X[split])
            test_probs = model.predict(test_X[split])
            
            all_train_conf[num_parameter_ind] += get_confusion(train_Y[split], train_probs, model_type)
            all_test_conf[num_parameter_ind] += get_confusion(test_Y[split], test_probs, model_type)

            print('Split {}/5 - Parameter {}/{}'.format(split+1, num_parameter_ind+1, len(num_hidden_units_arr)))
    
    for num_parameter_ind in range(len(num_hidden_units_arr)):
        all_train_conf[num_parameter_ind] = all_train_conf[num_parameter_ind]/5
        all_test_conf[num_parameter_ind] = all_test_conf[num_parameter_ind]/5

    #Plot accuracies
    plot_acc(num_hidden_units_arr, all_train_conf, 'Train Data - Number of Hidden Units with Num Epochs {} - {}'.format(num_epochs, model_type))
    plot_acc(num_hidden_units_arr, all_test_conf, 'Test Data - Number of Hidden Units with Num Epochs {} - {}'.format(num_epochs, model_type))

def epochs(train_X, train_Y, test_X, test_Y, model_type, special_value):

    num_components = 3
    num_epochs_arr = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    num_hidden_units = 50
    
    all_train_conf = []
    all_test_conf = []
    for num_parameter_ind in range(len(num_epochs_arr)):
        all_train_conf.append(np.zeros((3,3)))
        all_test_conf.append(np.zeros((3,3)))

    for split in range(5):
        # Train Models
        train_res = []
        test_res = []
        for num_parameter_ind, num_epochs in enumerate(num_epochs_arr):
            model, history, class_weight = create_model(train_X[split], train_Y[split], test_X[split], test_Y[split], model_type, special_value=special_value, num_components=num_components, num_epochs=num_epochs, num_hidden_units=num_hidden_units)

            train_probs = model.predict(train_X[split])
            test_probs = model.predict(test_X[split])
            
            all_train_conf[num_parameter_ind] += get_confusion(train_Y[split], train_probs, model_type)
            all_test_conf[num_parameter_ind] += get_confusion(test_Y[split], test_probs, model_type)

            print('Split {}/5 - Parameter {}/{}'.format(split+1, num_parameter_ind+1, len(num_epochs_arr)))
 
    for num_parameter_ind in range(len(num_epochs_arr)):
        all_train_conf[num_parameter_ind] = all_train_conf[num_parameter_ind]/5
        all_test_conf[num_parameter_ind] = all_test_conf[num_parameter_ind]/5

    #Plot accuracies
    plot_acc(num_epochs_arr, all_train_conf, 'Train Data - Number of Epochs with Num Hidden Units {} - {}'.format(num_hidden_units, model_type))
    plot_acc(num_epochs_arr, all_test_conf, 'Test Data - Number of Epochs with Num Hidden Units {} - {}'.format(num_hidden_units, model_type))

if __name__ == '__main__':
    file = open("data/study2_data_all.pkl",'rb')
    data = pkl.load(file)
    file.close()

    features = data['features']
    labels = data['labels'] 
    
    #Test Train Split
    model_type = 'lstm_regress'
    special_value = -10000.0

    train_X, train_Y, test_X, test_Y = train_test_split(features, labels, model_type, special_value=special_value)
    
    single_run(train_X, train_Y, test_X, test_Y, model_type, special_value)
    # components(train_X, train_Y, test_X, test_Y, model_type)
    # hidden_units(train_X, train_Y, test_X, test_Y, model_type, special_value)
    # epochs(train_X, train_Y, test_X, test_Y, model_type, special_value)
    
    plt.show()

    # plot_length(train_X, train_Y, train_probs, model_type, 'Train Series Length - ' + model_type, special_value=special_value)
    # plot_length(test_X, test_Y, test_probs, model_type, 'Test Series Length - ' + model_type, special_value=special_value)