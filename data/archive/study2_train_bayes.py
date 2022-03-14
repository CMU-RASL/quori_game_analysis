import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.mixture import GaussianMixture

def train_test_split(features, labels, test_perc):
    np.random.seed(6)
    num_series = len(features)
    num_train = np.floor((1-test_perc)*num_series).astype('int')
    inds = np.arange(num_series).astype('int')
    np.random.shuffle(inds)

    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    for ii, x in enumerate(features):
        if ii in inds[:num_train]:
            train_X.append(x.to_numpy())
            if labels[ii]['confidence'][0] < 2:
                train_Y.append(0)
            elif labels[ii]['confidence'][0] > 2:
                train_Y.append(2)
            else:
                train_Y.append(1)
        else:
            test_X.append(x.to_numpy())
            if labels[ii]['confidence'][0] < 2:
                test_Y.append(0)
            elif labels[ii]['confidence'][0] > 2:
                test_Y.append(2)
            else:
                test_Y.append(1)
    
    return train_X, train_Y, test_X, test_Y

def create_model(train_X, train_Y, num_components):
    cur_X = [[], [], []]
    class_weight = [0, 0, 0]
    for x, y in zip(train_X, train_Y):
        cur_X[y].append(x)
        class_weight[y] += 1
    class_weight = np.array(class_weight)
    class_weight = class_weight/np.sum(class_weight)
    models = []
    for ii in range(3):
        flat_X = np.vstack(cur_X[ii])
        gm = GaussianMixture(n_components=num_components).fit(flat_X)
        models.append(gm)
    
    return models, class_weight

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

def plot_confusion(X, Y, probs, label):
    true_y = np.copy(Y)
    predicted_y = np.zeros_like(true_y)
    for ii, prob in enumerate(probs):
        predicted_y[ii] = np.argmax(prob[-1,:])
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(true_y, predicted_y, display_labels=['Low', 'Medium', 'High'], ax=ax)
    ax.set_title(label)

def plot_prop(X, Y, probs, label):
    fig, ax = plt.subplots(3, 3, sharey=True, sharex=True)
    true_y = np.copy(Y)

    for series_num in range(len(X)):
        colors = ['r', 'y', 'g']
        for ii in range(3): 
            ax[true_y[series_num], ii].plot(range(probs[series_num].shape[0]), probs[series_num], color=colors[ii])
    
    ax[0, 0].set_ylabel('Low Confidence')
    ax[1, 0].set_ylabel('Medium Confidence')
    ax[2, 0].set_ylabel('High Confidence')
    ax[0, 0].set_title('Predicted Low Confidence')
    ax[0, 1].set_title('Predicted Medium Confidence')
    ax[0, 2].set_title('Predicted High Confidence')
    fig.suptitle(label)

def plot_length(X, Y, probs, label):
    fig, ax = plt.subplots()
    true_y = np.copy(Y)
    
    predicted_y = np.zeros_like(true_y)
    correct = [[], []]
    incorrect = [[], []]
    for ii, prob in enumerate(probs):
        predicted_y[ii] = np.argmax(prob[-1,:])
        series_len = prob.shape[0]
        if predicted_y[ii] == true_y[ii]:
            correct[0].append(series_len)
            correct[1].append(true_y[ii])
        else:
            incorrect[0].append(series_len)
            incorrect[1].append(true_y[ii])
    ax.scatter(correct[0], correct[1], color='g')
    ax.scatter(incorrect[0], incorrect[1], color='r')
    ax.set_title(label)

def get_acc(X, Y, probs):
    true_y = np.copy(Y)
    predicted_y = np.zeros_like(true_y)
    for ii, prob in enumerate(probs):
        predicted_y[ii] = np.argmax(prob[-1,:])
    acc = np.zeros(3)
    tot = np.zeros(3)
    for true, pred in zip(true_y, predicted_y):
        tot[true] += 1
        if true == pred:
            acc[true] += 1
    tot_acc = np.sum(acc)/np.sum(tot)
    acc = acc/tot

    return acc, tot_acc

if __name__ == '__main__':
    file = open("data/study2_data.pkl",'rb')
    data = pkl.load(file)
    file.close()

    features = data['features']
    labels = data['labels'] 
    
    #Test Train Split
    test_perc = 0.2
    train_X, train_Y, test_X, test_Y = train_test_split(features, labels, test_perc)

    #Train Models
    num_components = 3
    models, class_weight = create_model(train_X, train_Y, num_components)

    #Probability Propagation
    train_probs = prob_prop(models, train_X, train_Y, class_weight)
    test_probs = prob_prop(models, test_X, test_Y, class_weight)

    # #Plot Confusion
    # plot_confusion(train_X, train_Y, train_probs, 'Train Confusion - Bayes')
    # plot_confusion(test_X, test_Y, test_probs, 'Test Confusion - Bayes')

    # #Plot Propagation
    # plot_prop(train_X, train_Y, train_probs, 'Train Prediction Propagation - Bayes')
    # plot_prop(test_X, test_Y, test_probs, 'Test Prediction Propagation - Bayes')    

    plot_length(train_X, train_Y, train_probs, 'Train Series Length - Bayes')
    plot_length(test_X, test_Y, test_probs, 'Test Series Length - Bayes')

    #Get Accuracy
    # num_components_arr = range(7)
    # train_res = []
    # test_res = []
    # for num_components in num_components_arr:
    #     models, class_weight = create_model(train_X, train_Y, num_components)

    #     #Probability Propagation
    #     train_probs = prob_prop(models, train_X, train_Y, class_weight)
    #     test_probs = prob_prop(models, test_X, test_Y, class_weight)

    #     train_acc, train_tot_acc = get_acc(train_X, train_Y, train_probs)
    #     test_acc, test_tot_acc = get_acc(test_X, test_Y, test_probs)


    plt.show()
 

    