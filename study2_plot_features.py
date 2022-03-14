import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def features_for_user(features, labels, user, round):
    user_features = [0, 0, 0, 0, 0, 0, 0, 0]
    user_labels = [0, 0, 0, 0, 0, 0, 0, 0]

    for feature, label in zip(features, labels):
        if feature.iloc[0]['user'] == user:
            trial = int(feature.iloc[0]['trial']*8) - 1
            if feature.iloc[0]['round'] == round:
                user_features[trial] = feature
                user_labels[trial] = label
    return user_features, user_labels

def plot_features(features, labels):
    feature_arr = pd.concat(features, ignore_index=True)
    labels_arr = pd.concat(labels, ignore_index=True)

    fig, axs = plt.subplots(nrows=4, ncols=1)
    feature_arr.loc[:,['AU04', 'AU07', 'AU12', 'AU25', 'AU26'], 
                        ].plot(ax=axs[0])
    feature_arr.loc[:,['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']].plot(ax=axs[1])
                        
    feature_arr.loc[:, ['trial', 'round', 'difficulty', 'performance', 'prev_performance', 'perfect_prob']].plot(ax=axs[2])
    labels_arr.plot(ax=axs[3])
    plt.show()

if __name__ == '__main__':
    file = open("data/study2_data.pkl",'rb')
    data = pkl.load(file)
    file.close()

    features = data['features']
    labels = data['labels']

    #List of all users
    users = set([])
    for feature in features:
        users.add(feature.iloc[0]['user'])
    users = list(users)
    
    #Get features, labels in order for each user
    # round = 1
    # user_features, user_labels = features_for_user(features, labels, users[1], round)
    # print(users)
    # plot_features(user_features, user_labels)
    