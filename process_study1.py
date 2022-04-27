import pickle as pkl
import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from pingouin import mixed_anova
from statsmodels.graphics.api import interaction_plot

def two_way_anova(data, col_name, title):
    fig, ax = plt.subplots(figsize=(3,4))
    unique_arr = data["nonverbal"].unique()

    easy = np.vstack(data.loc[(data['difficulty'] == 'EASY'), col_name].to_numpy())
    easy_mean = np.mean(easy,axis=0)[0]
    easy_std = np.std(easy,axis=0)[0]
    
    difficult = np.vstack(data.loc[(data['difficulty'] == 'DIFFICULT'), col_name].to_numpy())
    difficult_mean = np.mean(difficult,axis=0)[0]
    difficult_std = np.std(difficult,axis=0)[0]

    data_list = []
    mean_list = []
    std_list = []
    for col in unique_arr:
        cur = np.vstack(data.loc[(data['nonverbal'] == col), col_name].to_numpy())
        mean_list.append(np.mean(cur,axis=0)[0])
        std_list.append(np.std(cur,axis=0)[0])
        data_list.append(cur)
    
    labels = []
    labels.extend(unique_arr)
    x_pos = np.arange(len(unique_arr))
    means = []
    means.extend(mean_list)
    stds = []
    stds.extend(std_list)

    ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    if not (col_name == 'last_mistake') and not (col_name == 'accuracy') and not (col_name == 'elapsed_time'):
        ax.set_ylim([-1, 5])
    elif col_name == 'last_mistake':
        ax.set_ylim([-2, 10])
    elif col_name == 'accuracy':
        ax.set_ylim([-0.2, 1.2])

    aov = mixed_anova(dv=col_name, between='nonverbal',
                  within='difficulty', subject='user_id', data=data)
    aov['0.05 level'] = 'NaN'
    aov.loc[aov["p-unc"] <= 0.05, "0.05 level"] = 'Significant'
    aov['0.01 level'] = 'NaN'
    aov.loc[aov["p-unc"] <= 0.01, "0.01 level"] = 'Significant'
    aov['0.001 level'] = 'NaN'
    aov.loc[aov["p-unc"] <= 0.001, "0.001 level"] = 'Significant'
    print('---')
    print(col_name)
    print(aov)
    print('---')

def posthoc(data, col_name):
    data['combo'] = data['difficulty'] + " / " + data['nonverbal']
    m_comp = pairwise_tukeyhsd(endog=data[col_name], groups=data['combo'], alpha=0.05)
    # m_comp = pairwise_tukeyhsd(endog=data[col_name], groups=data['nonverbal'], alpha=0.05)
    tukey_data = pd.DataFrame(data=m_comp._results_table.data[1:], columns = m_comp._results_table.data[0])
    print('---')
    print(col_name)
    print(tukey_data)
    print('---')

    fig, ax = plt.subplots()
    interaction_plot(data['difficulty'], data['nonverbal'], data['accuracy'], ax=ax)
    plt.show()

def sorted_feedback(data):
    difficulty = ['EASY', 'DIFFICULT']
    nonverbal = ['NEUTRAL', 'MATCHING', 'NONMATCHING']
    res = {'EASY': {}, 'DIFFICULT': {}, 'NEUTRAL': {}, 'MATCHING': {}, 'NONMATCHING': {}}
    all = {}
    for diff in difficulty:
        for non in nonverbal:
            # print(diff, non)
            feedback = data.loc[(data['difficulty'] == diff) & (data['nonverbal'] == non), ['feedback', 'round', 'user_id']]
            for ind, row in feedback.iterrows():
                if len(row['feedback']) > 0:
                    if row['round'] == 1:
                        #Get what the previous row was
                        prev_row = data.loc[(data['user_id'] == row['user_id']) & (data['round'] == 0), ['difficulty', 'nonverbal']]

                    str_list = row['feedback'].split()
                    unique_words = set(str_list)

                    for word in unique_words:
                        if word in all.keys():
                            all[word] += 1
                        else:
                            all[word] = 1
                        if word in res[diff].keys():
                            res[diff][word] += 1
                        else:
                            res[diff][word] = 1
                        if word in res[non].keys():
                            res[non][word] += 1
                        else:
                            res[non][word] = 1


    word_groups = [('learn', 'learned', 'learns'), ('understand', 'understood', 'understands'), ('easy', 'easier', 'simple', 'simpler'), ('difficult', 'difficulty'), ('enjoy', 'enjoyed', 'enjoys', 'lively', 'like', 'helpful', 'helped', 'helps', 'likes', 'liked', 'felt', 'feels')]
    # word_groups = [('distract', 'distracting', 'distracted', 'distracts', 'distracting.', 'distracts.', 'distracted.')]
    for group in word_groups:
        print('-------------------------------')
        print(group)

        #Neutral
        print('----------NEUTRAL')
        feedback = data.loc[(data['nonverbal'] == 'NEUTRAL'), ['feedback']]
        for ind, row in feedback.iterrows():
            str_list = row['feedback'].split()
            for word in group:
                if word in str_list and 'robot' in str_list:
                    print(row['feedback'])
                    break

        #Affective
        print('----------MATCHING')
        feedback = data.loc[(data['nonverbal'] == 'MATCHING'), ['feedback']]
        for ind, row in feedback.iterrows():
            str_list = row['feedback'].split()
            for word in group:
                if word in str_list and 'robot' in str_list:
                    print(row['feedback'])
                    break
        
        print('----------NONMATCHING')
        feedback = data.loc[(data['nonverbal'] == 'NONMATCHING'), ['feedback']]
        for ind, row in feedback.iterrows():
            str_list = row['feedback'].split()
            for word in group:
                if word in str_list and 'robot' in str_list:
                    print(row['feedback'])
                    break
        print('-------------------------------')

if __name__ == "__main__":
    file = open("data/study1a_data.pkl",'rb')
    study1a = pkl.load(file)
    file.close()
    study1a['nonverbal'] = study1a['nonverbal'].map({'NONVERBAL': 'MATCHING', 'NEUTRAL': 'NEUTRAL'})
    
    file = open("data/study1b_data.pkl",'rb')
    study1b = pkl.load(file)
    file.close()
    study1b['nonverbal'] = study1b['nonverbal'].map({'NONVERBAL': 'NONMATCHING', 'NEUTRAL': 'NEUTRAL'})
    study1b['user_id'] = study1b['user_id'] + 1000

    file = open("data/study1c_data.pkl",'rb')
    study1c = pkl.load(file)
    file.close()
    study1c['user_id'] = study1c['user_id'] + 2000
    
    # data = pd.concat([study1a, study1c], ignore_index=True)
    data = study1b
    # data = data.loc[data['nonverbal'] != 'NEUTRAL1']
    # data = data.loc[data['nonverbal'] != 'NEUTRAL2']
    # data = data.loc[data['nonverbal'] != 'NEUTRAL']
    # data = data.loc[data['nonverbal'] != 'MATCHING']
    # data = data.loc[data['nonverbal'] != 'NONMATCHING']
    # print(data)
    two_way_anova(data, 'accuracy', 'Accuracy')
    # two_way_anova(data, 'user_learning', 'User Learning')
    # two_way_anova(data, 'perceived_difficulty', 'Perceived Difficulty')
    # two_way_anova(data, 'engagement', 'Engagement')
    # two_way_anova(data, 'animacy', 'Animacy')
    # two_way_anova(data, 'intelligence', 'Intelligence')
    # posthoc(data, 'accuracy')
    # sorted_feedback(data)
    plt.show()
     