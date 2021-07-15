import sqlite3
import pandas as pd
from data.params_study1 import *
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from data.utils import *
from os import path

def read_tables():

    # Create your connection.
    cnx = sqlite3.connect('data/pilot_study1.db')

    condition_tab = pd.read_sql_query("SELECT * FROM condition", cnx)

    #Add new columns, for each round
    condition_tab['difficulty0'] = ['', '', '', '', '', '', '', '']
    condition_tab['difficulty1'] = ['', '', '', '', '', '', '', '']
    condition_tab['nonverbal0'] = ['', '', '', '', '', '', '', '']
    condition_tab['nonverbal1'] = ['', '', '', '', '', '', '', '']
    for index in range(condition_tab.shape[0]):
        condition_tab.at[index, 'difficulty'] = (CONDITIONS[index][0][0], CONDITIONS[index][1][0])
        condition_tab.at[index, 'nonverbal'] = (CONDITIONS[index][0][1], CONDITIONS[index][1][1])
        condition_tab.at[index, 'difficulty0'] = CONDITIONS[index][0][0]
        condition_tab.at[index, 'difficulty1'] = CONDITIONS[index][1][0]
        condition_tab.at[index, 'nonverbal0'] = CONDITIONS[index][0][1]
        condition_tab.at[index, 'nonverbal1'] = CONDITIONS[index][1][1]

    # print(condition_tab)

    user_tab = pd.read_sql_query("SELECT * from user", cnx)
    # print(user_tab)

    trials_tab = pd.read_sql_query("SELECT * from trial", cnx)
    for index in range(trials_tab.shape[0]):
        #Get User id
        user_id = trials_tab.at[index, 'user_id'].item()

        #Get condition id
        condition_id = user_tab.loc[user_tab['id'] == user_id, 'condition_id'].item()
        
        #Get round
        round_num = trials_tab.at[index, 'round_num'].item()

        #Get trial_num
        trial_num = trials_tab.at[index, 'trial_num'].item()

        #Get difficulty
        if round_num == 0:
            difficulty = condition_tab.loc[condition_tab['id'] == condition_id, 'difficulty0'].item()
        else:
            difficulty = condition_tab.loc[condition_tab['id'] == condition_id, 'difficulty1'].item()
        
        #Get correct bin for trial
        correct_bin = RULE_PROPS[difficulty]['answers'][round_num]
        trials_tab.at[index, 'correct_bin'] = correct_bin

        #Get rule
        trials_tab.at[index, 'rule_set'] = RULE_PROPS[difficulty]['rule']

    # print(trials_tab)

    demos_tab = pd.read_sql_query("SELECT * from demo", cnx)
    # print(demos_tab)

    survey_tab = pd.read_sql_query("SELECT * from survey", cnx)
    # print(survey_tab)

    return {'condition': condition_tab, 'user': user_tab, 'trial': trials_tab, 'demo': demos_tab, 'survey': survey_tab}

def compile_data(tabs):
    df = pd.DataFrame(columns=('difficulty', 'nonverbal', 'accuracy', 'user_id', 'condition_id', 'animacy', 'intelligence', 'user_learning'))

    for user_index in range(tabs['user'].shape[0]):

        #User Id
        user_id = tabs['user'].at[user_index, 'id'].item()

        #Condition Id
        condition_id = tabs['user'].at[user_index, 'condition_id'].item()

        # Round Params
        difficulty0 = tabs['condition'].loc[tabs['condition']['id'] == condition_id, 'difficulty0'].item()
        difficulty1 = tabs['condition'].loc[tabs['condition']['id'] == condition_id, 'difficulty1'].item()
        nonverbal0 = tabs['condition'].loc[tabs['condition']['id'] == condition_id, 'nonverbal0'].item()
        nonverbal1 = tabs['condition'].loc[tabs['condition']['id'] == condition_id, 'nonverbal1'].item()

        #Only use users that completed
        test = tabs['survey'].loc[(tabs['survey']['round_num'] == 1) &
                                             (tabs['survey']['user_id'] == user_id), ['animacy1', 'animacy2', 'animacy3']]
        if test.shape[0] > 0:

            # For each round
            for round_num in [0, 1]:

                #Accuracy
                answers = tabs['trial'].loc[(tabs['trial']['round_num'] == round_num) &
                                                (tabs['trial']['user_id'] == user_id), ['correct_bin', 'chosen_bin']]
                answers['correct'] = np.where(answers['correct_bin'] == answers['chosen_bin'], 1, 0)
                accuracy = np.mean(answers['correct'])
                
                #Switches
                switches = tabs['trial'].loc[(tabs['trial']['round_num'] == round_num) &
                                                (tabs['trial']['user_id'] == user_id), 'switches']
                switches_avg = switches.mean()

                #Survey questions
                animacy = tabs['survey'].loc[(tabs['survey']['round_num'] == round_num) &
                                                (tabs['survey']['user_id'] == user_id), ['animacy1', 'animacy2', 'animacy3']]
                animacy_avg = animacy.mean(axis=1)

                intelligence = tabs['survey'].loc[(tabs['survey']['round_num'] == round_num) &
                                                (tabs['survey']['user_id'] == user_id), ['intelligence1', 'intelligence2']]
                intelligence_avg = intelligence.mean(axis=1)

                user_learning = tabs['survey'].loc[(tabs['survey']['round_num'] == round_num) &
                                                (tabs['survey']['user_id'] == user_id), 'user_learning'].item()

                difficulty = tabs['survey'].loc[(tabs['survey']['round_num'] == round_num) &
                                                (tabs['survey']['user_id'] == user_id), 'difficulty'].item()

                # engagement = tabs['survey'].loc[(tabs['survey']['round_num'] == round_num) &
                #                                 (tabs['survey']['user_id'] == user_id), 'engagement'].item()
                engagement = 2
                
                #Add to dataframe
                data = {'condition_id': condition_id, 'user_id': user_id, 'accuracy': accuracy, 
                        'user_learning': user_learning, 'animacy': animacy_avg, 'intelligence': intelligence_avg,
                        'difficulty': difficulty, 'engagement': engagement,
                        'answers': answers['correct'].values.tolist(), 'switches': switches_avg, 'switches_arr': switches.values.tolist()}

                if round_num == 0:
                    data['difficulty'] = difficulty0
                    data['nonverbal'] = nonverbal0
                else:
                    data['difficulty'] = difficulty1
                    data['nonverbal'] = nonverbal1
                
                df = df.append(data, ignore_index=True)
    
    return(df)

def plot_answers(data):
    fig, ax = plt.subplots(2, 1)

    #Perfect Learners
    with np.load('data/easy_50.npz') as data_perf:
        easy_all_prob=data_perf['all_prob'][:,-8:]
        easy_all_prob_best=data_perf['all_prob_best'][:,-8:]
    with np.load('data/difficult_50.npz') as data_perf:
        difficult_all_prob=data_perf['all_prob'][:,-8:]
        difficult_all_prob_best=data_perf['all_prob_best'][:,-8:]

    easy_suboptimal_avg = np.mean(easy_all_prob_best, axis=0)
    easy_optimal_avg = np.mean(easy_all_prob, axis=0)
    difficult_suboptimal_avg = np.mean(difficult_all_prob, axis=0)
    difficult_optimal_avg = np.mean(difficult_all_prob_best, axis=0)
    easy_suboptimal_std = np.std(easy_all_prob_best, axis=0)
    easy_optimal_std = np.std(easy_all_prob, axis=0)
    difficult_suboptimal_std = np.std(difficult_all_prob, axis=0)
    difficult_optimal_std = np.std(difficult_all_prob_best, axis=0)
    
    #Top Graph - separated by difficulty - perfect
    ax[0].plot(np.arange(8), easy_optimal_avg, label='Easy-Perfect-Best Card')
    ax[0].plot(np.arange(8), difficult_optimal_avg, label='Difficult-Perfect-Best Card')
    ax[0].plot(np.arange(8), easy_suboptimal_avg, label='Easy-Perfect-2nd Best Card')
    ax[0].plot(np.arange(8), difficult_suboptimal_avg, label='Difficult-Perfect-2nd Best Card')

    #Top Graph - separated by difficulty - human
    easy_answers = np.vstack(data.loc[(data['difficulty'] == 'EASY'), 'answers'].to_numpy())
    easy_answers_avg = np.mean(easy_answers,axis=0)
    easy_answers_std = np.std(easy_answers,axis=0)
    ax[0].plot(np.arange(8), easy_answers_avg, label='Easy-Human-Best Card')

    difficulty_answers = np.vstack(data.loc[(data['difficulty'] == 'DIFFICULT'), 'answers'].to_numpy())
    difficulty_answers_avg = np.mean(difficulty_answers,axis=0)
    difficulty_answers_std = np.std(difficulty_answers,axis=0)
    ax[0].plot(np.arange(8), difficulty_answers_avg, label='Difficult-Human-Best Card')

    #Bottom Graph - separated by movement - perfect
    optimal_answers = np.vstack((easy_optimal_avg, difficult_optimal_avg))
    optimal_avg = np.mean(optimal_answers, axis=0)
    optimal_std = np.std(optimal_answers, axis=0)
    ax[1].plot(np.arange(8), optimal_avg, label='Perfect-Best Card')

    suboptimal_answers = np.vstack((easy_suboptimal_avg, difficult_suboptimal_avg))
    suboptimal_avg = np.mean(suboptimal_answers, axis=0)
    suboptimal_std = np.std(suboptimal_answers, axis=0)
    ax[1].plot(np.arange(8), suboptimal_avg, label='Perfect-2nd Best Card')

    #Bottom Graph - separated by movement - human
    neutral_answers = np.vstack(data.loc[(data['nonverbal'] == 'NEUTRAL'), 'answers'].to_numpy())
    neutral_answers_avg = np.mean(neutral_answers,axis=0)
    neutral_answers_std = np.std(neutral_answers,axis=0)
    ax[1].plot(np.arange(8), neutral_answers_avg, label='Neutral-Human-Best Card')

    nonverbal_answers = np.vstack(data.loc[(data['nonverbal'] == 'NONVERBAL'), 'answers'].to_numpy())
    nonverbal_answers_avg = np.mean(nonverbal_answers,axis=0)
    nonverbal_answers_std = np.std(nonverbal_answers,axis=0)
    ax[1].plot(np.arange(8), nonverbal_answers_avg, label='Nonverbal-Human-Best Card')

    ax[0].legend()
    ax[1].legend()
    ax[0].set_ylabel('Percentage Correct')
    ax[1].set_ylabel('Percentage Correct')
    ax[1].set_xlabel('Trial Number')
    plt.show()


if __name__ == '__main__':
    tabs = read_tables()
    data = compile_data(tabs)
    # print(data.head().to_string())
    plot_answers(data)
    