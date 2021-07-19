import sqlite3
import pandas as pd
from data.params_study1 import *
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from data.utils import *
from os import path
from datetime import datetime
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import ttest_ind

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
                
                #Get time per trial

                #Collect all previous trials and current trial times
                previous_trial_times = []
                current_trial_times = []

                #Last demo of round
                last_demo_time = tabs['demo'].loc[(tabs['demo']['round_num'] == round_num) &
                                        (tabs['demo']['user_id'] == user_id) &
                                        (tabs['demo']['demo_num'] == 2), 'timestamp'].item()
                last_demo_time = datetime.strptime(last_demo_time, '%Y-%m-%d %H:%M:%S.%f')
                previous_trial_times.append(last_demo_time)

                for trial_num in range(1, 9):
                    trial_time = tabs['trial'].loc[(tabs['trial']['round_num'] == round_num) &
                                        (tabs['trial']['user_id'] == user_id) &
                                        (tabs['trial']['trial_num'] == trial_num), 'timestamp'].item()
                    trial_time = datetime.strptime(trial_time, '%Y-%m-%d %H:%M:%S.%f')
                    if trial_num < 8:
                        previous_trial_times.append(trial_time)
                    current_trial_times.append(trial_time)
                
                elapsed_time = []
                for previous_time, current_time in zip(previous_trial_times, current_trial_times):
                    if ((current_time - previous_time).total_seconds()) > 60:
                        elapsed_time.append(60.0)
                    else:
                        elapsed_time.append((current_time - previous_time).total_seconds())
                elapsed_time_avg = np.mean(elapsed_time)

                #Add to dataframe
                data = {'condition_id': condition_id, 'user_id': user_id, 'accuracy': accuracy, 
                        'user_learning': user_learning, 'animacy': animacy_avg, 'intelligence': intelligence_avg,
                        'difficulty': difficulty, 'engagement': engagement,
                        'answers': answers['correct'].values.tolist(), 'switches': switches_avg, 'switches_arr': switches.values.tolist(),
                        'elapsed_time': elapsed_time_avg, 'elapsed_time_arr': elapsed_time}

                if round_num == 0:
                    data['difficulty'] = difficulty0
                    data['nonverbal'] = nonverbal0
                else:
                    data['difficulty'] = difficulty1
                    data['nonverbal'] = nonverbal1
                
                df = df.append(data, ignore_index=True)
    
    return(df)

def plot_answers(data):
    fig, ax = plt.subplots(3, 1)

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
    ax[1].plot(np.arange(8), difficult_optimal_avg, label='Difficult-Perfect-Best Card')
    ax[0].plot(np.arange(8), easy_suboptimal_avg, label='Easy-Perfect-2nd Best Card')
    ax[1].plot(np.arange(8), difficult_suboptimal_avg, label='Difficult-Perfect-2nd Best Card')

    #Top Graph - separated by difficulty - human
    easy_answers = np.vstack(data.loc[(data['difficulty'] == 'EASY'), 'answers'].to_numpy())
    easy_answers_avg = np.mean(easy_answers,axis=0)
    easy_answers_std = np.std(easy_answers,axis=0)
    ax[0].plot(np.arange(8), easy_answers_avg, label='Easy-Human-Best Card')

    difficulty_answers = np.vstack(data.loc[(data['difficulty'] == 'DIFFICULT'), 'answers'].to_numpy())
    difficulty_answers_avg = np.mean(difficulty_answers,axis=0)
    difficulty_answers_std = np.std(difficulty_answers,axis=0)
    ax[1].plot(np.arange(8), difficulty_answers_avg, label='Difficult-Human-Best Card')

    #Bottom Graph - separated by movement - perfect
    optimal_answers = np.vstack((easy_optimal_avg, difficult_optimal_avg))
    optimal_avg = np.mean(optimal_answers, axis=0)
    optimal_std = np.std(optimal_answers, axis=0)
    ax[2].plot(np.arange(8), optimal_avg, label='Perfect-Best Card')

    suboptimal_answers = np.vstack((easy_suboptimal_avg, difficult_suboptimal_avg))
    suboptimal_avg = np.mean(suboptimal_answers, axis=0)
    suboptimal_std = np.std(suboptimal_answers, axis=0)
    ax[2].plot(np.arange(8), suboptimal_avg, label='Perfect-2nd Best Card')

    #Bottom Graph - separated by movement - human
    neutral_answers = np.vstack(data.loc[(data['nonverbal'] == 'NEUTRAL'), 'answers'].to_numpy())
    neutral_answers_avg = np.mean(neutral_answers,axis=0)
    neutral_answers_std = np.std(neutral_answers,axis=0)
    ax[2].plot(np.arange(8), neutral_answers_avg, label='Neutral-Human-Best Card')

    nonverbal_answers = np.vstack(data.loc[(data['nonverbal'] == 'NONVERBAL'), 'answers'].to_numpy())
    nonverbal_answers_avg = np.mean(nonverbal_answers,axis=0)
    nonverbal_answers_std = np.std(nonverbal_answers,axis=0)
    ax[2].plot(np.arange(8), nonverbal_answers_avg, label='Nonverbal-Human-Best Card')

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[0].set_ylabel('Percentage Correct')
    ax[1].set_ylabel('Percentage Correct')
    ax[2].set_ylabel('Percentage Correct')
    ax[2].set_xlabel('Trial Number')
    plt.show()

def plot_time_series(data, col_name, name):
    fig, ax = plt.subplots(2, 1)

    #Top Graph - separated by difficulty - human
    easy_answers = np.vstack(data.loc[(data['difficulty'] == 'EASY'), col_name].to_numpy())
    easy_answers_avg = np.mean(easy_answers,axis=0)
    easy_answers_std = np.std(easy_answers,axis=0)
    ax[0].plot(np.arange(8), easy_answers_avg, label='Easy-Human-Best Card')

    difficulty_answers = np.vstack(data.loc[(data['difficulty'] == 'DIFFICULT'), col_name].to_numpy())
    difficulty_answers_avg = np.mean(difficulty_answers,axis=0)
    difficulty_answers_std = np.std(difficulty_answers,axis=0)
    ax[0].plot(np.arange(8), difficulty_answers_avg, label='Difficult-Human-Best Card')

    #Bottom Graph - separated by movement
    neutral_answers = np.vstack(data.loc[(data['nonverbal'] == 'NEUTRAL'), col_name].to_numpy())
    neutral_answers_avg = np.mean(neutral_answers,axis=0)
    neutral_answers_std = np.std(neutral_answers,axis=0)
    ax[1].plot(np.arange(8), neutral_answers_avg, label='Neutral-Human-Best Card')

    nonverbal_answers = np.vstack(data.loc[(data['nonverbal'] == 'NONVERBAL'), col_name].to_numpy())
    nonverbal_answers_avg = np.mean(nonverbal_answers,axis=0)
    nonverbal_answers_std = np.std(nonverbal_answers,axis=0)
    ax[1].plot(np.arange(8), nonverbal_answers_avg, label='Nonverbal-Human-Best Card')

    ax[0].legend()
    ax[1].legend()
    ax[0].set_ylabel(name)
    ax[1].set_ylabel(name)
    ax[1].set_xlabel('Trial Number')
    plt.show()

def two_way_anova(data, col_name):
    model = ols(col_name + ' ~ C(difficulty) + C(nonverbal) + C(difficulty):C(nonverbal)', data=data).fit()
    anova_table = sm.stats.anova_lm(model, type=2)
    print('---')
    print(col_name)
    print(anova_table)
    print('---')
# model = ols(' ~ C(Genotype) + C(years) + C(Genotype):C(years)', data=d_melt).fit()
# anova_table = sm.stats.anova_lm(model, typ=2)

def compare_dists_answers(data):
    #Perfect Learners
    with np.load('data/easy_50.npz') as data_perf:
        easy_all_prob=data_perf['all_prob'][:,-8:]
        easy_all_prob_best=data_perf['all_prob_best'][:,-8:]
    with np.load('data/difficult_50.npz') as data_perf:
        difficult_all_prob=data_perf['all_prob'][:,-8:]
        difficult_all_prob_best=data_perf['all_prob_best'][:,-8:]
    
    #Human Data
    easy_answers = np.vstack(data.loc[(data['difficulty'] == 'EASY'), 'answers'].to_numpy())
    difficult_answers = np.vstack(data.loc[(data['difficulty'] == 'DIFFICULT'), 'answers'].to_numpy())
    neutral_answers = np.vstack(data.loc[(data['nonverbal'] == 'NEUTRAL'), 'answers'].to_numpy())
    nonverbal_answers = np.vstack(data.loc[(data['nonverbal'] == 'NONVERBAL'), 'answers'].to_numpy())

    series = [easy_all_prob, easy_all_prob_best, difficult_all_prob, difficult_all_prob_best, easy_answers, difficult_answers, neutral_answers, nonverbal_answers]
    names = ['Easy-Perfect-Best Card', 'Easy-Perfect-2nd Best Card', 'Difficult-Perfect-Best Card', 'Difficult-Perfect-2nd Best Card', 'Easy-Human-Best Card', 'Difficulty-Human-Best Card', 'Neutral-Human-Best Card', 'Nonverbal-Human-Best-Card']

    df_t = pd.DataFrame(columns=names, index=names)
    df_p = pd.DataFrame(columns=names, index=names)
    alpha = 0.05
    for row_ind, row_name in enumerate(names):
        for col_ind, col_name in enumerate(names):
            t_values = []
            p_values = []
            for trial_num in range(8):
                p = series[row_ind][:,trial_num]
                q = series[col_ind][:,trial_num]
                t_values.append(ttest_ind(p, q, equal_var=False)[0])
                p_values.append(ttest_ind(p, q, equal_var=False)[1])
            df_t.at[row_name, col_name] = t_values
            df_p.at[row_name, col_name] = p_values
    
    print(np.round(df_p.at['Easy-Perfect-Best Card', 'Easy-Human-Best Card'], decimals=3))

def compare_dists_other(data, col_name):
    easy_answers = np.vstack(data.loc[(data['difficulty'] == 'EASY'), col_name].to_numpy())
    difficult_answers = np.vstack(data.loc[(data['difficulty'] == 'DIFFICULT'), col_name].to_numpy())
    neutral_answers = np.vstack(data.loc[(data['nonverbal'] == 'NEUTRAL'), col_name].to_numpy())
    nonverbal_answers = np.vstack(data.loc[(data['nonverbal'] == 'NONVERBAL'), col_name].to_numpy())

    series = [easy_answers, difficult_answers, neutral_answers, nonverbal_answers]
    names = ['Easy-Human-Best Card', 'Difficulty-Human-Best Card', 'Neutral-Human-Best Card', 'Nonverbal-Human-Best-Card']

    df_t = pd.DataFrame(columns=names, index=names)
    df_p = pd.DataFrame(columns=names, index=names)
    alpha = 0.05
    for row_ind, row_name in enumerate(names):
        for col_ind, col_name in enumerate(names):
            t_values = []
            p_values = []
            for trial_num in range(8):
                p = series[row_ind][:,trial_num]
                q = series[col_ind][:,trial_num]
                t_values.append(ttest_ind(p, q, equal_var=False)[0])
                p_values.append(ttest_ind(p, q, equal_var=False)[1])
            df_t.at[row_name, col_name] = t_values
            df_p.at[row_name, col_name] = p_values
    
    print(df_p)

if __name__ == '__main__':
    tabs = read_tables()
    data = compile_data(tabs)
    # plot_answers(data)
    # plot_time_series(data, 'elapsed_time_arr', 'Elapsed Time (sec)')
    # plot_time_series(data, 'switches_arr', 'Number of Switches')
    # two_way_anova(data, 'elapsed_time')
    # compare_dists_answers(data)
    # compare_dists_other(data, 'elapsed_time_arr')
    # compare_dists_other(data, 'switches_arr')
    