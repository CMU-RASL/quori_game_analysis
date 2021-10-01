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
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd

database_filenames = ['data/study1_full.db', 'data/study1_2_full.db']
approved_ids_filenames = ['data/study1_full_prolific_approval.txt', 'data/study1_2_full_prolific_approval.txt']
easy_answers = [1, 0, 0, 1, 1, 0, 1, 1]
difficult_answers = [0, 1, 0, 0, 0, 1, 0, 0]
plt.rcParams.update({'font.size': 16})

import warnings
warnings.filterwarnings("ignore")

def read_tables():   
    # Create your connection.
    cnx1 = sqlite3.connect(database_filenames[0])
    cnx2 = sqlite3.connect(database_filenames[1])

    #Condition
    condition_tab = pd.read_sql_query("SELECT * FROM condition", cnx1)

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

    condition_tab2 = pd.read_sql_query("SELECT * FROM condition", cnx1)
    for index in range(8):
        count = condition_tab2.at[index, 'count']
        condition_tab.at[index, 'count'] += count
    
    #User
    user_tab = pd.read_sql_query("SELECT * from user", cnx1)
    user_tab2 = pd.read_sql_query("SELECT * from user", cnx2)
    user_tab2['id'] += 200
    user_tab2.index += 200
    user_tab = pd.concat([user_tab, user_tab2])

    trials_tab = pd.read_sql_query("SELECT * from trial", cnx1)
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

    trials_tab2 = pd.read_sql_query("SELECT * from trial", cnx2)
    trials_tab2['user_id'] += 200

    for index in range(trials_tab2.shape[0]):
        #Get User id
        user_id = trials_tab2.at[index, 'user_id'].item()

        #Get condition id
        condition_id = user_tab.loc[user_tab['id'] == user_id, 'condition_id'].item()
        
        #Get round
        round_num = trials_tab2.at[index, 'round_num'].item()

        #Get trial_num
        trial_num = trials_tab2.at[index, 'trial_num'].item()

        #Get difficulty
        if round_num == 0:
            difficulty = condition_tab.loc[condition_tab['id'] == condition_id, 'difficulty0'].item()
        else:
            difficulty = condition_tab.loc[condition_tab['id'] == condition_id, 'difficulty1'].item()
        
        #Get correct bin for trial
        correct_bin = RULE_PROPS[difficulty]['answers'][round_num]
        trials_tab2.at[index, 'correct_bin'] = correct_bin

        #Get rule
        trials_tab2.at[index, 'rule_set'] = RULE_PROPS[difficulty]['rule']

    trials_tab = pd.concat([trials_tab, trials_tab2])

    demos_tab = pd.read_sql_query("SELECT * from demo", cnx1)
    demos_tab2 = pd.read_sql_query("SELECT * from demo", cnx2)
    demos_tab2['user_id'] += 200
    demos_tab = pd.concat([demos_tab, demos_tab2])

    survey_tab = pd.read_sql_query("SELECT * from survey", cnx1)
    survey_tab2 = pd.read_sql_query("SELECT * from survey", cnx2)
    survey_tab2['user_id'] += 200
    survey_tab = pd.concat([survey_tab, survey_tab2])
   
    return {'condition': condition_tab, 'user': user_tab, 'trial': trials_tab, 'demo': demos_tab, 'survey': survey_tab}

def compile_data(tabs):
    age = {0: "18-24", 1: "25-34", 2: "35-44", 3: "45-54", 4: "55-64", 5: "65-74", 6: "75-84", 7: "85 or older"}
    gender = {0: "Male", 1: "Female", 2: "Other"}
    education = {0: "Less than high school degree", 1: "High school graduate (high school diploma or equivalent including GED)", 2: "Some college but no degree", 3: "Associate degree in college (2-year)", 4: "Bachelor’s degree in college (4-year)", 5: "Master’s degree", 6: "Doctoral degree", 7: "Professional degree (JD, MD)"}
    ethnicity = {0: "White", 1: "Black or African American", 2: "American Indian or Alaska Native", 3: "Asian", 4: "Native Hawaiian or Pacific Islander", 5: "Other"}
    robot = {0: "Not at all", 1: "Slightly", 2: "Moderately", 3: "Very", 4: "Extremely"}

    #Read approved_ids
    approved_ids = []
    for approved_ids_filename in approved_ids_filenames:
        with open(approved_ids_filename, 'r') as txtfile:
            lines = txtfile.readlines()
            for line in lines:
                approved_ids.append(line[:-1])

    df = pd.DataFrame(columns=('condition_id', 'user_id', 'accuracy', 'user_learning', 'animacy', 'intelligence', 'difficulty', 'engagement', 
                                'answers', 'switches', 'switches_arr', 'elapsed_time', 'elapsed_time_arr', 'last_mistake', 'animacy_arr', 'intelligence_arr', 'feedback'))
    demographics = pd.DataFrame(columns=('age', 'ethnicity', 'education', 'gender', 'robot'))
    for user_id in tabs['user']['id'].tolist():
        username = tabs['user'].loc[(tabs['user']['id'] == user_id), 'username'].item()
        
        #Add from adjustments.py!

        #Only use approved ids
        if username in approved_ids:

            #Condition Id
            condition_id = tabs['user'].loc[(tabs['user']['id'] == user_id), 'condition_id'].item()

            # Round Params
            difficulty0 = tabs['condition'].loc[tabs['condition']['id'] == condition_id, 'difficulty0'].item()
            difficulty1 = tabs['condition'].loc[tabs['condition']['id'] == condition_id, 'difficulty1'].item()
            nonverbal0 = tabs['condition'].loc[tabs['condition']['id'] == condition_id, 'nonverbal0'].item()
            nonverbal1 = tabs['condition'].loc[tabs['condition']['id'] == condition_id, 'nonverbal1'].item()

            demographics = demographics.append(tabs['user'].loc[(tabs['user']['id'] == user_id)])

            # For each round
            for round_num in [0, 1]:

                #Accuracy
                answers = tabs['trial'].loc[(tabs['trial']['round_num'] == round_num) &
                                                (tabs['trial']['user_id'] == user_id), ['correct_bin', 'chosen_bin']]
                if (round_num == 0 and difficulty0 == 'EASY') or (round_num == 1 and difficulty1 == 'EASY'):
                    answers['correct_bin'] = easy_answers
                else:
                    answers['correct_bin'] = difficult_answers
                
                answers['correct'] = np.where(answers['correct_bin'] == answers['chosen_bin'], 1, 0)
                accuracy = np.mean(answers['correct'])
                
                #Last time a mistake was made
                incorrect_ind = np.where(answers['correct'].values == 0)[0]
                if incorrect_ind.shape[0] > 0:
                    last_mistake = incorrect_ind[-1]
                else:
                    last_mistake = -1

                #Switches
                switches = tabs['trial'].loc[(tabs['trial']['round_num'] == round_num) &
                                                (tabs['trial']['user_id'] == user_id), 'switches']
                switches_avg = switches.mean()

                #Survey questions
                animacy = tabs['survey'].loc[(tabs['survey']['round_num'] == round_num) &
                                                (tabs['survey']['user_id'] == user_id), ['animacy1', 'animacy2', 'animacy3']].iloc[0]
                animacy_avg = animacy.mean()
               
                intelligence = tabs['survey'].loc[(tabs['survey']['round_num'] == round_num) &
                                                (tabs['survey']['user_id'] == user_id), ['intelligence1', 'intelligence2']].iloc[0]
                intelligence_avg = intelligence.mean()

                user_learning = tabs['survey'].loc[(tabs['survey']['round_num'] == round_num) &
                                                (tabs['survey']['user_id'] == user_id), 'user_learning'].iloc[0]


                difficulty = tabs['survey'].loc[(tabs['survey']['round_num'] == round_num) &
                                                (tabs['survey']['user_id'] == user_id), 'difficulty'].iloc[0]

                if database_filenames[0] == 'data/pilot_study1.db':
                    engagement = 2
                else:
                    engagement = tabs['survey'].loc[(tabs['survey']['round_num'] == round_num) &
                                                    (tabs['survey']['user_id'] == user_id), 'engagement'].iloc[0]
                
                #Get time per trial

                #Collect all previous trials and current trial times
                previous_trial_times = []
                current_trial_times = []

                #Last demo of round
                last_demo_time = tabs['demo'].loc[(tabs['demo']['round_num'] == round_num) &
                                        (tabs['demo']['user_id'] == user_id) &
                                        (tabs['demo']['demo_num'] == 2), 'timestamp'].iloc[0]
                last_demo_time = datetime.strptime(last_demo_time, '%Y-%m-%d %H:%M:%S.%f')
                previous_trial_times.append(last_demo_time)

                for trial_num in range(1, 9):
                    trial_time = tabs['trial'].loc[(tabs['trial']['round_num'] == round_num) &
                                        (tabs['trial']['user_id'] == user_id) &
                                        (tabs['trial']['trial_num'] == trial_num), 'timestamp'].iloc[0]
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
                
                feedback = tabs['survey'].loc[(tabs['survey']['round_num'] == round_num) &
                        (tabs['survey']['user_id'] == user_id), 'opt_text'].iloc[0]

                #Add to dataframe
                data = {'condition_id': condition_id, 'user_id': user_id, 'accuracy': accuracy, 
                        'user_learning': user_learning, 'animacy': animacy_avg, 'intelligence': intelligence_avg,
                        'perceived_difficulty': difficulty, 'engagement': engagement,
                        'answers': answers['correct'].values.tolist(), 'switches': switches_avg, 'switches_arr': switches.values.tolist(),
                        'elapsed_time': elapsed_time_avg, 'elapsed_time_arr': elapsed_time,
                        'last_mistake': last_mistake, 'animacy_arr': animacy, 'intelligence_arr': intelligence,
                        'feedback': feedback, 'round': round_num}

                if round_num == 0:
                    data['difficulty'] = difficulty0
                    data['nonverbal'] = nonverbal0
                else:
                    data['difficulty'] = difficulty1
                    data['nonverbal'] = nonverbal1

                df = df.append(data, ignore_index=True)
    cols=['condition_id', 'user_id', 'user_learning', 'engagement', 'last_mistake']
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce', axis=1)
    demographics = demographics.replace({'age': age, 'robot': robot, 'ethnicity': ethnicity, 'education': education, 'gender': gender})
    return(df, demographics)

def plot_answers(data):
    fig, ax = plt.subplots(2, 1, sharey = True)

    #Perfect Learners
    with np.load('data/easy_50.npz') as data_perf:
        easy_all_prob_best=data_perf['all_prob_best'][:,-8:]
    with np.load('data/difficult_50.npz') as data_perf:
        difficult_all_prob_best=data_perf['all_prob_best'][:,-8:]

    easy_optimal_avg = np.mean(easy_all_prob_best, axis=0)
    difficult_optimal_avg = np.mean(difficult_all_prob_best, axis=0)
    easy_optimal_std = np.std(easy_all_prob_best, axis=0)
    difficult_optimal_std = np.std(difficult_all_prob_best, axis=0)
    
    #Separated by difficulty - perfect
    ax[0].errorbar(np.arange(8), easy_optimal_avg, yerr= easy_optimal_std,label='Easy-Perfect-Best Card')
    ax[1].errorbar(np.arange(8), difficult_optimal_avg, yerr= difficult_optimal_std,label='Difficult-Perfect-Best Card')

    #Separated by difficulty - human
    easy_answers = np.vstack(data.loc[(data['difficulty'] == 'EASY'), 'answers'].to_numpy())
    easy_answers_avg = np.mean(easy_answers,axis=0)
    easy_answers_std = np.std(easy_answers,axis=0)
    ax[0].errorbar(np.arange(8), easy_answers_avg, yerr= easy_answers_std,label='Easy-Human-Best Card')

    difficulty_answers = np.vstack(data.loc[(data['difficulty'] == 'DIFFICULT'), 'answers'].to_numpy())
    difficulty_answers_avg = np.mean(difficulty_answers,axis=0)
    difficulty_answers_std = np.std(difficulty_answers,axis=0)
    ax[1].errorbar(np.arange(8), difficulty_answers_avg, yerr= difficulty_answers_std,label='Difficult-Human-Best Card')

    ax[0].legend(loc='lower right')
    ax[1].legend(loc='lower right')
    ax[0].set_ylabel('Percentage Correct', )
    ax[1].set_ylabel('Percentage Correct')
    ax[1].set_xlabel('Trial Number')
    plt.show()

def plot_time_series(data, col_name, name):
    fig, ax = plt.subplots(2, 1, sharey=True)

    #Top Graph - separated by difficulty - human
    easy_answers = np.vstack(data.loc[(data['difficulty'] == 'EASY'), col_name].to_numpy())
    easy_answers_avg = np.mean(easy_answers,axis=0)
    easy_answers_std = np.std(easy_answers,axis=0)
    ax[0].errorbar(np.arange(8), easy_answers_avg, yerr=easy_answers_std, label='Easy-Human-Best Card')

    difficulty_answers = np.vstack(data.loc[(data['difficulty'] == 'DIFFICULT'), col_name].to_numpy())
    difficulty_answers_avg = np.mean(difficulty_answers,axis=0)
    difficulty_answers_std = np.std(difficulty_answers,axis=0)
    ax[0].errorbar(np.arange(8), difficulty_answers_avg, yerr=difficulty_answers_std, label='Difficult-Human-Best Card')

    #Bottom Graph - separated by movement
    neutral_answers = np.vstack(data.loc[(data['nonverbal'] == 'NEUTRAL'), col_name].to_numpy())
    neutral_answers_avg = np.mean(neutral_answers,axis=0)
    neutral_answers_std = np.std(neutral_answers,axis=0)
    ax[1].errorbar(np.arange(8), neutral_answers_avg, yerr=neutral_answers_std, label='Neutral-Human-Best Card')

    nonverbal_answers = np.vstack(data.loc[(data['nonverbal'] == 'NONVERBAL'), col_name].to_numpy())
    nonverbal_answers_avg = np.mean(nonverbal_answers,axis=0)
    nonverbal_answers_std = np.std(nonverbal_answers,axis=0)
    ax[1].errorbar(np.arange(8), nonverbal_answers_avg, yerr=nonverbal_answers_std, label='Nonverbal-Human-Best Card')

    ax[0].legend()
    ax[1].legend()
    ax[0].set_ylabel(name)
    ax[1].set_ylabel(name)
    ax[1].set_xlabel('Trial Number')
    plt.show()

def two_way_anova(data, col_name, title):
    fig, ax = plt.subplots()

    easy = np.vstack(data.loc[(data['difficulty'] == 'EASY'), col_name].to_numpy())
    easy_mean = np.mean(easy,axis=0)[0]
    easy_std = np.std(easy,axis=0)[0]
    
    difficult = np.vstack(data.loc[(data['difficulty'] == 'DIFFICULT'), col_name].to_numpy())
    difficult_mean = np.mean(difficult,axis=0)[0]
    difficult_std = np.std(difficult,axis=0)[0]

    neutral = np.vstack(data.loc[(data['nonverbal'] == 'NEUTRAL'), col_name].to_numpy())
    neutral_mean = np.mean(neutral,axis=0)[0]
    neutral_std = np.std(neutral,axis=0)[0]

    nonverbal = np.vstack(data.loc[(data['nonverbal'] == 'NONVERBAL'), col_name].to_numpy())
    nonverbal_mean = np.mean(nonverbal,axis=0)[0]
    nonverbal_std = np.std(nonverbal,axis=0)[0]
    
    labels = ['Easy', 'Difficult', 'Neutral', 'Affective']
    x_pos = np.arange(4)
    means = [easy_mean, difficult_mean, neutral_mean, nonverbal_mean]
    stds = [easy_std, difficult_std, neutral_std, nonverbal_std]
    
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
    plt.show()

    model = ols(col_name + ' ~ C(difficulty) + C(nonverbal) + C(difficulty):C(nonverbal)', data=data).fit()
    anova_table = sm.stats.anova_lm(model, type=2)
    print('---')
    print(col_name)
    print(anova_table)
    print('---')

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
            df_p.at[row_name, col_name] = np.where(np.array(p_values) < 0.05)
    
    print(df_p)

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
            df_p.at[row_name, col_name] = np.where(np.array(p_values) < 0.05)
    
    print(df_p)

def expected_accuracy():
    #Perfect Learners
    with np.load('data/easy_50.npz') as data_perf:
        easy_all_prob=data_perf['all_prob'][:,-8:]
        easy_all_prob_best=data_perf['all_prob_best'][:,-8:]
    with np.load('data/difficult_50.npz') as data_perf:
        difficult_all_prob=data_perf['all_prob'][:,-8:]
        difficult_all_prob_best=data_perf['all_prob_best'][:,-8:]
    
    easy = np.mean(easy_all_prob)
    easy_best = np.mean(easy_all_prob_best)
    difficult = np.mean(difficult_all_prob)
    difficult_best = np.mean(difficult_all_prob_best)
    print(easy, easy_best, difficult, difficult_best)

def cronbach(data):
    #Animacy
    subj = []
    items = []
    scores = []

    for row_ind in range(data.shape[0]):
        if not row_ind in [86, 200, 213, 262]:
            scores.append(data.at[row_ind, 'animacy_arr'][0])
            scores.append(data.at[row_ind, 'animacy_arr'][1])
            scores.append(data.at[row_ind, 'animacy_arr'][2])

            subj.append(str(data.at[row_ind, 'user_id'])+'-'+str(data.at[row_ind, 'difficulty'])+str(data.at[row_ind, 'nonverbal']))
            items.append('animacy1')
            
            subj.append(str(data.at[row_ind, 'user_id'])+'-'+str(data.at[row_ind, 'difficulty'])+str(data.at[row_ind, 'nonverbal']))
            items.append('animacy2')
            
            subj.append(str(data.at[row_ind, 'user_id'])+'-'+str(data.at[row_ind, 'difficulty'])+str(data.at[row_ind, 'nonverbal']))
            items.append('animacy3')
        
    
    animacy_data = pd.DataFrame({'subj': subj, 'items': items, 'scores': scores})
    print('Animacy')
    print(pg.cronbach_alpha(data=animacy_data, items='items', scores='scores', subject='subj'))

    #Intelligence
    subj = []
    items = []
    scores = []
    for row_ind in range(data.shape[0]):
        if not row_ind in [86, 200, 213, 262]:
            scores.append(data.at[row_ind, 'intelligence_arr'][0])
            scores.append(data.at[row_ind, 'intelligence_arr'][1])

            subj.append(str(data.at[row_ind, 'user_id'])+'-'+str(data.at[row_ind, 'difficulty'])+str(data.at[row_ind, 'nonverbal']))
            items.append('intelligence1')
            
            subj.append(str(data.at[row_ind, 'user_id'])+'-'+str(data.at[row_ind, 'difficulty'])+str(data.at[row_ind, 'nonverbal']))
            items.append('intelligence2')
            
        
    intelligence_data = pd.DataFrame({'subj': subj, 'items': items, 'scores': scores})
    print('Intelligence')
    print(pg.cronbach_alpha(data=intelligence_data, items='items', scores='scores', subject='subj'))

def sorted_feedback(data):
    difficulty = ['EASY', 'DIFFICULT']
    nonverbal = ['NEUTRAL', 'NONVERBAL']
    res = {'EASY': {}, 'DIFFICULT': {}, 'NEUTRAL': {}, 'NONVERBAL': {}}
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
                        # print('Round 0: ' + diff + '-' + non + '\tCurrent Round Feedback: ' + row['feedback'])
                        # print(row['feedback'])
                    else:
                        # print('\tCurrent Round Feedback: ' + row['feedback'])
                        # print(row['feedback'])
                        pass
                    
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


    # #Get 50 most frequent words
    # sorted_list = [(k, v) for k, v in sorted(all.items(), key=lambda item: item[1])]
    
    # no_list = ['the', 'I', 'was', 'but', 'it', 'to', 'The', 'of', 'a', 'this', 'did', 'there', 'It', 'about', 'as', 'on', 'be', 'would', 'in', 'that', 'This', 'have', 'had', 'and', 'not', 'very', 'is', 'out', 'i', 'still', 'just', 'than', 'when', "didn't", 'first', 'robot', 'me', 'rule.', 'or', 'at', 'bit', 'because', 'if', 'what', 'really', 'game', 'figure', 'which', 'could', 'rule', 'game.', 'rules', 'little', 'were', 'made', 'second', 'playing', 'pattern', 'for', 'so', "don't", 'by', 'one', 'with', 'my', 'time.', 'round.', 'few', 'do', "I'm", 'last', 'rule,', 'took', 'you', 'then', "wasn't", 'found', 'game', 'after', 'no', "it's", 'where', 'maybe', 'round,', 'clear', 'shapes', 'responses', 'robots', 'been', 'only', 'game,', 'up', 'some', "didn't", 'rounds', 'it.', 'cards', 'he', 'less', 'know', 'two', 'was.', 'around', 'an', 'end', 'patterns', 'card', 'fairly', 'since', 'lot', 'move', 'how', 'able', 'time']
    # all_dist = {}
    # tot = 40
    # cur = 0
    # for word, num in reversed(sorted_list):
    #     if not word in no_list:
    #         all_dist[word] = num
    #         cur += 1
    #     if cur == tot:
    #         break
    
    # colors = ["green", "blue", "red", "turquoise"]

    # # loop over the dictionary keys to plot each distribution
    # for i, label in enumerate(res):
    #     frequency = [res[label][term] for term in all_dist.keys()]
    #     color = colors[i]
    #     plt.plot(frequency, color=color, label=label)
    # plt.gca().grid(True)
    # plt.xticks(np.arange(0, len(all_dist.keys()), 1), all_dist.keys(), rotation=90)
    # plt.xlabel("Most common terms")
    # plt.ylabel("Frequency")
    # plt.legend(loc="upper right")
    # plt.show()

    # word_groups = [('learn', 'learned', 'learns'), ('understand', 'understood', 'understands'), ('easy', 'easier', 'simple', 'simpler'), ('difficult', 'difficulty')]
    word_groups = [('enjoy', 'enjoyed', 'enjoys', 'lively', 'like', 'helpful', 'helped', 'helps', 'likes', 'liked', 'felt', 'feels')]
    for group in word_groups:
        print('-------------------------------')
        print(group)
        # #Easy
        # print('----------EASY')
        # feedback = data.loc[(data['difficulty'] == 'EASY'), ['feedback']]
        # for ind, row in feedback.iterrows():
        #     str_list = row['feedback'].split()
        #     for word in group:
        #         if word in str_list:
        #             print(row['feedback'])
        #             break
        # #Difficult
        # print('----------DIFFICULT')
        # feedback = data.loc[(data['difficulty'] == 'DIFFICULT'), ['feedback']]
        # for ind, row in feedback.iterrows():
        #     str_list = row['feedback'].split()
        #     for word in group:
        #         if word in str_list:
        #             print(row['feedback'])
        #             break

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
        print('----------AFFECTIVE')
        feedback = data.loc[(data['nonverbal'] == 'NONVERBAL'), ['feedback']]
        for ind, row in feedback.iterrows():
            str_list = row['feedback'].split()
            for word in group:
                if word in str_list and 'robot' in str_list:
                    print(row['feedback'])
                    break

        print('-------------------------------')

def tukey_test(data):
    data['combo'] = data['difficulty'] + " / " + data['nonverbal']
    m_comp = pairwise_tukeyhsd(endog=data['accuracy'], groups=data['combo'], alpha=0.01)
    tukey_data = pd.DataFrame(data=m_comp._results_table.data[1:], columns = m_comp._results_table.data[0])
    print(tukey_data)

if __name__ == '__main__':
    tabs = read_tables()
    data, demographics = compile_data(tabs)


    # print((np.mean(data['accuracy']) - 1/8)/np.std(data['accuracy']))
    data = data.loc[data['accuracy'] > 1/8]
    # plot_answers(data)
    # plot_time_series(data, 'elapsed_time_arr', 'Elapsed Time (sec)')
    # plot_time_series(data, 'switches_arr', 'Number of Switches')

    # two_way_anova(data, 'accuracy', 'Accuracy')
    # two_way_anova(data, 'user_learning', 'User Learning')
    # two_way_anova(data, 'perceived_difficulty', 'Perceived Difficulty')
    # two_way_anova(data, 'engagement', 'Engagement')
    # two_way_anova(data, 'animacy', 'Animacy')
    # two_way_anova(data, 'intelligence', 'Intelligence')
    # two_way_anova(data, 'elapsed_time', 'Elapsed Time')
    # tukey_test(data[['accuracy', 'difficulty', 'nonverbal']])

    # compare_dists_answers(data)
    # compare_dists_other(data, 'elapsed_time_arr')
    # expected_accuracy()
    
    # cronbach(data)
    # sorted_feedback(data)