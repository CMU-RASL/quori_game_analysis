import sqlite3
import pandas as pd
from data.params_study1 import *
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from data.utils import *
from datetime import datetime
import pickle as pkl


database_filenames = ['data/study1b_1.db', 'data/study1b_2.db']
approved_ids_filenames = ['data/study1b_1_prolific_approval.txt', 'data/study1b_2_prolific_approval.txt']
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
    user_tab2['id'] += 500
    user_tab2.index += 500
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
    trials_tab2['user_id'] += 500

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
    demos_tab2['user_id'] += 500
    demos_tab = pd.concat([demos_tab, demos_tab2])

    survey_tab = pd.read_sql_query("SELECT * from survey", cnx1)
    survey_tab2 = pd.read_sql_query("SELECT * from survey", cnx2)
    survey_tab2['user_id'] += 500
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
        username = tabs['user'].loc[(tabs['user']['id'] == user_id), 'username'].iloc[0]
        
        #Add from adjustments.py!
        if username == 'qhubanonkwenkwezi@gmail.com ':
            username = '61af8a4bc54a7c9c17c7498f'
        if username == 'kgomoalbert@gmail.com':
            username = '614888949bc056b089fee592'
        if username == '616d4ba6893d91079ca83519@email.prolific.co':
            username = '616d4ba6893d91079ca83519'
        if username == 'igpapadi@csd.auth.gr':
            username = '607d99aa9d92febb1a714f8d'
        if username == 'Eraje':
            username = '60fb0cd3aa10c5ef5100190b'

        #Only use approved ids
        if username in approved_ids:

            #Condition Id
            condition_id = tabs['user'].loc[(tabs['user']['id'] == user_id), 'condition_id'].iloc[0]
            
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
                if user_id == 140:
                    last_demo_time = tabs['demo'].loc[(tabs['demo']['round_num'] == round_num) &
                                        (tabs['demo']['user_id'] == user_id) &
                                        (tabs['demo']['demo_num'] == 1), 'timestamp'].iloc[0]
                else:
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


if __name__ == '__main__':
    tabs = read_tables()
    data, demographics = compile_data(tabs)
    
    data = data.loc[data['accuracy'] > 1/8]
    filehandler = open('data/study1b_data.pkl',"wb")
    pkl.dump(data,filehandler)
    filehandler.close()