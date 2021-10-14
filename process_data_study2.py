import sqlite3
import pandas as pd
from data.params_study2 import *
import os
from datetime import datetime
import numpy as np

database_filename = 'data/study2_test.db'
features_filename = 'data/study2_test_users/'
easy_answers = [1, 0, 0, 1, 1, 0, 1, 1]
difficult_answers = [0, 1, 0, 0, 0, 1, 0, 0]

def read_tables():   
    # Create your connection.
    cnx1 = sqlite3.connect(database_filename)

    #Condition
    condition_tab = pd.read_sql_query("SELECT * FROM condition", cnx1)

    #Add new columns, for each round
    condition_tab['difficulty0'] = ['', '']
    condition_tab['difficulty1'] = ['', '']
    condition_tab['nonverbal0'] = ['', '']
    condition_tab['nonverbal1'] = ['', '']
    for index in range(condition_tab.shape[0]):
        condition_tab.at[index, 'difficulty'] = (CONDITIONS[index][0][0], CONDITIONS[index][1][0])
        condition_tab.at[index, 'nonverbal'] = (CONDITIONS[index][0][1], CONDITIONS[index][1][1])
        condition_tab.at[index, 'difficulty0'] = CONDITIONS[index][0][0]
        condition_tab.at[index, 'difficulty1'] = CONDITIONS[index][1][0]
        condition_tab.at[index, 'nonverbal0'] = CONDITIONS[index][0][1]
        condition_tab.at[index, 'nonverbal1'] = CONDITIONS[index][1][1]
    
    #User
    user_tab = pd.read_sql_query("SELECT * from user", cnx1)

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

    demos_tab = pd.read_sql_query("SELECT * from demo", cnx1)

    survey_tab = pd.read_sql_query("SELECT * from survey", cnx1)
   
    return {'condition': condition_tab, 'user': user_tab, 'trial': trials_tab, 'demo': demos_tab, 'survey': survey_tab}

def read_features():
    features = []
    for user_folder in os.listdir(features_filename):
        features_folder = '{}{}/features'.format(features_filename, user_folder)
        user_id = features_folder.split('/')[-2]
        for feature_file in os.listdir(features_folder):
            cur_feat = pd.read_csv('{}/{}'.format(features_folder, feature_file))
            if (cur_feat.shape[0] == 1):
                if not pd.isna(cur_feat['FaceRectX'].item()):
                    cur_feat['user'] = user_id

                    timestamp = cur_feat['input'].item().split('/')[-1][5:-4]
                    timestamp = datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S")
                    cur_feat['timestamp'] = timestamp

                    features.append(cur_feat)
                    
    features_tab = pd.concat(features)
    return features_tab

def compile_data(tabs, features):

    data = []

    for user_id in tabs['user']['id'].tolist():

        #Check if final survey is completed
        final_survey = tabs['survey'].loc[(tabs['survey']['user_id'] == user_id) & (tabs['survey']['round_num'] == 1)]
        if final_survey.shape[0] == 1:

            #Get condition
            condition_id = tabs['user'].loc[(tabs['user']['id'] == user_id), 'condition_id'].item()

            # Round Params
            difficulty0 = tabs['condition'].loc[tabs['condition']['id'] == condition_id, 'difficulty0'].item()
            difficulty1 = tabs['condition'].loc[tabs['condition']['id'] == condition_id, 'difficulty1'].item()
            nonverbal0 = tabs['condition'].loc[tabs['condition']['id'] == condition_id, 'nonverbal0'].item()
            nonverbal1 = tabs['condition'].loc[tabs['condition']['id'] == condition_id, 'nonverbal1'].item()

            #Get start and end times
            start_time = []
            end_time = []
            answers_arr = []
            confidence_arr = []
            rounds = 2*[1, 2, 3, 4, 5, 6, 7, 8]
            user_arr = 16*[user_id]
            features_arr = []
            for round_num in [0, 1]:
                
                #Accuracy
                answers = tabs['trial'].loc[(tabs['trial']['round_num'] == round_num) &
                                                (tabs['trial']['user_id'] == user_id), ['correct_bin', 'chosen_bin']]
                if (round_num == 0 and difficulty0 == 'EASY') or (round_num == 1 and difficulty1 == 'EASY'):
                    answers['correct_bin'] = easy_answers
                else:
                    answers['correct_bin'] = difficult_answers
                
                answers_arr.extend(np.where(answers['correct_bin'] == answers['chosen_bin'], 1, 0))

                #Get time of 2nd demo
                demo_time = tabs['demo'].loc[(tabs['demo']['user_id'] == user_id) & 
                                        (tabs['demo']['round_num'] == round_num) &
                                        (tabs['demo']['demo_num'] == 2), 'timestamp'].iloc[0]
                demo_time = datetime.strptime(demo_time, '%Y-%m-%d %H:%M:%S.%f')
                
                start_time.append(demo_time)
                #All trials
                for trial_num in range(1, 9):
                    trial_time = tabs['trial'].loc[(tabs['trial']['user_id'] == user_id) &
                                            (tabs['trial']['round_num'] == round_num) &
                                            (tabs['trial']['trial_num'] == trial_num), 'timestamp'].iloc[0]
                    trial_time = datetime.strptime(trial_time, '%Y-%m-%d %H:%M:%S.%f')
                    end_time.append(trial_time)
                    if trial_num < 8:
                        start_time.append(trial_time)
                    

                    confidence = tabs['trial'].loc[(tabs['trial']['user_id'] == user_id) &
                                            (tabs['trial']['round_num'] == round_num) &
                                            (tabs['trial']['trial_num'] == trial_num), 'confidence'].iloc[0]
                    confidence_arr.append(confidence)
                        
            for start, end in zip(start_time, end_time):
                features_arr.append(features.loc[(features['user'] == str(user_id)) &
                                    (features['timestamp'] >= start) & 
                                    (features['timestamp'] <= end)])


            df = pd.DataFrame({'start_time': start_time, 'end_time': end_time, 'answer': answers_arr, 'confidence': confidence, 'round': rounds, 'user': user_arr, 'features': features_arr})
            data.append(df)
    
    data = pd.concat(data)
    return(data)


if __name__ == '__main__':
    tabs = read_tables()
    features = read_features()

    data = compile_data(tabs, features)

    