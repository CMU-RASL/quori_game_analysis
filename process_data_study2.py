import sqlite3
import pandas as pd
import os
from datetime import datetime
import numpy as np
import pickle as pkl

database_filename = 'data/study2.db'
features_filename = 'data/study2_users/'

easy_answers = [1, 0, 0, 1, 1, 0, 1, 1]
difficult_answers = [0, 1, 0, 0, 0, 1, 0, 0]
CONDITIONS = [(('EASY', 'NONVERBAL'), ('DIFFICULT', 'NONVERBAL')), (('DIFFICULT', 'NONVERBAL'), ('EASY', 'NONVERBAL'))]
RULE_PROPS = {'EASY': {'rule': 'diamonds on left, all others on right', 'demo_cards': [54, 60], 'cards': [48, 47, 10, 58, 14, 18, 57, 32], 'demo_answers': [0, 1], 'answers': [1, 0, 0, 1, 1, 0, 1, 1]},
            'DIFFICULT': {'rule': 'green-one, red/purple on left, green two/three on right', 'demo_cards': [61, 34], 'cards': [33, 32, 42, 17, 68, 29, 26, 45],'demo_answers': [0, 1], 'answers': [0, 1, 0, 0, 0, 1, 0, 0]}}
pd.options.mode.chained_assignment = None
PERFECT = {'EASY': [0.52, 0.26, 0.39, 0.5,  1.,   1. ,  1.,   1.  ], 'DIFFICULT': [0.08, 0.32, 1. ,  1.  , 1.  , 1. ,  1. ,  1.  ]}
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

def read_facial(valid_users):
    features = []
    num = 0
    for user_folder in os.listdir(features_filename):
        features_folder = '{}{}/features'.format(features_filename, user_folder)
        user_id = features_folder.split('/')[-2]
        if int(user_id) in valid_users:
            for feature_file in os.listdir(features_folder):
                cur_feat = pd.read_csv('{}/{}'.format(features_folder, feature_file))
                if (cur_feat.shape[0] == 1):
                    if not pd.isna(cur_feat['FaceRectX'].item()):
                        cur_feat['user'] = user_id
                        timestamp = cur_feat['input'].item()[-23:-4]
                        timestamp = datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S")
                        cur_feat['timestamp'] = timestamp

                        features.append(cur_feat)
            num += 1
            print('{}/{}'.format(num, len(valid_users)))
    features_tab = pd.concat(features)
    return features_tab

def read_contextual(tabs):

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
            trial_arr = 2*[1, 2, 3, 4, 5, 6, 7, 8]
            user_arr = 16*[user_id]
            round_arr = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
            difficulty_arr = []
            perfect_prob_arr = []
            for round_num in [0, 1]:
                
                #Accuracy
                answers = tabs['trial'].loc[(tabs['trial']['round_num'] == round_num) &
                                                (tabs['trial']['user_id'] == user_id), ['correct_bin', 'chosen_bin']]
                if (round_num == 0 and difficulty0 == 'EASY') or (round_num == 1 and difficulty1 == 'EASY'):
                    answers['correct_bin'] = easy_answers
                    difficulty_arr.extend(8*[0])
                    perfect_prob_arr.extend(PERFECT['EASY'])
                else:
                    answers['correct_bin'] = difficult_answers
                    difficulty_arr.extend(8*[1])
                    perfect_prob_arr.extend(PERFECT['DIFFICULT'])
                
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

            df = pd.DataFrame({'start_time': start_time, 'end_time': end_time, 'answer': answers_arr, 'confidence': confidence_arr, 
                                'trial': trial_arr, 'user': user_arr, 'round': round_arr, 'difficulty': difficulty_arr,
                                'perfect_prob': perfect_prob_arr})
            data.append(df)
    
    data = pd.concat(data)
    return(data)

def compile_data(contextual, facial):
    data = contextual
    data['facial'] = ""
    for row_ind in range(data.shape[0]):
        feature = facial.loc[(facial['user'] == str(data.iloc[row_ind]['user'])) &
                    (facial['timestamp'] >= data.iloc[row_ind]['start_time']) & 
                    (facial['timestamp'] <= data.iloc[row_ind]['end_time'])]
        data['facial'].iloc[row_ind] = feature
    return data

def calculate_features(data):
    features_arr = []
    labels_arr = []
    for row_ind in range(data.shape[0]):
        #Contextual Features
        #User ID
        user = data.iloc[row_ind]['user']

        #Round
        round = data.iloc[row_ind]['round']

        #Trial within Round
        trial_raw = data.iloc[row_ind]['trial']
        trial = trial_raw/8.0

        #Difficulty
        difficulty = data.iloc[row_ind]['difficulty']

        #Perfect Learner Probablity
        perfect_prob = data.iloc[row_ind]['perfect_prob']

        #Average Accuracy within Round
        
        all_previous_answer = data.loc[(data['user'] == user) &
                                        (data['round'] == round) &
                                        (data['trial'] < trial_raw), ['trial', 'answer']]
        performance = all_previous_answer['answer'].mean()
        if pd.isna(performance):
            performance = 0.5
        
        #Accuracy for previous Trial
        previous_answer = data.loc[(data['user'] == user) &
                                        (data['round'] == round) &
                                        (data['trial'] == trial_raw-1), ['trial', 'answer']]
        prev_performance = previous_answer['answer'].mean()
        if pd.isna(prev_performance):
            prev_performance = 0.5
        
        #Labels
        correct = data.iloc[row_ind]['answer']

        confidence = data.iloc[row_ind]['confidence']

        #Facial features
        filename = data.iloc[row_ind]['facial']['input']
        timestamp = filename.apply(lambda x: x[-23:-4])
        timestamp.apply(lambda x: datetime.strptime(x, "%Y-%m-%d_%H-%M-%S"))
        features = data.iloc[row_ind]['facial'][['AU04', 'AU07', 'AU12', 'AU25', 'AU26', 
                                                'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']]
        # features['timestamp'] = timestamp
        features['user'] = user
        features['trial'] = trial
        features['round'] = round
        features['difficulty'] = difficulty
        features['performance'] = performance
        features['prev_performance'] = prev_performance
        features['perfect_prob'] = perfect_prob
        
        labels = pd.DataFrame.from_dict({'correct': data.iloc[row_ind]['facial'].shape[0]*[correct], 'confidence': data.iloc[row_ind]['facial'].shape[0]*[confidence]})
        
        if features.shape[0] > 0:
            features_arr.append(features)
            labels_arr.append(labels)

    return features_arr, labels_arr

if __name__ == '__main__':
    tabs = read_tables()
    contextual = read_contextual(tabs)
    
    valid_users = set(contextual.user.to_list())
    # facial = read_facial(valid_users)
    # filehandler = open('data/study2_facial.pkl',"wb")
    # pkl.dump(facial,filehandler)
    # filehandler.close()
    file = open("data/study2_facial.pkl",'rb')
    facial = pkl.load(file)
    file.close()

    data = compile_data(contextual, facial)
    features, labels = calculate_features(data)
    filehandler = open('data/study2_data.pkl',"wb")
    pkl.dump({'features': features, 'labels': labels},filehandler)
    filehandler.close()


    