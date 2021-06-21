import sqlite3
import pandas as pd
from data.params import *

# Create your connection.
cnx = sqlite3.connect('data/app.db')

condition_tab = pd.read_sql_query("SELECT * FROM condition", cnx)

#Add new columns, for each round
condition_tab['difficulty0'] = ['', '']
condition_tab['difficulty1'] = ['', '']
for index in range(condition_tab.shape[0]):
    condition_tab.at[index, 'difficulty'] = (CONDITIONS[index][0][0], CONDITIONS[index][1][0])
    condition_tab.at[index, 'nonverbal'] = (CONDITIONS[index][0][1], CONDITIONS[index][1][1])
    condition_tab.at[index, 'difficulty0'] = CONDITIONS[index][0][0]
    condition_tab.at[index, 'difficulty1'] = CONDITIONS[index][1][0]

print(condition_tab)

user_tab = pd.read_sql_query("SELECT * from user", cnx)
print(user_tab)

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

print(trials_tab)
