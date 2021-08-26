import csv
import pandas as pd
import sqlite3

prolific_file_name = 'data/study1_2_full_prolific.csv'
db_file_name = 'data/study1_2_full.db'
output_file_name = 'data/study1_2_full_prolific_approval.txt'

#Read CSV
ids = []
statuses = []
codes = []
decisions = []
with open(prolific_file_name, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        ids.append(row['participant_id'])
        statuses.append(row['status'])
        codes.append(row['entered_code'])
        if row['status'] == 'APPROVED':
            decisions.append('ACCEPT')
        else:
            decisions.append('REJECT')
        # if row['entered_code'] == 'i88ukKmKe':
        #     print('here')
prolific_tab = pd.DataFrame({'id': ids, 'status': statuses, 'code': codes, 'decision': decisions})

#Read Database
cnx = sqlite3.connect(db_file_name)
user_tab = pd.read_sql_query("SELECT * from user", cnx)

approved_ids = []

#Iterate through Prolific Database
for index in range(user_tab.shape[0]):
    user_id = user_tab.at[index, 'username']
    code = user_tab.at[index,'code']
    
    #Adjustments - Get this from adjustments.py!

    #Get row in prolific tab
    status = prolific_tab.loc[prolific_tab['id'] == user_id, 'status']
    if status.shape[0] == 1:
        
        #Check if status matches
        status = prolific_tab.loc[prolific_tab['id'] == user_id, 'status'].item()
        if status == 'AWAITING REVIEW':
            
            #Check if code matches
            prolific_code = prolific_tab.loc[prolific_tab['id'] == user_id, 'code'].item()

            if code == prolific_code:
                prolific_tab.loc[prolific_tab['id'] == user_id, 'decision'] = 'ACCEPT'
                approved_ids.append(user_id)
        if status == 'APPROVED':
            approved_ids.append(user_id)

#Write to text file
with open(output_file_name, 'w') as txtfile:
    for user_id in approved_ids:
        txtfile.write(user_id)
        txtfile.write('\n')