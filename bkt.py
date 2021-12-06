from os import P_DETACH
from pyBKT.models import Model
import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = open('data/study1_data.pkl','rb')
df = pkl.load(file)
file.close()


data = pd.DataFrame(columns=('row', 'user_id', 'correct', 'opportunity', 'skill_name'))
ind = 0
for index, row in df.iterrows():
    for trial in range(8):
        tmp = {'row': ind, 'user_id': row['user_id'], 'correct': row['answers'][trial], 'opportunity': trial+1, 'skill_name': row['difficulty']}
        data = data.append(tmp, ignore_index=True)
        ind += 1

if __name__ == '__main__':
    # model = Model(seed = 42, num_fits = 1)
    # model.fit(data = data)
    # with open('data/bkt_model.pkl', 'wb') as output:
    #     pkl.dump({'model': model}, output)

    with open('data/bkt_model.pkl', 'rb') as f:
        tmpdata = pkl.load(f)
    model = tmpdata['model']
    print(model.params())
    
    preds_df = model.predict(data = data)
    
    difficulty = 'DIFFICULT'

    answers = [[], [], [], [], [], [], [], []]
    for index, row, in preds_df.iterrows():
        if row['skill_name'] == difficulty:
            res = np.random.choice([0, 1], p=[1 - row['correct_predictions'], row['correct_predictions']])
            answers[row['opportunity']-1].append(res)
    
    fig, ax = plt.subplots()
    answers = np.array(answers).T
    model_avg = np.mean(answers, axis=0)
    model_std = np.std(answers, axis=0)
    ax.errorbar(np.arange(8), model_avg, yerr=model_std, label='Model Learner')

    human_acc = np.vstack(df.loc[(df['difficulty'] == difficulty), 'answers'].to_numpy())
    human_avg = np.mean(human_acc,axis=0)
    human_std = np.std(human_acc,axis=0)
    ax.errorbar(np.arange(8), human_avg, yerr= human_std,label='Human Learner', color='deeppink')

    ax.legend()
    plt.show()