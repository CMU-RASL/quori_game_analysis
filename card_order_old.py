import numpy as np
from utils import create_card_order, create_all_cards, learner_model, sort_card
from os import path
import matplotlib.pyplot as plt
import pickle as pkl
import matplotlib.cm as cm


num_iter = 160
card_num = 10
blank_hyp = np.zeros((2, 2, 4, 3))
perfect_acc = np.zeros((2, num_iter, 8))
for difficulty_num, difficulty in enumerate(['EASY', 'DIFFICULT']):

    if difficulty == 'EASY':
        #Easy - diamonds on left, all others on right
        true_hyp = np.copy(blank_hyp)
        true_hyp[0, 0, 2, 0] = 1
        true_hyp[1, 0, 2, 1] = 1
        true_hyp[1, 1, 2, 2] = 1
        card_order = [54, 60, 48, 47, 10, 58, 14, 18, 57, 32]
    else:
        card_order = [61, 34, 33, 32, 42, 17, 68, 29, 26, 45]
        #Difficulty - green-one, red/purple on left, green two/three on right
        true_hyp = np.copy(blank_hyp)
        true_hyp[0, 0, 0, 1] = 1 #green
        true_hyp[0, 0, 3, 0] = 1 #one
        true_hyp[0, 1, 0, 0] = 1 #red
        true_hyp[0, 1, 0, 2] = 1 #purple

        true_hyp[1, 0, 0, 1] = 1 #green
        true_hyp[1, 0, 3, 1] = 1 #two
        true_hyp[1, 0, 3, 2] = 1 #three

        true_hyp[0, 0, 0, 1] = 1 #green
        true_hyp[0, 0, 3, 0] = 1 #one
        true_hyp[0, 1, 0, 0] = 1 #red
        true_hyp[0, 2, 0, 2] = 1 #purple

        true_hyp[1, 0, 0, 1] = 1 #green
        true_hyp[1, 0, 3, 1] = 1 #two
        true_hyp[1, 1, 0, 1] = 1 #green
        true_hyp[1, 1, 3, 2] = 1 #three


    for ind in range(num_iter):
        acc, hyp_diff = learner_model(card_order, true_hyp)
        perfect_acc[difficulty_num, ind,:] = acc
        print(ind)

perfect_avg = np.mean(perfect_acc, axis=1)
perfect_std = np.std(perfect_acc, axis=1)

print(perfect_avg)
plt.rcParams['font.size'] = 16
fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)

ax[0].errorbar(np.arange(8), perfect_avg[0, :], yerr=perfect_std[0, :], label='Learner Model')
ax[1].errorbar(np.arange(8), perfect_avg[1, :], yerr=perfect_std[1,:], label='Learner Model')


ax[0].set_ylabel('Accuracy')
ax[1].set_ylabel('Accuracy')

file = open('data/study1a_data.pkl','rb')
data = pkl.load(file)
file.close()

human_acc = np.vstack(data.loc[(data['difficulty'] == 'EASY'), 'answers'].to_numpy())
print(human_acc.shape)
human_avg = np.mean(human_acc,axis=0)
human_std = np.std(human_acc,axis=0)
ax[0].errorbar(np.arange(8), human_avg, yerr= human_std,label='Human Learner', color='deeppink')

human_acc = np.vstack(data.loc[(data['difficulty'] == 'DIFFICULT'), 'answers'].to_numpy())
print(human_acc.shape)
human_avg = np.mean(human_acc,axis=0)
human_std = np.std(human_acc,axis=0)
ax[1].errorbar(np.arange(8), human_avg, yerr= human_std,label='Human Learner', color='deeppink')

ax[0].legend()
ax[1].legend()
plt.show()

