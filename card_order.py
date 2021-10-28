import numpy as np
from utils import create_card_order, create_all_cards, perfect_learner, sort_card
from os import path
import matplotlib.pyplot as plt
import pickle as pkl
import matplotlib.cm as cm

def create_hyp(name):
    blank_hyp = np.zeros((2, 2, 4, 3))
    if name == 'EASY':
        #Easy - diamonds on left, all others on right
        true_hyp = np.copy(blank_hyp)
        true_hyp[0, 0, 2, 0] = 1
        true_hyp[1, 0, 2, 1] = 1
        true_hyp[1, 1, 2, 2] = 1

       
    elif name == 'DIFFICULT':
        #Difficulty - green-one, red/purple on left, green two/three on right
        true_hyp = np.copy(blank_hyp)
        true_hyp[0, 0, 0, 1] = 1 #green
        true_hyp[0, 0, 3, 0] = 1 #one
        true_hyp[0, 1, 0, 0] = 1 #red
        true_hyp[0, 1, 0, 2] = 1 #purple

        true_hyp[1, 0, 0, 1] = 1 #green
        true_hyp[1, 0, 3, 1] = 1 #two
        true_hyp[1, 0, 3, 2] = 1 #three

    return true_hyp

difficulty = 'DIFFICULT'
num_iter = 50
staggered=True

true_hyp = create_hyp(difficulty)
card_num = 10

card_order, num_hypotheses_arr, equiv_cards_arr = create_card_order(card_num, true_hyp, staggered=staggered)

fig, ax = plt.subplots(2, 1)

perfect_acc = np.zeros((num_iter, 8))
perfect_hyp_diff = np.zeros((num_iter, 2, 8))
for ind in range(num_iter):
    acc, hyp_diff = perfect_learner(card_order, true_hyp, difficulty_adjustment=False, prop_weights=False, staggered=staggered)
    perfect_acc[ind,:] = acc
    perfect_hyp_diff[ind, :, :] = hyp_diff
perfect_avg = np.mean(perfect_acc, axis=0)
perfect_std = np.std(perfect_acc, axis=0)
perfect_hyp_avg = np.mean(perfect_hyp_diff, axis=0)
perfect_hyp_std = np.std(perfect_hyp_diff, axis=0)
print(perfect_hyp_avg)
ax[0].errorbar(np.arange(8), perfect_avg, yerr=perfect_std, label='Perfect Learner')
ax[1].errorbar(np.arange(8), perfect_hyp_avg[0,:], yerr=perfect_hyp_std[0,:], label='Easy Remaining')
ax[1].errorbar(np.arange(8), perfect_hyp_avg[1,:], yerr=perfect_hyp_std[1,:], label='Difficult Remaining')
ax[0].set_title(difficulty)
ax[0].set_ylabel('Accuracy')
ax[1].set_ylabel('Remaining Hypotheses')


file = open('data/study1_data.pkl','rb')
data = pkl.load(file)
file.close()
human_acc = np.vstack(data.loc[(data['difficulty'] == difficulty), 'answers'].to_numpy())
human_avg = np.mean(human_acc,axis=0)
human_std = np.std(human_acc,axis=0)
ax[0].errorbar(np.arange(8), human_avg, yerr= human_std,label='Human Learner', color='deeppink')

ax[0].legend(loc='lower right')
ax[1].legend(loc='lower right')
plt.show()


###################
# fig, ax = plt.subplots()

# difficulty_adjustment = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
# prop_weights = len(difficulty_adjustment)*[False]
# labels = ['Multiplier for easy - {}'.format(diff) for diff in difficulty_adjustment]

# prop_weights = [[1, 1, 1, 1], [2, 1, 2, 1], [4, 1, 4, 1], [6, 1, 6, 1], [8, 1, 8, 1], [10, 1, 10, 1], [12, 1, 12, 1], [14, 1, 14, 1], [16, 1, 16, 1], [18, 1, 18, 1], [20, 1, 20, 1], [22, 1, 22, 1], [24, 1, 24, 1]]
# difficulty_adjustment = len(prop_weights)*[False]
# labels = ['Multiplier for color/shape - {}'.format(weight) for weight in range(0, 26, 2)]

# color_interval = np.linspace(0, 1, len(labels))
# colors = [cm.RdYlBu(x) for x in color_interval]

# for series in range(len(labels)):
#     perfect_acc = []
#     for ind in range(num_iter):
#         perfect_acc.append(perfect_learner(card_order, true_hyp, difficulty_adjustment=difficulty_adjustment[series], prop_weights=prop_weights[series]))
#     perfect_acc = np.vstack(perfect_acc)
#     perfect_avg = np.mean(perfect_acc, axis=0)
#     perfect_std = np.std(perfect_acc, axis=0)
#     ax.errorbar(np.arange(8), perfect_avg, yerr= perfect_std,label='Perfect Learner - {}'.format(labels[series]), color=colors[series])

# file = open('data/study1_data.pkl','rb')
# data = pkl.load(file)
# file.close()
# human_acc = np.vstack(data.loc[(data['difficulty'] == difficulty), 'answers'].to_numpy())
# human_avg = np.mean(human_acc,axis=0)
# human_std = np.std(human_acc,axis=0)
# ax.errorbar(np.arange(8), human_avg, yerr= human_std,label='Human Learner', color='deeppink')

# ax.legend(loc='lower right')
# ax.set_title(difficulty)
# plt.show()

