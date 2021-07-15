import sqlite3
import pandas as pd
from data.params import *
from datetime import datetime
import numpy as np
from data.utils import *
from os import path

def read_tables():

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

    return {'condition': condition_tab, 'user': user_tab, 'trial': trials_tab, 'demo': demos_tab}

def card_order_values(card_order, true_hyp, all_hyp, cards):
    card_num = len(card_order)
    num_bins = np.array([0, 0])
    num_hypotheses_arr = np.zeros((card_num))
    equiv_cards_arr = np.zeros((card_num))
    prob_arr = np.zeros((card_num))
    hyp_valid = np.ones((len(all_hyp))).astype('int')

    for ind in range(card_num):
        
        #Number of hypotheses remaining
        num_hypotheses_arr[ind] = np.sum(hyp_valid)

        #Find number of hypotheses eliminated by each card
        num_hyp_removed = np.zeros((81))
        hyp_removed_ind = []
        for test_card in range(81):
            if test_card in card_order[:ind]:
                num_hyp_removed[test_card] = -10
                hyp_removed_ind.append([])
            else:
                tmp1, tmp2 = calc_hyp_removed(all_hyp, hyp_valid, true_hyp, cards[test_card,:,:])
                num_hyp_removed[test_card] = tmp1
                hyp_removed_ind.append(tmp2)

        #Cards eliminating the maximum number of hypotheses
        max_val = np.max(num_hyp_removed)
        max_inds = np.where(num_hyp_removed == max_val)[0]
        equiv_cards_arr[ind] = len(max_inds)

        #Chosen card
        current_card_ind = card_order[ind]
        current_card = cards[current_card_ind, :, :]
        current_bin = sort_card(true_hyp, current_card)
        
        #Find probability that we would have sorted this card correctly
        num_corr = 0
        for hyp_ind in np.arange(len(hyp_valid)):
            if hyp_valid[hyp_ind]:
                if sort_card(all_hyp[hyp_ind], current_card) == current_bin:
                    num_corr += 1
        prob_arr[ind] = num_corr/float(np.sum(hyp_valid))

        #Add card and remove hypotheses
        hyp_valid[hyp_removed_ind[current_card_ind]] = 0
        num_bins[current_bin] += 1

    return num_hypotheses_arr, equiv_cards_arr, prob_arr   

def compute_features(tabs, rule_params):
    features = np.zeros((tabs['trial'].shape[0], 8))

    for index in range(tabs['trial'].shape[0]):
        #Get User id
        user_id = tabs['trial'].at[index, 'user_id'].item()
        
        #Get condition id
        condition_id = tabs['user'].loc[tabs['user']['id'] == user_id, 'condition_id'].item()

        #Get round
        round_num = tabs['trial'].at[index, 'round_num'].item()

        #Number of card presented
        trial_num = tabs['trial'].at[index, 'trial_num'].item()

        #Get difficulty
        if round_num == 0:
            difficulty = tabs['condition'].loc[tabs['condition']['id'] == condition_id, 'difficulty0'].item()
        else:
            difficulty = tabs['condition'].loc[tabs['condition']['id'] == condition_id, 'difficulty1'].item()

        #Time of first demo
        first_demo_time = tabs['demo'].loc[(tabs['demo']['user_id'] == user_id) & 
                                            (tabs['demo']['demo_num'] == 1) & 
                                            (tabs['demo']['round_num'] == round_num), 'timestamp'].item()
        first_demo_time = datetime.strptime(first_demo_time, '%Y-%m-%d %H:%M:%S.%f')

        #Time of current trial start
        trial_time = tabs['trial'].at[index, 'timestamp']
        trial_time = datetime.strptime(trial_time, '%Y-%m-%d %H:%M:%S.%f')

        #Get all previous accuracies
        prev_acc_arr = np.zeros((trial_num-1))
        for prev_trial_num in range(1, trial_num):
            prev_correct_bin = tabs['trial'].loc[(tabs['trial']['user_id'] == user_id) & 
                                            (tabs['trial']['trial_num'] == trial_num - 1) & 
                                            (tabs['trial']['round_num'] == round_num), 'correct_bin'].item()
            prev_chosen_bin = tabs['trial'].loc[(tabs['trial']['user_id'] == user_id) & 
                                            (tabs['trial']['trial_num'] == trial_num - 1) & 
                                            (tabs['trial']['round_num'] == round_num), 'chosen_bin'].item()
            if prev_correct_bin == prev_chosen_bin:
                prev_acc_arr[prev_trial_num-1] = 1

        #FEATURES

        #Rule Difficulty (Easy = 0, Difficult = 1)
        if difficulty == 'EASY':
            task_difficulty = 0
        else:
            task_difficulty = 1
        
        #Number of card presented
        #trial_num
        
        #Time Elapsed in seconds
        time_elapsed = (trial_time - first_demo_time).total_seconds()

        #Hyp Number before card
        hyp_before_card = rule_params[difficulty]['num_hypotheses'][trial_num-1]

        #Probability of sorting correctly
        prob_correct_sort = rule_params[difficulty]['prob'][trial_num-1]

        #Accuracy of previous card
        if len(prev_acc_arr) == 0:
            prev_acc = 0.
        else:
            prev_acc = prev_acc_arr[-1]
        
        #Accuracy of previous 4 cards
        if len(prev_acc_arr) == 0:
            prev_4_acc = 0.
        elif len(prev_acc_arr) < 4:
            prev_4_acc = np.mean(prev_acc_arr)
        else:
            prev_4_acc = np.mean(prev_acc_arr[-4:])
        
        #Accuracy of all cards
        if len(prev_acc_arr) == 0:
            prev_all_acc = 0.
        else:
            prev_all_acc = np.mean(prev_acc_arr)
        
        features[index, :] = [task_difficulty, trial_num, time_elapsed, hyp_before_card, prob_correct_sort, prev_acc, prev_4_acc, prev_all_acc]

    return features

if __name__ == '__main__':
    tabs = read_tables()
    
    blank_hyp = np.zeros((2, 10, 4, 3))
    all_hyp = create_all_hypotheses()
    all_cards = create_all_cards()

    easy_demos = RULE_PROPS['EASY']['demo_cards']
    easy_cards = RULE_PROPS['EASY']['cards']
    easy_cards.extend(easy_demos)
    easy_rule = np.copy(blank_hyp)
    easy_rule[0, 0, 2, 0] = 1
    easy_rule[1, 0, 2, 1] = 1
    easy_rule[1, 1, 2, 2] = 1

    if not path.exists('data/easy.npz'):
        easy_num_hypotheses, easy_equiv_cards, easy_prob = card_order_values(easy_cards, easy_rule, all_hyp, all_cards)
        np.savez('data/easy.npz', num_hypotheses=easy_num_hypotheses[len(easy_demos):], 
                                    equiv_cards=easy_equiv_cards[len(easy_demos):], 
                                    prob=easy_prob[len(easy_demos):])
    easy_params = np.load('data/easy.npz')

    difficult_demos = RULE_PROPS['DIFFICULT']['demo_cards']
    difficult_cards = RULE_PROPS['DIFFICULT']['cards']
    difficult_cards.extend(difficult_demos)
    difficult_rule = np.copy(blank_hyp)
    difficult_rule[0, 0, 2, 0] = 1
    difficult_rule[1, 0, 2, 1] = 1
    difficult_rule[1, 1, 2, 2] = 1

    if not path.exists('data/difficult.npz'):
        difficult_num_hypotheses, difficult_equiv_cards, difficult_prob = card_order_values(difficult_cards, difficult_rule, all_hyp, all_cards)
        np.savez('data/difficult.npz', num_hypotheses=difficult_num_hypotheses[len(difficult_demos):], 
                                    equiv_cards=difficult_equiv_cards[len(difficult_demos):], 
                                    prob=difficult_prob[len(difficult_demos):])
    difficult_params = np.load('data/difficult.npz')

    features = compute_features(tabs, {'EASY': easy_params, 'DIFFICULT': difficult_params})
    
