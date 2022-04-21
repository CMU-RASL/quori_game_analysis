import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

def create_blank_hyp():
    #Indexed by: bin, row, property, value
    #props = [['red', 'green', 'purple'],['open', 'striped', 'solid'],['diamond', 'oval', 'squiggle'],['one', 'two', 'three']]

    blank_hyp = np.zeros((2, 6, 4, 3))
    return blank_hyp

def create_all_cards():
    #Indexed by card_num, property, value
    #Matches how the card images are labeled 
    cards = np.zeros((81, 4, 3))
    card_ind = 0
    for color in range(3):
        for shading in range(3):
            for shape in range(3):
                for number in range(3):
                    cards[card_ind, 0, color] = 1
                    cards[card_ind, 1, shading] = 1
                    cards[card_ind, 2, shape] = 1
                    cards[card_ind, 3, number] = 1
                    card_ind += 1
    return cards

def create_specific_hypothesis(difficulty):
    if difficulty == 'EASY':
        #Easy - diamonds on left, oval or squiggle on right
        true_hyp = create_blank_hyp()
        true_hyp[0, 0, 2, 0] = 1
        true_hyp[1, 0, 2, 1] = 1
        true_hyp[1, 1, 2, 2] = 1
    else:
        #Difficulty - green-one, red, or purple on left, green-two or green-three on right
        true_hyp = create_blank_hyp()
        true_hyp[0, 0, 0, 1] = 1 #green
        true_hyp[0, 0, 3, 0] = 1 #one
        true_hyp[0, 1, 0, 0] = 1 #red
        true_hyp[0, 2, 0, 2] = 1 #purple

        true_hyp[1, 0, 0, 1] = 1 #green
        true_hyp[1, 0, 3, 1] = 1 #two
        true_hyp[1, 1, 0, 1] = 1 #green
        true_hyp[1, 1, 3, 2] = 1 #three
    return true_hyp

def create_all_hypotheses():
    all_hyp = []
    diff_arr = []
    # #Easy hypotheses
    # for prop1 in range(4): #pick property the rule is defined over
    #     for prop1val in range(3): #pick the value of that property that belongs by itself

    #         #Get the properties that belong together
    #         other_prop1vals = [0, 1, 2]
    #         other_prop1vals.remove(prop1val) 
            
    #         for bin0, bin1 in zip([0, 1], [1, 0]):
    #             #One in bin0 and two in bin1
    #             hyp = create_blank_hyp()
    #             hyp[bin0, 0, prop1, prop1val] = 1
    #             hyp[bin1, 0, prop1, other_prop1vals[0]] = 1
    #             hyp[bin1, 1, prop1, other_prop1vals[1]] = 1
    #             all_hyp.append(hyp)
    
    #Difficult Hypotheses
    for prop1 in range(4):
        for prop2 in range(4):
            if not (prop1 == prop2): #Ensures we get two different properties

                for prop1val1 in range(3):
                    other_prop1vals = [0, 1, 2]
                    other_prop1vals.remove(prop1val1)
                    prop1val2 = other_prop1vals[0]
                    prop1val3 = other_prop1vals[1]

                    for prop2val1 in range(3):
                        other_prop2vals = [0, 1, 2]
                        other_prop2vals.remove(prop2val1)
                        prop2val2 = other_prop2vals[0]
                        prop2val3 = other_prop2vals[1]
                        
                        for bins in zip([0, 1], [1, 0]):
                            #1-1/2-1, 1-1/2-2, 1-1/2-3, 1-2, 1-3
                            #Those are the pairings we have, and each can belong in either bin0 or bin1
                            #This includes easy rules, so we don't need to add those separately
                            for comb1 in range(2):
                                for comb2 in range(2):
                                    for comb3 in range(2):
                                        for comb4 in range(2):
                                            for comb5 in range(2):
                                                hyp = create_blank_hyp()
                                                current_row = [0, 0]

                                                #1-1/2-1
                                                hyp[bins[comb1], current_row[bins[comb1]], prop1, prop1val1] = 1
                                                hyp[bins[comb1], current_row[bins[comb1]], prop2, prop2val1] = 1
                                                current_row[bins[comb1]]+=1

                                                #1-1/2-2
                                                hyp[bins[comb2], current_row[bins[comb2]], prop1, prop1val1] = 1
                                                hyp[bins[comb2], current_row[bins[comb2]], prop2, prop2val2] = 1
                                                current_row[bins[comb2]]+=1

                                                #1-1/2-3
                                                hyp[bins[comb3], current_row[bins[comb3]], prop1, prop1val1] = 1
                                                hyp[bins[comb3], current_row[bins[comb3]], prop2, prop2val3] = 1
                                                current_row[bins[comb3]]+=1

                                                #Is an easy rule if 1-1/2-1, 1-1/2-2, 1-1/2-3 all belong in same bin
                                                if (bins[comb1] == bins[comb2]) and (bins[comb2]== bins[comb3]):
                                                    diff_arr.append(0)
                                                else:
                                                    diff_arr.append(1)

                                                #1-2
                                                hyp[bins[comb4], current_row[bins[comb4]], prop1, prop1val2] = 1
                                                current_row[bins[comb4]]+=1

                                                #1-3
                                                hyp[bins[comb5], current_row[bins[comb5]], prop1, prop1val3] = 1
                                                current_row[bins[comb5]]+=1

                                                all_hyp.append(hyp)

    return all_hyp, diff_arr

def print_hyp(hyp):
    props = [['red', 'green', 'purple'],['open', 'striped', 'solid'],['diamond', 'oval', 'squiggle'],['one', 'two', 'three']]
    prop_names = ['color', 'shading', 'shape', 'number']
    for bin in range(2):
        print('Bin:', bin)
        for row in range(hyp.shape[1]):
            prop_str = ''
            for prop in range(4):
                for val in range(3):
                    if hyp[bin, row, prop, val] == 1:
                        prop_str += ' {}-{} AND'.format(prop_names[prop], props[prop][val])
            if len(prop_str) > 0:
                print(prop_str[:-4])

def print_card(card):
    props = [['red', 'green', 'purple'],['open', 'striped', 'solid'],['diamond', 'oval', 'squiggle'],['one', 'two', 'three']]
    prop_str = ''
    for prop in range(4):
        for val in range(3):
            if card[prop, val] == 1:
                prop_str += '{}-'.format(props[prop][val])
    print(prop_str[:-1])

def sort_card(hyp, card):
    for bin in range(2):
        for row in range(hyp.shape[1]):
            #Everything in the row must be true
            if np.sum(hyp[bin, row, :, :]) > 0:
                match = True
                for prop in range(4):
                    for val in range(3):
                        if (hyp[bin, row, prop, val] == 1) and (card[prop, val] == 0):
                            match = False
                        
                if match:
                    return bin

def calc_hyp_removed(all_hyp, hyp_valid, true_hyp, test_card):
    true_bin, props_matched = sort_card(true_hyp, test_card)
    props = [['red', 'green', 'purple'],['open', 'striped', 'solid'],['diamond', 'oval', 'squiggle'],['one', 'two', 'three']]
    
    hyp_removed = 0
    hyp_removed_ind = []
    for hyp_ind, hyp in enumerate(all_hyp):
        if hyp_valid[hyp_ind]:
            #Check if will sort this card correctly
            if not sort_card(hyp, test_card)[0] == true_bin:
                hyp_removed += 1 #/(hyp.shape[1])
                hyp_removed_ind.append(hyp_ind)    

    return hyp_removed, hyp_removed_ind

def is_hyp_removed(true_hyp, test_card, test_hyp):
    if sort_card(true_hyp, test_card) == sort_card(test_hyp, test_card):
        return False
    else:
        return True

def learner_model(true_hyp, card_order):
    cards = create_all_cards()
    hypotheses, diff_arr = create_all_hypotheses()
    diff_arr = np.array(diff_arr)
    hyp_valid = np.ones((len(hypotheses))).astype('int')
    num_cards = len(card_order)
    num_demos = 2
    acc_arr = np.zeros((num_cards - num_demos))
    hyp_remain_arr = np.zeros((num_cards - num_demos))

    #Demonstrations
    for card_num in range(num_demos):
        #Remove hypotheses based on seeing where card goes
        for hyp_ind, hyp in enumerate(hypotheses):
            if hyp_valid[hyp_ind] == 1:
                if is_hyp_removed(true_hyp, cards[card_order[card_num]], hyp):
                    hyp_valid[hyp_ind] = 0
    
    #Trials
    for card_num in range(num_demos, num_cards):
        #Sorting Stage

        #Get all hypotheses that are easy and have not been eliminated
        cur_hyp_ind = np.where((hyp_valid == 1) & (diff_arr == 0))[0]
        
        #If there are no easy hyp remaining, add in difficult
        if len(cur_hyp_ind) == 0:
            cur_hyp_ind = np.where((hyp_valid == 1))[0]

        hyp_remain_arr[card_num - num_demos] = len(cur_hyp_ind)
        #Pick a random hypothesis out of the set
        chosen_hyp = np.random.choice(cur_hyp_ind, 1)[0]

        #Sort based on that hyp
        chosen_bin = sort_card(hypotheses[chosen_hyp], cards[card_order[card_num]])

        #Record whether correct or not
        acc_arr[card_num - num_demos] = (chosen_bin == sort_card(true_hyp, cards[card_order[card_num]]))

        #Elimination Stage
        for hyp_ind, hyp in enumerate(hypotheses):
            if hyp_valid[hyp_ind] == 1:
                if is_hyp_removed(true_hyp, cards[card_order[card_num]], hyp):
                    hyp_valid[hyp_ind] = 0
    
    return acc_arr

def create_card_order(true_hyp, num_cards):
    cards = create_all_cards()
    hypotheses, diff_arr = create_all_hypotheses()
    diff_arr = np.array(diff_arr)
    hyp_valid = np.ones((len(hypotheses))).astype('int')
    card_order = np.zeros((num_cards)).astype('int')
    bin_order = np.zeros((num_cards)).astype('int')
    num_bins = np.array([0, 0])

    for card_ind in range(num_cards):
        #Get all hypotheses that are easy and have not been eliminated
        cur_hyp_ind = np.where((hyp_valid == 1) & (diff_arr == 0))[0]
        
        #If there are no easy hyp remaining, add in difficult
        if len(cur_hyp_ind) == 0:
            cur_hyp_ind = np.where((hyp_valid == 1))[0]
        
        #Go through all possible cards and see how many hypotheses they remove
        num_removed = np.zeros((81))
        for test_card_ind, test_card in enumerate(cards):
            if test_card_ind in card_order:
                num_removed[test_card_ind] = -1
            else:
                #Get number of hyp that would be removed by that card
                for hyp_ind in cur_hyp_ind: #Only consider hypotheses that haven't been removed yet
                    if hyp_valid[hyp_ind] == 1:
                        if is_hyp_removed(true_hyp, test_card, hypotheses[hyp_ind]):
                            num_removed[test_card_ind] += 1
        
        #Cards eliminating the maximum number of hypotheses
        max_val = np.max(num_removed)
        max_inds = np.where(num_removed == max_val)[0]
        
        #If choice of more than one optimal card, pick one that is sorted into the bin that has less cards in it
        if len(max_inds) > 1:
            if np.sum(num_bins) == 0:
                probs = 0.5*np.ones((2))
            else:
                probs = num_bins/np.sum(num_bins)
            probs = 1 - probs
            p = []
            for equiv_ind in max_inds:
                bin = sort_card(true_hyp, cards[equiv_ind, :, :])
                p.append(probs[bin])
            if np.sum(p) > 0:
                p = np.array(p)/np.sum(p)
            else:
                p = np.ones_like(p)/np.sum(np.ones_like(p))
            current_card_ind = np.random.choice(max_inds, p=p)
        else:
            current_card_ind = max_inds[0]

        #Get the chosen card
        current_card = cards[current_card_ind]
        current_bin = sort_card(true_hyp, current_card)

        #Update the arrays and number of cards in each bin
        card_order[card_ind] = current_card_ind
        bin_order[card_ind] = current_bin
        num_bins[current_bin] += 1

        #Elimination Stage
        for hyp_ind, hyp in enumerate(hypotheses):
            if hyp_valid[hyp_ind] == 1:
                if is_hyp_removed(true_hyp, current_card, hyp):
                    hyp_valid[hyp_ind] = 0
    
    return card_order

def plot_human_comparison():

    easy_card_order = [54, 60, 48, 47, 10, 58, 14, 18, 57, 32]
    difficult_card_order = [61, 34, 33, 32, 42, 17, 68, 29, 26, 45]
    card_orders = [easy_card_order, difficult_card_order]
    card_num = 10
    
    num_iter = 80
    
    learner_acc = np.zeros((2, num_iter, 8))
    for difficulty_num, difficulty in enumerate(['EASY', 'DIFFICULT']):
        true_hyp = create_specific_hypothesis(difficulty)
        for ind in range(num_iter):
            acc = learner_model(true_hyp, card_orders[difficulty_num])
            learner_acc[difficulty_num, ind,:] = acc
            print(ind)
    learner_avg = np.mean(learner_acc, axis=1)
    learner_std = np.std(learner_acc, axis=1)

    print(learner_avg)
    plt.rcParams['font.size'] = 16
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)

    ax[0].errorbar(np.arange(8), learner_avg[0, :], yerr=learner_std[0, :], label='Learner Model')
    ax[1].errorbar(np.arange(8), learner_avg[1, :], yerr=learner_std[1,:], label='Learner Model')


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

plot_human_comparison()


