import numpy as np

def all_props():
    return [['red', 'green', 'purple'],['open', 'striped', 'solid'],['diamond', 'oval', 'squiggle'],['one', 'two', 'three']]

def create_all_cards():
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

def create_all_hypotheses():
    blank_hyp = np.zeros((2, 3, 4, 3))

    all_hyp = []

    #One Prop hyp
    for prop1 in range(4):
        for prop1val in range(3):
            for bin0, bin1 in zip([0, 1], [1, 0]):
                #prop1val goes in bin0
                hyp = np.copy(blank_hyp)
                for cur_prop in range(3):
                    if cur_prop == prop1val:
                        hyp[bin0, 0, prop1, cur_prop] = 1
                    else:
                        hyp[bin1, 0, prop1, cur_prop] = 1
                all_hyp.append(hyp)

    #Two Prop hyp
    for prop1 in range(4):
        for prop2 in range(4):
            if not prop1 == prop2:
                #Choose propval from prop1 that we will separate
                for prop1val in range(3):

                    #Get the prop1 complement
                    complement_prop1val = []
                    for cur_prop in range(3):
                        if not cur_prop == prop1val:
                            complement_prop1val.append(cur_prop)

                    #Choose the one for prop2
                    for prop2val in range(3):

                        #Get the prop2 complement
                        complement_prop2val = []
                        for cur_prop in range(3):
                            if not cur_prop == prop2val:
                                complement_prop2val.append(cur_prop)
                        
                        for bin0, bin1 in zip([0, 1], [1, 0]):
                            hyp = np.copy(blank_hyp)
                            #The one and one goes in bin0
                            hyp[bin0, 0, prop1, prop1val] = 1
                            hyp[bin0, 0, prop2, prop2val] = 1

                            #The one and two goes in bin1
                            hyp[bin1, 0, prop1, prop1val] = 1
                            hyp[bin1, 0, prop2, complement_prop2val[0]] = 1
                            hyp[bin1, 0, prop2, complement_prop2val[1]] = 1

                            #Two for prop1 either goes in left or right
                            for row1_bin in [0, 1]:
                                #Make sure to clear the first row
                                hyp = np.copy(hyp)
                                hyp[:, 1, :, :] = 0

                                hyp[row1_bin, 1, prop1, complement_prop1val[0]] = 1
                                hyp[row1_bin, 1, prop1, complement_prop1val[1]] = 1
                                all_hyp.append(hyp)
    
    return all_hyp

def card_str(card, props=all_props()):
    cur_str = ''
    for prop_ind, prop_vals in enumerate(props):
        cur_str += prop_vals[np.where(card[prop_ind,:] == 1)[0][0]]
        cur_str += '-'
    return cur_str

def hyp_str(hyp, props=all_props()):
    cur_str = ''

    for bin in range(hyp.shape[0]):
        cur_str += '---Bin ' + str(bin) + ': '
        for row in range(hyp.shape[1]):
            if np.sum(hyp[bin, row, :, :]) > 0:
                cur_str += 'Row ' + str(row) + ': '
                for prop in range(hyp.shape[2]):
                    vals_true = np.where(hyp[bin, row, prop,:] == 1)[0]
                    if len(vals_true) == 1:
                        cur_str += props[prop][vals_true[0]] + '-'
                    if len(vals_true) == 2:
                        cur_str += '(' + props[prop][vals_true[0]] + ',' + props[prop][vals_true[1]] + ')-'

    return cur_str

def sort_card(hyp, card):
    props_matched = []
    for bin in range(hyp.shape[0]):
        #One row needs to be true
        num_rows_true = 0
        for row in range(hyp.shape[1]):
            num_props_true = 0
            
            #All properties must match
            for prop in range(hyp.shape[2]):
                vals_true = np.where(hyp[bin, row, prop,:] == 1)[0]
                card_true = np.where(card[prop,:] == 1)[0][0]
                if len(vals_true) > 0:
                    if card_true in vals_true:
                        num_props_true += 1
                        if np.sum(hyp[bin, row, :, :]) > 0:
                            props_matched.append(prop)
                else:
                    num_props_true += 1
            if num_props_true == 4:
                num_rows_true += 1
            if np.sum(hyp[bin, row, :, :]) == 0:
                num_rows_true -= 1
        if num_rows_true > 0:
            return bin, props_matched

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

def create_card_order(card_num, true_hyp, staggered=False):
    all_hyp = create_all_hypotheses()
    cards = create_all_cards()
    
    #Initialize variables
    card_order = np.zeros((card_num)).astype('int')
    bin_order = np.zeros((card_num)).astype('int')
    num_bins = np.array([0, 0])
    num_hypotheses_arr = np.zeros((card_num))
    equiv_cards_arr = np.zeros((card_num))
    hyp_valid = np.ones((len(all_hyp))).astype('int')
    for ind in range(card_num):
        
        #Number of hypotheses remaining
        if staggered and np.sum(hyp_valid[:24]) > 0: #Only look at early if staggered and all early have not been eliminated
            num_hypotheses_arr[ind] = np.sum(hyp_valid[:24])
        else:
            num_hypotheses_arr[ind] = np.sum(hyp_valid)


        #Find number of hypotheses eliminated by each card
        num_hyp_removed = np.zeros((81))
        hyp_removed_ind = []
        for test_card in range(81):
            if test_card in card_order:
                num_hyp_removed[test_card] = -10
                hyp_removed_ind.append([])
            else:
                if staggered and np.sum(hyp_valid[:24]) > 0: #Only look at early if staggered and all early have not been eliminated
                    tmp1, tmp2 = calc_hyp_removed(all_hyp[:24], hyp_valid[:24], true_hyp, cards[test_card,:,:])
                else:
                    tmp1, tmp2 = calc_hyp_removed(all_hyp, hyp_valid, true_hyp, cards[test_card,:,:])
                num_hyp_removed[test_card] = tmp1
                hyp_removed_ind.append(tmp2)

        #Cards eliminating the maximum number of hypotheses
        max_val = np.max(num_hyp_removed)
        max_inds = np.where(num_hyp_removed == max_val)[0]

        max_val = np.max(num_hyp_removed)
        max_inds = np.where(num_hyp_removed == max_val)[0]
        if max_val == 0:
            equiv_cards_arr[ind] = 81 - ind
        else:
            equiv_cards_arr[ind] = len(max_inds)

        #If choice of more than one optimal card
        if len(max_inds) > 1:
            if np.sum(num_bins) == 0:
                probs = 0.5*np.ones((2))
            else:
                probs = num_bins/np.sum(num_bins)
            probs = 1 - probs
            p = []
            for equiv_ind in max_inds:
                bin = sort_card(true_hyp, cards[equiv_ind, :, :])[0]
                p.append(probs[bin])
            if np.sum(p) > 0:
                p = np.array(p)/np.sum(p)
            else:
                p = np.ones_like(p)/np.sum(np.ones_like(p))
            current_card_ind = np.random.choice(max_inds, p=p)
        else:
            current_card_ind = max_inds[0]

        #Chosen card
        current_card = cards[current_card_ind, :, :]
        current_bin = sort_card(true_hyp, current_card)[0]
        
        # #Find probability that we would have sorted this card correctly
        # num_corr = 0
        # for hyp_ind in np.arange(len(hyp_valid)):
        #     if hyp_valid[hyp_ind]:
        #         if sort_card(all_hyp[hyp_ind], current_card) == current_bin:
        #             num_corr += 1
        # prob_arr[ind] = num_corr/float(np.sum(hyp_valid))

        #Add card and remove hypotheses
        card_order[ind] = current_card_ind
        bin_order[ind] = current_bin
        hyp_valid[hyp_removed_ind[current_card_ind]] = 0

        num_bins[current_bin] += 1
        # print('Card', ind, '-', card_order[ind], card_str(cards[card_order[ind],:,:], all_props()), 'Bin', current_bin)

    return card_order, num_hypotheses_arr, equiv_cards_arr

def learner_model(card_order, true_hyp):
    
    all_hyp = create_all_hypotheses()
    cards = create_all_cards()
    hyp_valid = np.ones((len(all_hyp))).astype('int')
    num_cards = 10
    num_demos = 2
    acc_arr = np.ones((8))
    hyp_diff = np.zeros((2, 10))
    #weighting - color, shading, shape, number

    for card_num in range(num_demos):
        for hyp_ind in np.arange(len(hyp_valid)):
            if np.sum(hyp_valid[:24]) > 0 and hyp_ind >= 24:
                pass
            else:
                if hyp_valid[hyp_ind]:
                    #Difficulty
                    if np.sum(all_hyp[hyp_ind][:, 1, :, :]) == 0:
                        hyp_diff[0, card_num] += 1
                    else:
                        hyp_diff[1, card_num] += 1
        tmp1, hyp_removed_ind = calc_hyp_removed(all_hyp, hyp_valid, true_hyp, cards[card_order[card_num],:,:])
        hyp_valid[hyp_removed_ind] = 0
        
    
    for trial_num, card_num in zip(range(num_cards-num_demos), range(num_demos, num_cards)):
        current_bin = sort_card(true_hyp, cards[card_order[card_num]])[0]

        corr_arr = []
        p_arr = []
        for hyp_ind in np.arange(len(hyp_valid)):
            if np.sum(hyp_valid[:24]) > 0 and hyp_ind >= 24: #Skip hypotheses that are difficult if staggered and all easy have not been eliminated
                pass
            else:
                if hyp_valid[hyp_ind]:
                    sorted_bin, props_matched = sort_card(all_hyp[hyp_ind], cards[card_order[card_num]])
                    if sorted_bin == current_bin:
                        corr_arr.append(1)
                    else:
                        corr_arr.append(0)

                    p_arr.append(1)

                    #Difficulty
                    if np.sum(all_hyp[hyp_ind][:, 1, :, :]) == 0:
                        hyp_diff[0, card_num] += 1
                    else:
                        hyp_diff[1, card_num] += 1
        p_arr = np.array(p_arr)/np.sum(p_arr)
        
        acc_arr[trial_num] = np.random.choice(corr_arr, p=p_arr)

        tmp1, hyp_removed_ind = calc_hyp_removed(all_hyp, hyp_valid, true_hyp, cards[card_order[card_num],:,:])

        hyp_valid[hyp_removed_ind] = 0
    
    return acc_arr, hyp_diff[:, num_demos:]
