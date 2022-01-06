from typing import Sequence
from submission import *
from hmm_submission_test import *
import copy

class Path:
    def __init__(self):
        self.seq = []
        self. prob = -1 

    def apped_to_seq(self, state, prob):
        self.seq.append(state)
        if self.prob == -1:
            self.prob = prob
        else:
            self.prob = self.prob * prob
    
    def update_path(self, obj, transition_probs, y, emission_paras):
        curr = self.seq[-1]
        prev = obj.seq[-1]
        value_curr_curr = self.prob * transition_probs[curr][curr]
        value_prev_curr = obj.prob * transition_probs[prev][curr]
        emission_prob = gaussian_prob(y, emission_paras[curr])

        if value_prev_curr > value_curr_curr:
            # scrap the old path
            self.seq = obj.seq.copy()
            
            self.apped_to_seq(curr, transition_probs[prev][curr] * emission_prob)
        else:
            # keep the old path
            t_p = transition_probs[curr][curr]
            e_p = gaussian_prob(y, emission_paras[curr])
            if curr == 'B2':
                print("{}: t_p = {}, e_p = {}".format(curr, t_p, e_p))

            self.apped_to_seq(curr, transition_probs[curr][curr] * emission_prob)


def get_best_path(P1, P2, P3):
    best_obj = None
    if P1.prob > P2.prob:
        if P1.prob > P3.prob:
            best_obj = P1
        else:
            best_obj = P3
    else:
        if P2.prob > P3.prob:
            best_obj = P2
        else:
            best_obj = P3
    
    return best_obj 

def make_path(evidence_vector, states, prior_probs,
            transition_probs, emission_paras):

    path_1 = Path()
    path_2 = Path()
    path_3 = Path()

    state_1 = states[0] 
    state_2 = states[1]
    state_3 = states[2]

    for i in range(0, len(evidence_vector)):
        y = evidence_vector[i]
        if i == 0:
            prior =  prior_probs[state_1]
            gaus_prob = gaussian_prob(y, emission_paras[state_1])
            exp_prob = prior * gaus_prob
            path_1.apped_to_seq(state_1,  exp_prob)
        
        if i == 1:
            # update state_2's list
            path_2.seq = path_1.seq.copy()
            path_2.prob = path_1.prob
            # calculate the transition from state 1 to state 2
            t_p = transition_probs[state_1][state_2]
            e_p = gaussian_prob(y, emission_paras[state_2])
            print("{}: t_p = {}, e_p = {}".format(state_2, t_p, e_p))
            exp_prob = transition_probs[state_1][state_2] * gaussian_prob(y, emission_paras[state_2])

            #update path 2
            path_2.apped_to_seq(state_2, exp_prob)

            # update path 1
            exp_prob = transition_probs[state_1][state_1] * gaussian_prob(y, emission_paras[state_1])
            path_1.apped_to_seq(state_1, exp_prob)

        if i == 2:
            # update path 3
            path_3.seq = path_2.seq.copy()
            path_3.prob = path_2.prob
            exp_prob = transition_probs[state_2][state_3] * gaussian_prob(y, emission_paras[state_3])
            path_3.apped_to_seq(state_3, exp_prob)

            # update path 2
            path_2.update_path(path_1, transition_probs, y, emission_paras) 

            # update path 1
            path_1.apped_to_seq(state_1, transition_probs[state_1][state_1] * gaussian_prob(y, emission_paras[state_1]))


    for i in range(3, len(evidence_vector)):
        y = evidence_vector[i]
        # update 3 
        path_3.update_path(path_2, transition_probs, y, emission_paras)

        # update 2 
        path_2.update_path(path_1, transition_probs, y, emission_paras)
        # update 1

        # update path in the end
        path_1.apped_to_seq(state_1, transition_probs[state_1][state_1] * gaussian_prob(y, emission_paras[state_1]))

    return get_best_path(path_1, path_2, path_3)


def viterbi(evidence_vector, states, prior_probs,
            transition_probs, emission_paras):
    """Viterbi Algorithm to calculate the most likely states give the evidence.

    Args:
        evidence_vector (list): List of right hand Y-axis positions (interger).

        states (list): List of all states in a word. No transition between words.
                       example: ['B1', 'B2', 'B3', 'Bend']

        prior_probs (dict): prior distribution for each state.
                            example: {'X1': 0.25,
                                      'X2': 0.25,
                                      'X3': 0.25,
                                      'Xend': 0.25}

        transition_probs (dict): dictionary representing transitions from each
                                 state to every other state.

        emission_paras (dict): parameters of Gaussian distribution 
                                from each state.

    Return:
        tuple of
        ( A list of states the most likely explains the evidence,
          probability this state sequence fits the evidence as a float )

    Note:
        You are required to use the function gaussian_prob to compute the
        emission probabilities.

    """
    if evidence_vector == []:
        return (None, 0)    

    sample_states = [states[0], states[1], states[2], states[3]]
    b_max = make_path(evidence_vector, sample_states, prior_probs, transition_probs, emission_paras)

    sample_states = [states[4], states[5], states[6], states[7]]
    c_max = make_path(evidence_vector, sample_states, prior_probs, transition_probs, emission_paras)

    sample_states = [states[8], states[9], states[10], states[11]]
    h_max = make_path(evidence_vector, sample_states, prior_probs, transition_probs, emission_paras)

    model = get_best_path(b_max, c_max, h_max)
    sequence = model.seq
    probability = model.prob
    return sequence, probability

if __name__ == "__main__":
#     TestPart1b().test_viterbi_case1(part_1_a, viterbi)
#     TestPart1b().test_viterbi_case2(part_1_a, viterbi)
#     TestPart1b().test_viterbi_case3(part_1_a, viterbi)
    # TestPart1b().test_viterbi_realsample1(part_1_a, viterbi)
    # TestPart1b().test_viterbi_realsample2(part_1_a, viterbi)
    # TestPart1b().test_viterbi_realsample3(part_1_a, viterbi)
    TestPart1b().test_viterbi_realsample1(part_1_a, viterbi)
    TestPart1b().test_viterbi_realsample2(part_1_a, viterbi)
    TestPart1b().test_viterbi_realsample3(part_1_a, viterbi)






    

