import numpy as np
import copy
import operator


def gaussian_prob(x, para_tuple):
    """Compute the probability of a given x value

    Args:
        x (float): observation value
        para_tuple (tuple): contains two elements, (mean, standard deviation)

    Return:
        Probability of seeing a value "x" in a Gaussian distribution.

    Note:
        We simplify the problem so you don't have to take care of integrals.
        Theoretically speaking, the returned value is not a probability of x,
        since the probability of any single value x from a continuous 
        distribution should be zero, instead of the number outputed here.
        By definition, the Gaussian percentile of a given value "x"
        is computed based on the "area" under the curve, from left-most to x. 
        The proability of getting value "x" is zero bcause a single value "x"
        has zero width, however, the probability of a range of value can be
        computed, for say, from "x - 0.1" to "x + 0.1".

    """
    if para_tuple == (None, None):
        return 0.0

    mean, std = para_tuple
    gaussian_percentile = (2 * np.pi * std**2)**-0.5 * \
                          np.exp(-(x - mean)**2 / (2 * std**2))
    return gaussian_percentile


def part_1_a():
    """Provide probabilities for the word HMMs outlined below.

    Word BUY, CAR, and HOUSE.

    Review Udacity Lesson 8 - Video #29. HMM Training

    Returns:
        tuple() of
        (prior probabilities for all states for word BUY,
         transition probabilities between states for word BUY,
         emission parameters tuple(mean, std) for all states for word BUY,
         prior probabilities for all states for word CAR,
         transition probabilities between states for word CAR,
         emission parameters tuple(mean, std) for all states for word CAR,
         prior probabilities for all states for word HOUSE,
         transition probabilities between states for word HOUSE,
         emission parameters tuple(mean, std) for all states for word HOUSE,)


        Sample Format (not complete):
        (
            {'B1': prob_of_starting_in_B1, 'B2': prob_of_starting_in_B2, ...},
            {'B1': {'B1': prob_of_transition_from_B1_to_B1,
                    'B2': prob_of_transition_from_B1_to_B2,
                    'B3': prob_of_transition_from_B1_to_B3,
                    'Bend': prob_of_transition_from_B1_to_Bend},
             'B2': {...}, ...},
            {'B1': tuple(mean_of_B1, standard_deviation_of_B1),
             'B2': tuple(mean_of_B2, standard_deviation_of_B2), ...},
            {'C1': prob_of_starting_in_C1, 'C2': prob_of_starting_in_C2, ...},
            {'C1': {'C1': prob_of_transition_from_C1_to_C1,
                    'C2': prob_of_transition_from_C1_to_C2,
                    'C3': prob_of_transition_from_C1_to_C3,
                    'Cend': prob_of_transition_from_C1_to_Cend},
             'C2': {...}, ...}
            {'C1': tuple(mean_of_C1, standard_deviation_of_C1),
             'C2': tuple(mean_of_C2, standard_deviation_of_C2), ...}
            {'H1': prob_of_starting_in_H1, 'H2': prob_of_starting_in_H2, ...},
            {'H1': {'H1': prob_of_transition_from_H1_to_H1,
                    'H2': prob_of_transition_from_H1_to_H2,
                    'H3': prob_of_transition_from_H1_to_H3,
                    'Hend': prob_of_transition_from_H1_to_Hend},
             'H2': {...}, ...}
            {'H1': tuple(mean_of_H1, standard_deviation_of_H1),
             'H2': tuple(mean_of_H2, standard_deviation_of_H2), ...}
        )
    """

    """Word BUY"""
    b_prior_probs = {
        'B1': 0.333,
        'B2': 0.000,
        'B3': 0.000,
        'Bend': 0.000
    }
    b_transition_probs = {
        'B1': {'B1': 0.625, 'B2': 0.375, 'B3': 0.000, 'Bend': 0.000},
        'B2': {'B1': 0.000, 'B2': 0.625, 'B3': 0.375, 'Bend': 0.000},
        'B3': {'B1': 0.000, 'B2': 0.000, 'B3': 0.625, 'Bend': 0.375},
        'Bend': {'B1': 0.000, 'B2': 0.000, 'B3': 0.000, 'Bend': 1.000}
    }
    # Parameters for end state is not required
    b_emission_paras = {
        'B1': (41.75, 2.773),
        'B2': (58.625, 5.678),
        'B3': (53.125, 5.418),
        'Bend': (None, None)
    }

    """Word CAR"""
    c_prior_probs = {
        'C1': 0.333,
        'C2': 0.000,
        'C3': 0.000,
        'Cend': 0.000
    }
    c_transition_probs = {
        'C1': {'C1': 0.667, 'C2': 0.333, 'C3': 0.000, 'Cend': 0.000},
        'C2': {'C1': 0.000, 'C2': 0.000, 'C3': 1.000, 'Cend': 0.000},
        'C3': {'C1': 0.000, 'C2': 0.000, 'C3': 0.800, 'Cend': 0.200},
        'Cend': {'C1': 0.000, 'C2': 0.000, 'C3': 0.000, 'Cend': 1.000}
    }
    # Parameters for end state is not required
    c_emission_paras = {
        'C1': (35.667, 4.899),
        'C2': (43.667, 1.700),
        'C3': (44.200, 7.341),
        'Cend': (None, None)
    }

    """Word HOUSE"""
    h_prior_probs = {
        'H1': 0.333,
        'H2': 0.000,
        'H3': 0.000,
        'Hend': 0.000
    }
    # Probability of a state changing to another state.
    h_transition_probs = {
        'H1': {'H1': 0.667, 'H2': 0.333, 'H3': 0.000, 'Hend': 0.000},
        'H2': {'H1': 0.000, 'H2': 0.857, 'H3': 0.143, 'Hend': 0.000},
        'H3': {'H1': 0.000, 'H2': 0.000, 'H3': 0.812, 'Hend': 0.188},
        'Hend': {'H1': 0.000, 'H2': 0.000, 'H3': 0.000, 'Hend': 1.000}
    }
    # Parameters for end state is not required
    h_emission_paras = {
        'H1': (45.333, 3.972),
        'H2': (34.952, 8.127),
        'H3': (67.438, 5.733),
        'Hend': (None, None)
    }

    return (b_prior_probs, b_transition_probs, b_emission_paras,
            c_prior_probs, c_transition_probs, c_emission_paras,
            h_prior_probs, h_transition_probs, h_emission_paras)

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

    def update_path_multi(self, obj_list, transition_probs, y, emission_paras):    
        curr_state = self.seq[-1]
        max_seq = self.seq
        p_max = self.prob * transition_probs[curr_state][curr_state][0] * transition_probs[curr_state][curr_state][1]
        for obj in obj_list:
            obj_state = obj.seq[-1]
            p_obj_curr = obj.prob * transition_probs[obj_state][curr_state][0] * transition_probs[obj_state][curr_state][1]
            if p_obj_curr >= p_max:
                max_seq = obj.seq
                p_max = p_obj_curr
        emission_prob = gaussian_prob(y[0], emission_paras[curr_state][0]) * gaussian_prob(y[1], emission_paras[curr_state][1])
        self.seq = max_seq.copy() 
        self.seq.append(curr_state)
        self.prob = p_max * emission_prob
            
    def update_path(self, obj, transition_probs, y, emission_paras):
        curr = self.seq[-1]
        prev = obj.seq[-1]
        value_curr_curr = self.prob * transition_probs[curr][curr]
        value_prev_curr = obj.prob * transition_probs[prev][curr]
        emission_prob = gaussian_prob(y, emission_paras[curr])

        if value_prev_curr >= value_curr_curr:
            # scrap the old path
            self.seq = obj.seq.copy()
            self.prob = obj.prob 
            self.apped_to_seq(curr, transition_probs[prev][curr] * emission_prob)
        else:
            # keep the old path
            t_p = transition_probs[curr][curr]
            e_p = gaussian_prob(y, emission_paras[curr])
            self.apped_to_seq(curr, transition_probs[curr][curr] * emission_prob)

def get_best_path_multi(path_list):
    best_obj = copy.deepcopy(path_list[0])
    for i in range(1, len(path_list)):
        if best_obj.prob <= path_list[i].prob:
            best_obj = copy.deepcopy(path_list[i])
    return best_obj


def get_best_path(path_list):
    
    P1 = path_list[0]
    P2 = path_list[1]
    P3 = path_list[2]
    best_obj = Path()
    if P1.prob == 0 and P2.prob == 0 and P3.prob == 0:
        best_obj.seq = None
        best_obj.prob = 0
        return best_obj

    if P1.prob >= P2.prob:
        if P1.prob > P3.prob:
            best_obj = P1
        else:
            best_obj = P3
    else:
        if P2.prob >= P3.prob:
            best_obj = P2
        else:
            best_obj = P3
    
    return best_obj 

def make_path(evidence_vector, states, prior_probs, transition_probs, emission_paras):
    path_1 = Path()
    path_2 = Path()
    path_3 = Path()

    state_1 = states[0] 
    state_2 = states[1]
    state_3 = states[2]

    y = evidence_vector[0]
    prior =  prior_probs[state_1]
    gaus_prob = gaussian_prob(y, emission_paras[state_1])
    exp_prob = prior * gaus_prob
    path_1.apped_to_seq(state_1,  exp_prob)

    prior = prior_probs[state_2]
    gaus_prob = gaussian_prob(y, emission_paras[state_2])
    exp_prob = prior * gaus_prob
    path_2.apped_to_seq(state_2, exp_prob)

    prior = prior_probs[state_3]
    gaus_prob = gaussian_prob(y, emission_paras[state_3])
    exp_prob = prior * gaus_prob
    path_3.apped_to_seq(state_3, exp_prob)
    
    for i in range(1, len(evidence_vector)):
        y = evidence_vector[i]
        # update 3 
        path_3.update_path(path_2, transition_probs, y, emission_paras)
        # update 2 
        path_2.update_path(path_1, transition_probs, y, emission_paras)
        # update 1
        path_1.apped_to_seq(state_1, transition_probs[state_1][state_1] * gaussian_prob(y, emission_paras[state_1]))

    return get_best_path([path_1, path_2, path_3])


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

    max_path_list = [b_max, c_max, h_max]
    model = get_best_path(max_path_list)
    sequence = model.seq
    probability = model.prob
    
    return sequence, probability

def part_2_a():
    """Provide probabilities for the word HMMs outlined below.

    Now, at each time frame you are given with 2 observations (right hand Y
    position & left hand Y position). Use the result you derived in
    part_1_a, accompany with the provided probability for left hand, create
    a tuple of (right-y, left-y) to represent high-dimention transition & 
    emission probabilities.
    """
    """Word BUY"""
    b_prior_probs = {
        'B1': 0.333,
        'B2': 0.000,
        'B3': 0.000,
        'Bend': 0.000
    }
    # example: {'B1': {'B1' : (right-hand Y, left-hand Y), ... }
    b_transition_probs = {
        'B1': {'B1': (0.625, 0.700), 'B2': (0.375, 0.300), 'B3': (0.000, 0.000), 'Bend': (0.000, 0.000)},
        'B2': {'B1': (0.000, 0.000), 'B2': (0.625, 0.050), 'B3': (0.375, 0.950), 'Bend': (0.000, 0.000)},
        'B3': {'B1': (0.000, 0.000), 'B2': (0.000, 0.000), 'B3': (0.625, 0.727), 'Bend': (0.125, 0.091), 'C1': (0.125,0.091), 'H1': (0.125,0.091)},
        'Bend': {'B1': (0.000, 0.000), 'B2': (0.000, 0.000), 'B3': (0.000, 0.000), 'Bend': (1.000, 1.000)},
    }
    # example: {'B1': [(right-mean, right-std), (left-mean, left-std)] ...}
    b_emission_paras = {
        'B1': [(41.75, 2.773), (108.200, 17.314)],
        'B2': [(58.625, 5.678), (78.670, 1.886)],
        'B3': [(53.125, 5.418), (64.182, 5.573)],
        'Bend': [(None, None), (None, None)]
    }

    """Word Car"""
    c_prior_probs = {
        'C1': 0.333,
        'C2': 0.000,
        'C3': 0.000,
        'Cend': 0.000
    }
    c_transition_probs = {
        'C1': {'C1': (0.667, 0.700), 'C2': (0.333, 0.300), 'C3': (0.000, 0.000), 'Cend': (0.000, 0.000)},
        'C2': {'C1': (0.000, 0.000), 'C2': (0.000, 0.625), 'C3': (1.000, 0.375), 'Cend': (0.000, 0.000)},
        'C3': {'C1': (0.000, 0.000), 'C2': (0.000, 0.000), 'C3': (0.800, 0.625), 'Cend': (0.067, 0.125), 'B1': (0.067, 0.125), 'H1': (0.067, 0.125)},
        'Cend': {'C1': (0.000, 0.000), 'C2': (0.000, 0.000), 'C3': (0.000, 0.), 'Cend': (1.000, 1.000)}
    }
    c_emission_paras = {
        'C1': [(35.667, 4.899), (56.300, 10.659)],
        'C2': [(43.667, 1.700), (37.110, 4.306)],
        'C3': [(44.200, 7.341), (50.000, 7.826)],
        'Cend': [(None, None), (None, None)]
    }

    """Word HOUSE"""
    h_prior_probs = {
        'H1': 0.333,
        'H2': 0.000,
        'H3': 0.000,
        'Hend': 0.000
    }
    h_transition_probs = {
        'H1': {'H1': (0.667, 0.700), 'H2': (0.333, 0.300), 'H3': (0.000, 0.000), 'Hend': (0.000, 0.000)},
        'H2': {'H1': (0.000, 0.000), 'H2': (0.857, 0.842), 'H3': (0.143, 0.158), 'Hend': (0.000, 0.000)},
        'H3': {'H1': (0.000, 0.000), 'H2': (0.000, 0.000), 'H3': (0.812, 0.824), 'Hend': (0.063, 0.059), 'B1': (0.063, 0.059), 'C1': (0.063, 0.059)},
        'Hend': {'H1': (0.000, 0.000), 'H2': (0.000, 0.000), 'H3': (0.000, 0.000), 'Hend': (1.000, 1.000)}
    }
    h_emission_paras = {
        'H1': [(45.333, 3.972), (53.600, 7.392)],
        'H2': [(34.952, 8.127), (37.168, 8.875)],
        'H3': [(67.438, 5.733), (74.176, 8.347)],
        'Hend': [(None, None), (None, None)]
    }

    return (b_prior_probs, b_transition_probs, b_emission_paras,
            c_prior_probs, c_transition_probs, c_emission_paras,
            h_prior_probs, h_transition_probs, h_emission_paras,)


def make_path_multi(evidence_vector, states, prior_probs, transition_probs, emission_paras):
    path_list_curr = []
    path_list_prev = []

    # 2. Initialise all the possible states by running a loop through all the states
    y = evidence_vector[0]
    for state in states:
        prior = prior_probs[state]
        gauss_prob = gaussian_prob(y[0], emission_paras[state][0]) * gaussian_prob(y[1], emission_paras[state][1])
        exp_prob = prior * gauss_prob
        path = Path() 
        path.apped_to_seq(state, exp_prob)
        path_list_curr.append(path)
    
    path_list_prev = copy.deepcopy(path_list_curr)
    
    # 3. Go through the evidence vector starting from index 1 and update paths as necessary
    for j in range(1 ,len(evidence_vector)):
        # for each sample
        y = evidence_vector[j]
        for i in range(0, len(path_list_curr)):
            path = path_list_curr[i]
            # for each path
            if path.seq[-1] == 'B1':
                obj_list = [path_list_prev[5], path_list_prev[8]]
            elif path.seq[-1] == 'C1':
                obj_list = [path_list_prev[8], path_list_prev[2]]
            elif path.seq[-1] == 'H1':
                obj_list = [path_list_prev[2], path_list_prev[5]]
            else:
                obj_list = [path_list_prev[i-1]]
            
            path.update_path_multi(obj_list, transition_probs, y, emission_paras)

        path_list_prev = copy.deepcopy(path_list_curr)

    return path_list_curr



def multidimensional_viterbi(evidence_vector, states, prior_probs,
                             transition_probs, emission_paras):
    """Decode the most likely word phrases generated by the evidence vector.

    States, prior_probs, transition_probs, and emission_probs will now contain
    all the words from part_2_a.
    """
    if evidence_vector == []:
        return (None, 0)  

    sample_states = []

    # 1. Get the sample states by removing the end states
    for state in states:
        if state == 'Bend' or state == 'Cend' or state == 'Hend':
            continue
        else:
            sample_states.append(state)
    
    path_list = make_path_multi(evidence_vector, sample_states, prior_probs, transition_probs, emission_paras)
    max_path = get_best_path_multi(path_list)

    sequence = max_path.seq
    probability = max_path.prob 

    return sequence, probability


def return_your_name():
    """Return your name
    """
    return "Manan Patel"