from math import pi
import sys
from typing_extensions import ParamSpec
from networkx.algorithms.traversal.breadth_first_search import bfs_successors
from networkx.algorithms.tree.operations import join
from networkx.generators.geometric import thresholded_random_geometric_graph
from pgmpy.utils import state_name

'''
WRITE YOUR CODE BELOW.
'''
from numpy import mat, zeros, float32
import numpy as np
import random
#  pgmpy
import pgmpy
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
#You are not allowed to use following set of modules from 'pgmpy' Library.
#
# pgmpy.sampling.*
# pgmpy.factors.*
# pgmpy.estimators.*

def make_power_plant_net():
    """Create a Bayes Net representation of the above power plant problem. 
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    BayesNet = BayesianModel()
# TODO: finish this function    
    BayesNet.add_node("alarm")
    BayesNet.add_node("faulty alarm")
    BayesNet.add_node("gauge")
    BayesNet.add_node("faulty gauge")
    BayesNet.add_node("temperature")

#   BayesNet.add_edge(<parent node name>,<child node name>)

    BayesNet.add_edge("temperature", "faulty gauge")
    BayesNet.add_edge("temperature", "gauge")
    BayesNet.add_edge("faulty gauge", "gauge")
    BayesNet.add_edge("gauge", "alarm")
    BayesNet.add_edge("faulty alarm", "alarm")

    return BayesNet


def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system.
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    # TODO: set the probability distribution for each node
    cpd_T = TabularCPD('temperature', 2, values=[[0.80], [0.20]])                 # P(T)
    #                                                 [P(~FG|~T), P(~FG|T)], [P(FG|~T), P(FG|T)]
    cpd_FG_T = TabularCPD('faulty gauge', 2, values=[[  0.95    ,   0.20 ],  [0.05  ,   0.80]], evidence=['temperature'], evidence_card=[2])            # P(FG|T) distribution

    cpd_G_TandFG = TabularCPD('gauge', 2, values=[[0.95, 0.20, 0.05, 0.80], [0.05, 0.8, 0.95, 0.20]], evidence=['temperature', 'faulty gauge'], evidence_card=[2, 2])

    cpd_FA = TabularCPD('faulty alarm', 2, values=[[0.85], [0.15]])

    cpd_A = TabularCPD('alarm', 2, values=[[0.90, 0.55, 0.10, 0.45], [0.10, 0.45, 0.90, 0.55]], evidence=['gauge', 'faulty alarm'], evidence_card=[2, 2])

    bayes_net.add_cpds(cpd_T, cpd_FG_T, cpd_G_TandFG, cpd_FA, cpd_A)
    
    return bayes_net

def get_FG_prob(bayes_net):
    """Calculate the marginal
    probability of the gauge being
    faulty."""
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['faulty gauge'], joint=False)
    prob = marginal_prob['faulty gauge'].values
    return prob[1]

def get_alarm_prob(bayes_net):
    """Calculate the marginal 
    probability of the alarm 
    ringing in the 
    power plant system."""
    # TODO: finish this function
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['alarm'], joint=False)
    alarm_prob = marginal_prob['alarm'].values[1]
    return alarm_prob


def get_gauge_prob(bayes_net):
    """Calculate the marginal
    probability of the gauge 
    showing hot in the 
    power plant system."""
    # TODO: finish this function
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['gauge'], joint=False)
    gauge_prob = marginal_prob['gauge'].values[1]
    return gauge_prob


def get_temperature_prob(bayes_net):
    """Calculate the conditional probability 
    of the temperature being hot in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""
    # TODO: finish this function
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['temperature'], evidence={'alarm':1, 'faulty alarm':0, 'faulty gauge':0}, joint=False)
    temp_prob = conditional_prob['temperature'].values[1]
    return temp_prob


def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """

    # TODO: fill this out
    BayesNet = BayesianModel()
    BayesNet.add_node("A")
    BayesNet.add_node("B")
    BayesNet.add_node("C")
    BayesNet.add_node("AvB")
    BayesNet.add_node("BvC")
    BayesNet.add_node("CvA")
    BayesNet.add_edge("A", "AvB")
    BayesNet.add_edge("A", "CvA")
    BayesNet.add_edge("B", "AvB")
    BayesNet.add_edge("B", "BvC")
    BayesNet.add_edge("C", "CvA")
    BayesNet.add_edge("C", "BvC")
    cpd_A = TabularCPD('A', 4, values=[[0.15], [0.45], [0.30], [0.10]])
    cpd_B = TabularCPD('B', 4, values=[[0.15], [0.45], [0.30], [0.10]])
    cpd_C = TabularCPD('C', 4, values=[[0.15], [0.45], [0.30], [0.10]])
    final_array = []
    A = [0, 1, 2, 3]    
    B = [0, 1, 2, 3]
    result = [0, 1, 2]
    outcome_table = [[0.1, 0.1, 0.8], [0.2, 0.6, 0.2], [0.15, 0.75, 0.10], [0.05, 0.90, 0.05]]
    skill_level = [0.15, 0.45, 0.30, 0.10]
    for r in result:
        prob_array = []
        for a_skill in A:                   # A is team 1
            for b_skill in B:               # B is team 2
                skill_diff = b_skill - a_skill
                if(skill_diff < 0):
                    if(r == 0):
                        multiplier = outcome_table[abs(skill_diff)][r+1]
                    elif(r == 1):
                        multiplier = outcome_table[abs(skill_diff)][r-1]
                    elif(r == 2):
                        multiplier = outcome_table[abs(skill_diff)][r] 
                else:
                    multiplier = outcome_table[skill_diff][r]
                prob_array.append(round(multiplier, 3))
        final_array.append(prob_array)
    cpd_AvB = TabularCPD('AvB', 3, values=[final_array[0], final_array[1], final_array[2]], evidence=['A', 'B'], evidence_card=[4, 4])
    cpd_BvC = TabularCPD('BvC', 3, values=[final_array[0], final_array[1], final_array[2]], evidence=['B', 'C'], evidence_card=[4, 4])
    cpd_CvA = TabularCPD('CvA', 3, values=[final_array[0], final_array[1], final_array[2]], evidence=['C', 'A'], evidence_card=[4, 4])
    BayesNet.add_cpds(cpd_A, cpd_B, cpd_C, cpd_AvB, cpd_BvC, cpd_CvA)
    return BayesNet


def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0,0,0]
    # TODO: finish this function    
    solver = VariableElimination(bayes_net)
    conditional_probability = solver.query(variables=['BvC'], evidence={'AvB':0, 'CvA':2}, joint=False)
    for i in range(0, 3):
        posterior[i] = conditional_probability['BvC'].values[i]
    return posterior # list 


def get_prob(match_table, team1, team2, result):
    r = result
    prob = 0
    if(team2 - team1 < 0):
        if(r == 0):
            prob = match_table[r+1][team2][team1]
        elif(r == 1):
            prob = match_table[r-1][team2][team1]
        elif(r == 2):
            prob = match_table[r][team1][team2]
    else:
        prob = match_table[r][team1][team2]
    return prob


def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """

    # TODO: finish this function
    skill_level = [0, 1, 2, 3]
    result = [0, 1, 2]

    A_0 = None
    B_0 = None
    C_0 = None
    AvB_0 = 0   # A beats B
    BvC_0 = None
    CvA_0 = 2   # A ties with C

    var_idx_list = [0, 1, 2, 4]
    A_cpd = bayes_net.get_cpds("A")
    team_table = A_cpd.values
    BvC_cpd = bayes_net.get_cpds("BvC")
    match_table = BvC_cpd.values

    # Check for empty/None list
    if(initial_state == None or initial_state == []):
        # Setup a random initial state
        A_0 = random.choice(skill_level)
        B_0 = random.choice(skill_level)
        C_0 = random.choice(skill_level)
        BvC_0 = random.choice(result)
        initial_state = [A_0, B_0, C_0, AvB_0, BvC_0, CvA_0]
    else:
        A_0 = initial_state[0]
        B_0 = initial_state[1]
        C_0 = initial_state[2]
        BvC_0 = initial_state[4]

    # pick a variable to sample
    var_idx = random.choice(var_idx_list)
    value = None
    if(var_idx == 0):
        # TO DO WHEN A is to be updated
        term_list = []
        p_B = team_table[B_0]
        p_C = team_table[C_0]
        for a_skill in skill_level:
            p_A = team_table[a_skill]
            p_AvB = get_prob(match_table, a_skill, B_0, AvB_0)
            p_BvC = get_prob(match_table, B_0, C_0, BvC_0)
            p_CvA = get_prob(match_table, C_0, a_skill, CvA_0)
            term = p_AvB * p_BvC * p_CvA * p_A * p_B * p_C
            term_list.append(term)
        sum_terms = sum(term_list)
        wt0 = term_list[0] / sum_terms
        wt1 = term_list[1] / sum_terms
        wt2 = term_list[2] / sum_terms
        wt3 = term_list[3] / sum_terms
        prob_wts = [wt0, wt1, wt2, wt3]
        value = random.choices(skill_level, prob_wts)[0]
    elif(var_idx == 1):
        # TO DO WHEN B is to be updated
        term_list = []
        p_A = team_table[A_0]
        p_C = team_table[C_0]
        for b_skill in skill_level:
            p_B = team_table[b_skill]
            p_AvB = get_prob(match_table, A_0, b_skill, AvB_0)
            p_BvC = get_prob(match_table, b_skill, C_0, BvC_0)
            p_CvA = get_prob(match_table, C_0, A_0, CvA_0)
            term = p_AvB * p_BvC * p_CvA * p_A * p_B * p_C
            term_list.append(term)
        sum_terms = sum(term_list)
        wt0 = term_list[0] / sum_terms
        wt1 = term_list[1] / sum_terms
        wt2 = term_list[2] / sum_terms
        wt3 = term_list[3] / sum_terms
        prob_wts = [wt0, wt1, wt2, wt3]
        value = random.choices(skill_level, prob_wts)[0]
    elif(var_idx == 2):
        # TO DO WHEN C is to be updated
        term_list = []
        p_A = team_table[A_0]
        p_B = team_table[B_0]
        for c_skill in skill_level:
            p_C = team_table[c_skill]
            p_AvB = get_prob(match_table, A_0, B_0, AvB_0)
            p_BvC = get_prob(match_table, B_0, c_skill, BvC_0)
            p_CvA = get_prob(match_table, c_skill, A_0, CvA_0)
            term = p_AvB * p_BvC * p_CvA * p_A * p_B * p_C
            term_list.append(term)
        sum_terms = sum(term_list)
        wt0 = term_list[0] / sum_terms
        wt1 = term_list[1] / sum_terms
        wt2 = term_list[2] / sum_terms
        wt3 = term_list[3] / sum_terms
        prob_wts = [wt0, wt1, wt2, wt3]
        value = random.choices(skill_level, prob_wts)[0]
    else:
        # TO DO WHEN BvC is to be updated
        prob_wts = []
        for r in range(0, 3):
            # Calculate the weights for the random number
            prob = get_prob(match_table, B_0, C_0, r)
            prob_wts.append(prob)
        value = random.choices(result, prob_wts)[0]
            
    initial_state[var_idx] = value 
    sample = tuple(initial_state)    
    return sample


def get_pState(team_table, match_table, A, B, C, AvB, BvC, CvA):
    pA = team_table[A]
    pB = team_table[B] 
    pC = team_table[C]
    pAvB = get_prob(match_table, A, B, AvB)
    pBvC = get_prob(match_table, B, C, BvC)
    pCvA = get_prob(match_table, C, A, CvA)
    num = pAvB * pBvC * pCvA * pA * pB * pC
    return num


def pick_from_table(value, table):
    if(random.choice([0, 1])):
        return random.choice(table)
    else:
        return value


def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """
    A_cpd = bayes_net.get_cpds("A")      
    AvB_cpd = bayes_net.get_cpds("AvB")
    match_table = AvB_cpd.values
    team_table = A_cpd.values

    skill_level = [0, 1, 2, 3]
    result = [0, 1, 2]
    A_0 = None
    B_0 = None
    C_0 = None
    AvB_0 = 0
    BvC_0 = None
    CvA_0 = 2

    # 1. If initial_state is empty or None
    if(initial_state == None or initial_state == []):
        A_0 = random.choice(skill_level)
        B_0 = random.choice(skill_level)
        C_0 = random.choice(skill_level)
        BvC_0 = random.choice(result)
        initial_state = [A_0, B_0, C_0, AvB_0, BvC_0, CvA_0]
        # return the initial state?
    else:
        A_0 = initial_state[0]
        B_0 = initial_state[1]
        C_0 = initial_state[2]
        BvC_0 = initial_state[4]
    
    # 2. Generating a new state using uniform distribution
    A_1 = pick_from_table(A_0, skill_level) 
    B_1 = pick_from_table(B_0, skill_level)
    C_1 = pick_from_table(C_0, skill_level)
    BvC_1 = pick_from_table(BvC_0, result) 
    new_state = [A_1, B_1, C_1, AvB_0, BvC_1, CvA_0]

    # 3. Compute the acceptance ratio
    # 3.1 Computer P(initial_state|AvB_0, CvA_0) = P(AvB_0, BvC_0, CvA_0 | A_0, B_0, C_0) * P(A_0) * P(B_0) * P(C_0) / P(AvB_0, BvC_0, CvA_0)
    pState_0 = get_pState(team_table, match_table, A_0, B_0, C_0, AvB_0, BvC_0, CvA_0)
    # 3.2 Compute P(new_state|AvB_0, CvA_0)
    pState_1 = get_pState(team_table, match_table, A_1, B_1, C_1, AvB_0, BvC_1, CvA_0)

    # 4. Computer acceptance ratio
    acpt_ratio = pState_1 / pState_0

    # 5. Decide whether to accept State_1
    # 5.1 Draw a u from Uniform (0, 1)
    u = random.uniform(0, 1)
    if(u <= acpt_ratio):
        initial_state = new_state
    sample = tuple(initial_state)    
    return sample


def compare_sampling(bayes_net, initial_state):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""    
    N = 100
    delta = 0.0001
 
    Gibbs_count = 0
    Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs 
    prev_Gibbs_convergence = [0, 0, 0]
    Gibbs_initial_state = initial_state
    counter = 0
    counter_started = False
    BvC_0 = 0
    BvC_1 = 0
    BvC_2 = 0
    i = 1

    while(True):
        state = Gibbs_sampler(bayes_net, Gibbs_initial_state)
        Gibbs_initial_state = list(state)
        BvC = state[4]
        if(BvC == 0):
            BvC_0 += 1
        elif(BvC == 1):
            BvC_1 += 1
        elif(BvC == 2):
            BvC_2 += 1
        Gibbs_convergence = [BvC_0 / i, BvC_1 / i, BvC_2 / i]
        if(abs(Gibbs_convergence[0] - prev_Gibbs_convergence[0]) < delta and
           abs(Gibbs_convergence[1] - prev_Gibbs_convergence[1]) < delta and 
           abs(Gibbs_convergence[2] - prev_Gibbs_convergence[2]) < delta):
           counter_started = True
        else:
            counter_started = False
            counter = 0
        if(counter_started):
            if(counter == N):
                break
            else:
                counter += 1
        prev_Gibbs_convergence = Gibbs_convergence
        i += 1
    Gibbs_count = i

    MH_count = 0
    MH_rejection_count = 100
    MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH
    prev_MH_convergence = [0, 0, 0]
    MH_initial_state = initial_state
    BvC_0 = 0
    BvC_1 = 0
    BvC_2 = 0
    i = 1
    counter = 0
    counter_started = False
    reject = True
    reject_count = 0
    while(True):
        state = MH_sampler(bayes_net, MH_initial_state)
        MH_initial_state = list(state)
        if(reject):
            reject_count += 1
            if(reject_count == MH_rejection_count):
                reject = False
        BvC = state[4]
        if(BvC == 0):
            BvC_0 += 1
        elif(BvC == 1):
            BvC_1 += 1
        elif(BvC == 2):
            BvC_2 += 1
        MH_convergence = [BvC_0 / i, BvC_1 / i, BvC_2 / i]
        if(abs(MH_convergence[0] - prev_MH_convergence[0]) < delta and
           abs(MH_convergence[1] - prev_MH_convergence[1]) < delta and 
           abs(MH_convergence[2] - prev_MH_convergence[2]) < delta):
           counter_started = True
        else:
            counter_started = False
            counter = 0
        if(counter_started):
            if(counter == N):
                break
            else:
                counter += 1
        prev_MH_convergence = MH_convergence
        i += 1
    MH_count = i
    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count


def sampling_question():
    """Question about sampling performance."""
    choice = 1
    options = ['Gibbs','Metropolis-Hastings']
    factor = 1
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    return "Manan Patel"