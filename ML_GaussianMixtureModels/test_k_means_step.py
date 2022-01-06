from numpy.core.fromnumeric import repeat
from numpy.lib.type_check import imag
import mixture_tests as tests
import numpy as np
from helper_functions import *
import math

def get_initial_means(array, k):
    """
    Picks k random points from the 2D array 
    (without replacement) to use as initial 
    cluster means

    params:
    array = numpy.ndarray[numpy.ndarray[float]] - m x n | datapoints x features

    k = int

    returns:
    initial_means = numpy.ndarray[numpy.ndarray[float]]
    """
    array = np.array(array, float)
    num_rows = np.shape(array)[0]
    rand_indices = np.random.choice(num_rows, size=k, replace=False)
    sliced_array = array[rand_indices, :]

    return sliced_array

def k_means_step(X, k, means):
    """
    A single update/step of the K-means algorithm
    Based on a input X and current mean estimate,
    predict clusters for each of the pixels and 
    calculate new means. 
    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n | pixels x features (already flattened)
    k = int
    means = numpy.ndarray[numpy.ndarray[float]] - k x n

    returns:
    (new_means, clusters)
    new_means = numpy.ndarray[numpy.ndarray[float]] - k x n
    clusters = numpy.ndarray[int] - m sized vector
    """
    X = np.array(X, float)
    means = np.array(means, float)

    # consider just the first data point
    xRows = np.shape(X)[0]
    xCols = np.shape(X)[1]

    mRows = np.shape(means)[0]
    mCols = np.shape(means)[1]

    # Creating clusters
    clusters = []
    for i in range(0, xRows):
        x1 = X[i]
        x1 = np.reshape(x1, (1, xCols))
        x1 = np.repeat(x1, mRows, 0)

        # now do element wise subtractions, square, add and take square root
        meansCpy = np.copy(means)
        sub_matrix = np.subtract(meansCpy, x1)
        sq_matrix = np.square(sub_matrix)
        sum_vector = np.sum(sq_matrix, 1)
        qrt_vector = sum_vector
        # qrt_vector = np.sqrt(sum_vector)        # a vector of mRows elements

        # find the index with the minimum value
        idx_min = np.argmin(qrt_vector)

        # clusters
        clusters.append(idx_min)

    clusters = np.array(clusters, int)
    sum_clusters = np.sum(clusters)

    # based on cluster index, find the new means for mRows centres
    new_means = []
    for i in range(0, mRows):
        # get all the indices of the dataPoints belonging to index i
        indicesArray = np.where(clusters == i)[0]

        # use this index list to slice from the data
        dataPoints = X[indicesArray, :]

        # get the new mean
        newMean = np.mean(dataPoints, 0)
        new_means.append(newMean)
    new_means = np.array(new_means)

    return (new_means, clusters)

def k_means_segment(image_values, k=3, initial_means=None):
    """
    Separate the provided RGB values into
    k separate clusters using the k-means algorithm,
    then return an updated version of the image
    with the original values replaced with
    the corresponding cluster values.

    params:
    image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - r x c x ch
    k = int
    initial_means = numpy.ndarray[numpy.ndarray[float]] or None

    returns:
    updated_image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - r x c x ch
    """

    # 1. Flatten the data
    image_values = np.array(image_values)
    x = np.shape(image_values)[0]
    y = np.shape(image_values)[1]
    z = np.shape(image_values)[2]        # likely to be three RGB

    image_values = np.reshape(image_values, (x * y, z))
    inital_means = get_initial_means(image_values, k)

    inital_means, initial_cluster = k_means_step(image_values, k, inital_means)
    while True:
        new_means, new_cluster = k_means_step(image_values, k, initial_means)
        if np.array_equal(new_cluster, initial_cluster):
            break
        else:
            initial_means = new_means
            initial_cluster = new_cluster
    
    update_image_values = np.zeros(np.shape(image_values))
    for i in range(0, k):
        indices_array = np.where(new_cluster == i)[0]
        update_image_values[indices_array, :] = new_means[i]
    
    update_image_values = np.reshape(update_image_values, (x, y, z))

    return update_image_values

def initialize_parameters(X, k):
    """
    Return initial values for training of the GMM
    Set component mean to a random
    pixel's value (without replacement),
    based on the mean calculate covariance matrices,
    and set each component mixing coefficient (PIs)
    to a uniform values
    (e.g. 4 components -> [0.25,0.25,0.25,0.25]).
    
    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int
    
    returns:
    (MU, SIGMA, PI)
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k x 1 
    """
    data = np.array(X)
    rand_indices = np.random.choice(np.shape(data)[0], size=k, replace=False)

    MU = data[rand_indices, :]
    SIGMA = compute_sigma(data, MU)
    PI = np.full((np.shape(MU)[0], 1), 1 / np.shape(MU)[0])

    return (MU, SIGMA, PI)

def compute_sigma(X, MU):
    """
    Calculate covariance matrix, based in given X and MU values
    
    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    
    returns:
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    """
    data = np.array(X)
    means = np.array(MU)
    dataRows = np.shape(data)[0]
    numFeatures = np.shape(data)[1]
    cov_mat_array = []
    for i in range(0, np.shape(means)[0]):
        data_cpy = np.copy(data)
        row_i = np.reshape(means[i], (3, 1))
        row_i_mat = np.transpose(np.repeat(row_i, dataRows, 1))
        sub_mat = np.subtract(data_cpy, row_i_mat)
    
        # now fill in the covariance matrix for the ith row
        cov_mat = np.zeros((numFeatures, numFeatures))

        for p in range(0, numFeatures):
            for q in range(0, numFeatures):
                if p > q:
                    # for similar off diagonal elements
                    cov_mat[p][q] = cov_mat[q][p]
                    continue
                M = np.multiply(sub_mat[:, p], sub_mat[:, q])
                M = np.mean(M)
                cov_mat[p][q] = M
        cov_mat_array.append(cov_mat)

    SIGMA = np.array(cov_mat_array)

    return SIGMA

def prob(x, mu, sigma):
    """Calculate the probability of x (a single
    data point or an array of data points) under the
    component with the given mean and covariance.
    The function is intended to compute multivariate
    normal distribution, which is given by N(x;MU,SIGMA).

    params:
    x = numpy.ndarray[float] or numpy.ndarray[numpy.ndarray[float]]
    mu = numpy.ndarray[float]
    sigma = numpy.ndarray[numpy.ndarray[float]]

    returns:
    probability = float or numpy.ndarray[float]
    """
    x = np.array(x)
    mu = np.array(mu)
    x_dim = np.ndim(x)

    sigma = np.array(sigma)
    det_sigma = np.linalg.det(sigma)

    if x_dim > 1:
        rows = np.shape(x)[0]
        cols = np.shape(x)[1]
        x = np.transpose(x)
        denom = ((2 * math.pi) ** (cols/2)) * (det_sigma) ** (1/2)
        mu = np.reshape(mu, (np.size(mu), 1))
        mu = np.repeat(mu, rows, 1)
        x_mu = np.subtract(x, mu)
        exponent = (-1/2) * np.transpose(x_mu)
        exponent = np.matmul(exponent, np.linalg.inv(sigma))
        ans2 = np.multiply(exponent, np.transpose(x_mu))
        ans2 = np.sum(ans2, 1)
        ans2 = np.exp(ans2)
        numerator = ans2

    else:
        denom = ((2 * math.pi) ** (3/2)) * ((det_sigma) ** (1/2))
        x_mu = np.subtract(x, mu)
        exponent = (-1/2) * np.transpose(x_mu)
        exponent = np.matmul(exponent, np.linalg.inv(sigma))
        exponent = np.matmul(exponent, x_mu)
        numerator = math.e ** exponent

    probability = (1 / denom) * numerator
    return probability

def E_step(X,MU,SIGMA,PI,k):
    """
    E-step - Expectation 
    Calculate responsibility for each
    of the data points, for the given 
    MU, SIGMA and PI.
    
    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k x 1
    k = int
    
    returns:
    responsibility = numpy.ndarray[numpy.ndarray[float]] - k x m
    """
    X = np.array(X)
    MU = np.array(MU)
    PI = np.array(PI)
    numDataPoints = np.shape(X)[0]
    SIGMA = np.array(SIGMA)
    sigma_0 = SIGMA[0]
    MU_0 = MU[0]
    prob_vector = prob(X, MU_0, sigma_0)
    responsibility = np.reshape(prob_vector, (1, np.size(prob_vector)))
    for i in range(1, k):
        sigma_i = SIGMA[i]
        MU_i = MU[i]
        prob_vector = prob(X, MU_i, sigma_i)
        prob_vector = np.reshape(prob_vector, (1, np.size(prob_vector)))
        responsibility = np.concatenate((responsibility, prob_vector), 0)
    
    PI = np.array(PI)
    PI = np.reshape(PI, (np.size(PI), 1))
    PI = np.repeat(PI, numDataPoints, 1)

    responsibility = np.multiply(responsibility, PI)
    sum_cols = np.sum(responsibility, 0)
    sum_cols = np.reshape(sum_cols, (1, np.size(sum_cols)))
    sum_cols = np.repeat(sum_cols, k, 0)
    responsibility = np.divide(responsibility, sum_cols)

    return responsibility


def M_step(X, r, k):
    """
    M-step - Maximization
    Calculate new MU, SIGMA and PI matrices
    based on the given responsibilities.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    r = numpy.ndarray[numpy.ndarray[float]] - k x m
    k = int

    returns:
    (new_MU, new_SIGMA, new_PI)
    new_MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    new_SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    new_PI = numpy.ndarray[float] - k x 1
    """
    X = np.array(X)
    numDataPoints = np.shape(X)[0]
    numFeatures = np.shape(X)[1]

    r = np.array(r)
    
    m_c = np.sum(r, 1)
    new_PI = (1 / numDataPoints) * m_c
    
    new_MU = []
    for c in range(0, k):
        r_c = r[c]
        r_c = np.reshape(r_c, (np.size(r_c), 1))
        r_c = np.repeat(r_c, numFeatures, 1)
        mu_c = np.multiply(X, r_c)
        mu_c = np.sum(mu_c, 0)
        # mu_c = np.reshape(mu_c, (1, np.size(mu_c)))
        mu_c = (1 / m_c[c]) * mu_c
        new_MU.append(mu_c)
    new_MU = np.array(new_MU)

    new_SIGMA = []
    for c in range(0, k):
        mc = m_c[c]                             # one value
        rc = r[c]       
        rc = np.reshape(rc, (np.size(rc), 1))   # 1 x numDataPoints
        rc = np.repeat(rc, numFeatures, 1)
        mu_c = new_MU[c]
        mu_c = np.reshape(mu_c, (1, np.size(mu_c)))
        mu_c = np.repeat(mu_c, numDataPoints, 0)
        x_mu_c = np.subtract(X, mu_c)
        x_mu_rc = np.multiply(x_mu_c, rc)
        sigma_c = (1 / mc) * np.matmul(np.transpose(x_mu_c), x_mu_rc)
        new_SIGMA.append(sigma_c)
    new_SIGMA = np.array(new_SIGMA)

    return (new_MU, new_SIGMA, new_PI)

#export
def likelihood(X, PI, MU, SIGMA, k):
    """Calculate a log likelihood of the 
    trained model based on the following
    formula for posterior probability:
    
    log(Pr(X | mixing, mean, stdev)) = sum((n=1 to N), log(sum((k=1 to K),
                                      mixing_k * N(x_n | mean_k,stdev_k))))

    Make sure you are using natural log, instead of log base 2 or base 10.
    
    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k x 1
    k = int

    returns:
    log_likelihood = float
    """
    X = np.array(X)
    PI = np.array(PI)
    MU = np.array(MU)
    SIGMA = np.array(SIGMA)
    
    numDataPoints = np.shape(X)[0]
    numFeatures = np.shape(X)[1]

    prob_list = []
    for i in range(0, k):
        mu_i = MU[i]
        sigma_i = SIGMA[i]
        p_i = prob(X, mu_i, sigma_i)        # 1 x numDataPoints
        prob_list.append(p_i)
    prob_list = np.array(prob_list)

    PI = np.reshape(PI, (np.size(PI), 1))   # column vector
    PI = np.repeat(PI, numDataPoints, 1)

    prob_list = np.multiply(PI, prob_list)
    prob_sum = np.sum(prob_list, 0)
    log_prob = np.log(prob_sum)
    log_likelihood = np.sum(log_prob) 

    return log_likelihood


def train_model(X, k, convergence_function, initial_values = None):
    """
    Train the mixture model using the 
    expectation-maximization algorithm. 
    E.g., iterate E and M steps from 
    above until convergence.
    If the initial_values are None, initialize them.
    Else it's a tuple of the format (MU, SIGMA, PI).
    Convergence is reached when convergence_function
    returns terminate as True,
    see default convergence_function example 
    in `helper_functions.py`

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int
    convergence_function = func
    initial_values = None or (MU, SIGMA, PI)

    returns:
    (new_MU, new_SIGMA, new_PI, responsibility)
    new_MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    new_SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    new_PI = numpy.ndarray[float] - k x 1
    responsibility = numpy.ndarray[numpy.ndarray[float]] - k x m
    """
    if initial_values == None:
        initial_values = initialize_parameters(X, k)
    
    prev_MU = np.array(initial_values[0])
    prev_SIGMA = np.array(initial_values[1])
    prev_PI = np.array(initial_values[2])
    prev_likelihood = likelihood(X, prev_PI, prev_MU, prev_SIGMA, k)

    r = E_step(X, prev_MU, prev_SIGMA, prev_PI, k)
    (new_MU, new_SIGMA, new_PI) = M_step(X, r, k)
    new_likelihood = likelihood(X, new_PI, new_MU, new_SIGMA, k)

    conv_ctr, terminate = convergence_function(prev_likelihood, new_likelihood, 0)
    prev_likelihood = new_likelihood
    while not terminate:
        r = E_step(X, new_MU, new_SIGMA, new_PI, k)
        (new_MU, new_SIGMA, new_PI) = M_step(X, r, k)
        new_likelihood = likelihood(X, new_PI, new_MU, new_SIGMA, k)
        conv_ctr, terminate = convergence_function(prev_likelihood, new_likelihood, conv_ctr)
        prev_likelihood = new_likelihood

    return (new_MU, new_SIGMA, new_PI, r)


def cluster(r):
    """
    Based on a given responsibilities matrix
    return an array of cluster indices.
    Assign each datapoint to a cluster based,
    on component with a max-likelihood 
    (maximum responsibility value).
    
    params:
    r = numpy.ndarray[numpy.ndarray[float]] - k x m - responsibility matrix
    
    return:
    clusters = numpy.ndarray[int] - m x 1 
    """
    r = np.array(r)
    clusters = np.argmax(r, 0)

    return clusters

def segment(X, MU, k, r):
    """
    Segment the X matrix into k components. 
    Returns a matrix where each data point is 
    replaced with its max-likelihood component mean.
    E.g., return the original matrix where each pixel's
    intensity replaced with its max-likelihood 
    component mean. (the shape is still mxn, not 
    original image size)

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    k = int
    r = numpy.ndarray[numpy.ndarray[float]] - k x m - responsibility matrix

    returns:
    new_X = numpy.ndarray[numpy.ndarray[float]] - m x n
    """
    X = np.array(X)
    MU = np.array(MU)
    r = np.array(r)
    indices = np.array(cluster(r))          # m x 1
    new_X = MU[indices, :]

    return new_X

def best_segment(X,k,iters):



    """Determine the best segmentation
    of the image by repeatedly
    training the model and
    calculating its likelihood.
    Return the segment with the
    highest likelihood.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int
    iters = int

    returns:
    (likelihood, segment)
    likelihood = float
    segment = numpy.ndarray[numpy.ndarray[float]]
    """
    max_likelihood_value = None
    initial_values = None
    max_segment = None
    for i in range(0, iters):
        # 1. train model
        (MU, SIGMA, PI, r) = train_model(X, k, default_convergence, initial_values)
        initial_values = (MU, SIGMA, PI, r)
        likelihood_value = likelihood(X, PI, MU, SIGMA, k)

        # 2. generate clusters based on r
        c = cluster(r)

        # 3. segment the data
        curr_segment = segment(X, MU, k, r)

        # 4. Update values
        if max_likelihood_value == None:
            max_likelihood_value = likelihood_value
            max_segment = curr_segment
        elif likelihood_value > max_likelihood_value:
            max_segment = curr_segment
    
    return (max_likelihood_value, max_segment)
            

def improved_initialization(X,k):
    """
    Initialize the training
    process by setting each
    component mean using some algorithm that
    you think might give better means to start with,
    based on the mean calculate covariance matrices,
    and set each component mixing coefficient (PIs)
    to a uniform values
    (e.g. 4 components -> [0.25,0.25,0.25,0.25]).
    
    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int
    
    returns:
    (MU, SIGMA, PI)
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k x 1 
    """
    (MU, SIGMA, PI, r) = train_model(X, k, default_convergence, None)

    return (MU, SIGMA, PI)


#export
def error_percent(previous, new):
    """
    Returns the max error percentage between
    previous and new values.

    params:
    previous = [np.ndarray[float]]
    new = [np.ndarray[float]]
    
    return:
    max_error_percent = float
    """
    previous = np.array(previous)
    new = np.array(new)

    error = np.abs(np.divide(np.subtract(new, previous), previous)) * 100
    max_error_percent = np.amax(error)

    return max_error_percent


def new_convergence_function(previous_variables, new_variables, conv_ctr,
                             conv_ctr_cap=10):
    """
    Convergence function
    based on parameters:
    when all variables vary by
    less than 10% from the previous
    iteration's variables, increase
    the convergence counter.

    params:
    previous_variables = [numpy.ndarray[float]]
                         containing [means, variances, mixing_coefficients]
    new_variables = [numpy.ndarray[float]]
                    containing [means, variances, mixing_coefficients]
    conv_ctr = int
    conv_ctr_cap = int

    return:
    (conv_crt, converged)
    conv_ctr = int
    converged = boolean
    """
    if conv_ctr > conv_ctr_cap:
        return (conv_ctr, True)

    prev_means = np.array(previous_variables[0])
    prev_variances = np.array(previous_variables[1])
    prev_mixing_coef = np.array(previous_variables[2])

    new_means = np.array(new_variables[0])
    new_variances = np.array(new_variables[1])
    new_mixing_coef = np.array(new_variables[2])

    error_list = [error_percent(prev_means, new_means),
                  error_percent(prev_variances, new_variances),
                  error_percent(prev_mixing_coef, new_mixing_coef)]
    error_list = np.array(error_list)
    max_error_percent = np.amax(np.array(error_list))

    condition = False
    if max_error_percent < 10.00:
        condition = True

    conv_ctr += 1
    return (conv_ctr, condition)   

import copy
def train_model_improved(X, k, convergence_function, initial_values = None):
    """
    Train the mixture model using the 
    expectation-maximization algorithm. 
    E.g., iterate E and M steps from 
    above until convergence.
    If the initial_values are None, initialize them.
    Else it's a tuple of the format (MU, SIGMA, PI).
    Convergence is reached when convergence_function
    returns terminate as True. Use new_convergence_fuction 
    implemented above. 

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int
    convergence_function = func
    initial_values = None or (MU, SIGMA, PI)

    returns:
    (new_MU, new_SIGMA, new_PI, responsibility)
    new_MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    new_SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    new_PI = numpy.ndarray[float] - k x 1
    responsibility = numpy.ndarray[numpy.ndarray[float]] - k x m
    """
    if initial_values == None:
        initial_values = improved_initialization(X, k)
    
    prev_MU = np.array(initial_values[0])
    prev_SIGMA = np.array(initial_values[1])
    prev_PI = np.array(initial_values[2])

    r = E_step(X, prev_MU, prev_SIGMA, prev_PI, k)
    (new_MU, new_SIGMA, new_PI) = M_step(X, r, k)
    new_values = (new_MU, new_SIGMA, new_PI)

    conv_ctr, terminate = new_convergence_function(initial_values, new_values, 0)
    prev_values = copy.copy(new_values)

    while not terminate:
        r = E_step(X, new_MU, new_SIGMA, new_PI, k)
        (new_MU, new_SIGMA, new_PI) = M_step(X, r, k)
        new_values = (new_MU, new_SIGMA, new_PI)
        conv_ctr, terminate = convergence_function(prev_values, new_values, conv_ctr)
        prev_values = copy.copy(new_values)

    return (new_MU, new_SIGMA, new_PI, r)
    
def bayes_info_criterion(X, PI, MU, SIGMA, k):
    """
    See description above
    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k x 1
    k = int
    
    return:
    bayes_info_criterion = int
    """
    numDataPoints = np.shape(X)[0]
    numFeatures = np.shape(X)[1]
    components = numFeatures * k + (k * numFeatures * (numFeatures + 1) / 2) + k
    return -2 * likelihood(X, PI, MU, SIGMA, k) + components * math.log(numDataPoints)



def BIC_likelihood_model_test(image_matrix, comp_means):
    """Returns the number of components
    corresponding to the minimum BIC 
    and maximum likelihood with respect
    to image_matrix and comp_means.
    
    params:
    image_matrix = numpy.ndarray[numpy.ndarray[float]] - m x n
    comp_means = list(numpy.ndarray[numpy.ndarray[float]]) - list(k x n) (means for each value of k)

    returns:
    (n_comp_min_bic, n_comp_max_likelihood)
    n_comp_min_bic = int
    n_comp_max_likelihood = int
    """
    image_matrix = np.array(image_matrix)
    numFeatures = np.shape(image_matrix)[1]

    n_comp_min_bic = None
    n_comp_max_likelihood = None

    max_likelihood = None
    min_bic = None

    for MU_i in comp_means: 
        k = np.shape(MU_i)[0]
        SIGMA = compute_sigma(image_matrix, MU_i)
        PI = np.full((k, 1), 1 / k)

        (MU, SIGMA, PI, r) = train_model(image_matrix, k, default_convergence, (MU_i, SIGMA, PI))
        curr_likelihood = likelihood(image_matrix, PI, MU, SIGMA, k)
        curr_BIC = bayes_info_criterion(image_matrix, PI, MU, SIGMA, k)

        if n_comp_max_likelihood == None:
            n_comp_max_likelihood = numFeatures * k + (k * numFeatures * (numFeatures + 1) / 2) + k
            max_likelihood = curr_likelihood
        elif curr_likelihood > max_likelihood:
            max_likelihood = curr_likelihood
            n_comp_max_likelihood = numFeatures * k + (k * numFeatures * (numFeatures + 1) / 2) + k
        
        if n_comp_min_bic == None:
            n_comp_min_bic = numFeatures * k + (k * numFeatures * (numFeatures + 1) / 2) + k
            min_bic = curr_BIC
        elif curr_BIC < min_bic:
            min_bic = curr_BIC
            n_comp_min_bic = numFeatures * k + (k * numFeatures * (numFeatures + 1) / 2) + k

    return (n_comp_min_bic, n_comp_max_likelihood)



def test_model():
    MU_list = []
    image_file = 'images/Starry.png'
    image_matrix = image_to_matrix(image_file).reshape(-1, 3)
    for i in range(1, 3):
        rand_indices = np.random.choice(np.shape(image_matrix)[0], size=i, replace=False)
        MU_i = image_matrix[rand_indices, :]
        MU_list.append(MU_i)
    
    print(BIC_likelihood_model_test(image_matrix, MU_list))
    print("passed!")

test_model()
# tests.GMMTests().test_bayes_info(bayes_info_criterion)