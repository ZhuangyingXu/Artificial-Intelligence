import numpy as np

data = [[1, 1, 1], [1, 2, 1], [1, 3, 2], [1, 4, 3]]

data = np.array(data)
dataRows = np.shape(data)[0]

# pick k = 2
k = 2
num_rows = np.shape(data)[0]
rand_indices = np.random.choice(num_rows, size=2, replace=False)

means = data[rand_indices, :]   # this holds the means for each cluster
numFeaturs = np.shape(data)[1]

# go through meansRows times and calculate the covariance matrix each time
cov_mat_array = []
for i in range(0, np.shape(means)[0]):
    data_cpy = np.copy(data)
    row_i = np.reshape(means[i], (3, 1))            # will be (3, 1)
    row_i_mat = np.transpose(np.repeat(row_i, dataRows, 1))
    sub_mat = np.subtract(data_cpy, row_i_mat)

    # now fill in the 3 x 3 matrix
    cov_mat = np.zeros((numFeaturs, numFeaturs))

    for i in range(0, numFeaturs):
        for j in range(0, numFeaturs):
            # put an if condition
            if i > j:
                cov_mat[i][j] = cov_mat[j][i]
                continue
            M = np.multiply(sub_mat[:, i], sub_mat[:, j])
            M = np.mean(M)  # should be a single value
            cov_mat[i][j] = M
    cov_mat_array.append(cov_mat)

cov_mat_array = np.array(cov_mat_array)
p = 1 / np.shape(means)[0]
x = np.full((np.shape(means)[0], 1), p)

print(x)
print(np.shape(x))




