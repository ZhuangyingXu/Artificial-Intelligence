import numpy as np


a = [1, 2, 3, 4]
a = np.array(a)

a = np.reshape(a, (1, 4))
b = np.repeat(a, 3, 0)
c = b

d = [[1, 2, 3, 4], [5, 7, 9, 9], [9, 2, 3, 2]]

d = np.array(d)
sub_mat = np.subtract(d, c)
# print(sub_mat)
sq_mat = np.square(sub_mat)
# print(sq_mat)
sum_mat = np.sum(sq_mat, 1)
# print(sum_mat)
qrt_mat = np.sqrt(sum_mat)
# print(qrt_mat)

a = [1, 0, 0, 3]
c = [[1, 2, 3], [4, 5, 7], [8, 9, 10]]
c = np.array(c, int)
a = np.array(a)
# b = np.array(np.where(a == 0), int)
# b = np.reshape(b, (np.shape(b)[1]))
f = np.where(a == -1)[0]
# returns an empty list if no element found
# print(f)
b = np.where(a == 0)[0]
# print(b)

e = c[b, :]

# print(e)

t = np.sum(e, 0)
# print(t)

y = [1, 2, 3]
y = np.array(y, int)
# print(y)

a = [[[1, 2, 3], [1, 2, 3], [3, 4, 5]]]
print(np.shape(a))
a = np.array(a, int)
print(a)
d1 = np.shape(a)[0]
d2 = np.shape(a)[1]
d3 = np.shape(a)[1]

b = np.reshape(a, (d1 * d2, d3))
c = np.reshape(b, (d1, d2, d3))
print(b)
print(c)

b[[1, 2], :] = [0, 0, 0]
print(b)

print("new")
a = [1, 2, 3]
a = np.array(a)
a = np.reshape(a, (3, 1))
b = np.repeat(a, 5, 1)
b = np.transpose(b)
print(b)
print(np.sum(b, 0))

c = [1, 2, 3]
c = np.array(c)
c = np.sum(c)
print(c)


