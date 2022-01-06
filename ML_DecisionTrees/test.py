import numpy as np

b = [1, 2, 3]

b = np.array(b)
a = np.zeros(len(b), int)
print(a)
print(b)

a = np.array([])
print(a)
if not np.size(a) > 0:
    print('dfdfd')

c = [[1, 2, 3], [3, 3, 4], [2, 4, 4]]
print(np.transpose(c))
c = np.array(c)
d = np.zeros(np.shape(c), int)
print(d)
d[0] = [1, 0, 1]
print(d)

print(c[:, 0])

data = [[1, 2, 3], [3, 4, 5], [7, 8, 0]]
data = np.array(data, int)

print(data)
new_data = np.delete(data, 1, 1)
print(new_data)

