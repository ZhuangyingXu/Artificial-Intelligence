import copy
a_list = ['B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'H1', 'H2', 'H3']
def condition(x): return x[-1] == '1'
b = enumerate(a_list)
output = [element for idx, element in enumerate(a_list) if condition(element)]
print(output)

class Path:
    def __init__ (self):
        x = 0

a = [[1, 2, 3], [3, 4]]
b = a.copy()
c = copy.deepcopy(a)
a[0].append(1)

print(a)
print(b)
print(c)