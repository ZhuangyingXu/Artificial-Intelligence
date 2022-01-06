from numpy.core.einsumfunc import einsum
from submission import *
import math

h = [45, 45, 46, 48, 51, 51, 49, 45, 42]
ans = 6.33777634 * math.pow(10, -13) 
ans = ans / 0.333
ans2 = 0.333
for value in h:
    ans = ans / gaussian_prob(value, para_tuple=(45.333, 3.972))
    ans2 = ans2 * gaussian_prob(value, para_tuple=(45.333, 3.972))
 

print(ans)
print(ans2)