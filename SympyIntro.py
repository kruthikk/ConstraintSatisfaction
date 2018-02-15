import matplotlib as mpl
import matplotlib.pyplot as plt

from util import constraint
from IPython.display import display
from sympy import *
init_printing()

x = Symbol('x')
display(x)

i,j,k = symbols(['i','j','k'])
display((i,j,k))

X = symbols("X:3")
display(X)

x,y = symbols('x y')
or_relation = x | y
display(or_relation)
y = x
display(y)

x, y, z = symbols("x y z")
display([x**2, x - y, Ne(x, y), (~x & y & z)])

x,y = symbols(['x','y'])
display (x)
display(y)
sameas = constraint("SameAs", Eq(x,y))
display(sameas)

E = None
sub = (x,y) = symbols('a b')
E = x|y
# test for completion

A, B, C = symbols(['A', 'B', 'C'])
maxAbsDiff = constraint("MaxAbsDiff", abs(A - B) < C)
display(maxAbsDiff)

Aval = symbols("A:3")
print(Aval[0])
maxAbsDiff_copy = maxAbsDiff.subs({A:Aval[0], B:Aval[1], C:Aval[2]})

display(maxAbsDiff_copy)
display(maxAbsDiff.free_symbols)
display(maxAbsDiff_copy.free_symbols)

assert(maxAbsDiff.free_symbols != maxAbsDiff_copy.free_symbols)
assert(len(maxAbsDiff_copy.free_symbols) == len(maxAbsDiff_copy.args))
inputs = {(0, 6, 7): True, (6, 0, 7): True, (7, 6, 0): False}
assert(all(maxAbsDiff_copy.subs(zip(Aval[:3], vals)) == truth for vals, truth in inputs.items()))
print("All tests passed!")


A = symbols('A:3')
display(A[0])
#allDiff = constraint("allDiff", ~(Eq(A[0],A[1])| Eq(A[2],A[1])| Eq(A[0],A[2])))
allDiff = constraint("allDiff", ~(Eq(A[0],A[1])^ Eq(A[2],A[1])^ Eq(A[0],A[2])))
display(allDiff)
display(allDiff.free_symbols)



