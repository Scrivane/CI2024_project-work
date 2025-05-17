import operator
import numpy as np

from icecream import ic
from gxgp import Node
import gxgp
from gxgp import *

tree = Node(np.multiply, [Node(np.add, [Node(10), Node('x')]), Node(2)])
#tree.draw()
tree(x=np.array([1, 2, 3, 4, 5]))
tree2 = Node(
    lambda a, b, c, d, e: (a + b * c - d) / e, [Node('x'), Node('y'), Node(3), Node('z'), Node(2)]
)
#tree2.draw()
# Explicit keyword arguments (unused arguments are ignored)
tree2(x=1, y=2, z=3, foo=42)
# Unpacking a dictionary
vars = {'x': 1, 'y': 2, 'z': 3, 'foo': 42}
tree2(**vars)
str(tree)



""" import os

# List all .py files in the current directory (excluding hidden files)
for file in os.listdir('../'):
    #if file.endswith('.py') and not file.startswith('.'):
        print(file) """



i=1
problem = np.load('/home/adri/universita/magistrale/5_anno_1_sem/computational_intelligence/project_work/CI2024_project-work/data/problem_{}.npz'.format(i))
x = problem['x']
y = problem['y']
ic(x.shape)
import math

""" def round_factorial(x):
    
    vec_fact = np.vectorize(math.factorial, otypes=[float])
    return vec_fact(np.round(x)) """
def safe_factorial(x):
    n = int(round(x))
    if n < 0:
        n = 0
    elif n > 20:
        n = 20
    return float(math.factorial(n))

def ignore_op(x):
    return 0.0


dag = gxgp.DagGP(
    operators=[np.pow,np.add,np.negative,ignore_op,np.multiply,np.sin],  #np.ldexp],
    variables=x.shape[0],
    constants=[],#np.linspace(-2, 2, 500),
)

""" single=dag.create_individual()

print(str(single))
single=dag.create_individual()


single.draw(True) """

""" multiple=dag.create_individuals()
for el in multiple:
    print(str(el)) """

nelements=10000
population = [dag.create_individual(100) for _ in range(nelements)]
from math import inf
minpossible=inf
for el in population:
    print(str(el))
    ris=dag.mse(el, x, y)
    if minpossible>ris:
        minpossible=ris

    
print(minpossible)
#14219046346.709366
#8443461224.795478
#5652132292.672331
#13073143822.91186
#3241142940.98623



#14218963895.333452
#14218826655.570127
#print(gp.mse())
#mse(individual: Node, X, y, variable_names=None):