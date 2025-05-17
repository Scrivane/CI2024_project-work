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



i=0
problem = np.load('/home/adri/universita/magistrale/5_anno_1_sem/computational_intelligence/project_work/CI2024_project-work/data/problem_{}.npz'.format(i))
x = problem['x']
y = problem['y']
ic(x.shape)


dag = gxgp.DagGP(
    operators=[np.sin,np.ldexp],
    variables=x.shape[0],
    constants=[],#np.linspace(-2, 2, 500),
)

""" single=dag.create_individual()

print(str(single))
single=dag.create_individual()


single.draw(True) """

multiple=dag.create_individuals()
for el in multiple:
    print(str(el))


for el in range(10):
    single=dag.create_individual()
    print(str(single))
#print(gp.mse())
#mse(individual: Node, X, y, variable_names=None):