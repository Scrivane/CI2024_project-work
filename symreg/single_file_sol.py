import operator
import numpy as np

from icecream import ic
from gxgp import Node
import gxgp
from gxgp import *
from tqdm import tqdm
from dataclasses import dataclass


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
""" problem = np.load('/home/adri/universita/magistrale/5_anno_1_sem/computational_intelligence/project_work/CI2024_project-work/data/problem_{}.npz'.format(i))
x = problem['x']
y = problem['y'] """

history=[]

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

def tournament_sel_array(population,n=2): # tournament selection to decide which individual to crossover



    # Choose n elements at random from the specified row
    fitness_list=[]
    random_elements_indexes = np.random.choice(len(population), size=n, replace=True)
    for el in random_elements_indexes:
        fitness_list.append((population[el]).fitness)


    bestIndex=np.argmin(fitness_list)
    bestValue=fitness_list[bestIndex]
    
    chosen=population[random_elements_indexes[bestIndex]]

    return chosen

@dataclass
class Individual:
    genome: Node
    fitness: float = None

def fitness(tree,nodefun,x, y):


    with np.errstate(all="raise"):          # only for the code inside
        try:
            mse_val = tree.mse(nodefun, x, y)
        except (FloatingPointError, ZeroDivisionError, ValueError):
            return math.inf                 # numerical failure → worst fitness

    # Some operations may finish without raising but still yield nan/inf
    if not np.isfinite(mse_val):
        return math.inf 
    

    """ try :
        result=tree.mse(nodefun,x,y)
    except err:
        result=math.inf


    return result """


def popInizialize(gpTree,nelements,x,y):
    #pop.append(Individual(genome=startgreedy(el)))
    localnode=gpTree.create_individual(15)  #change
    population = [Individual(genome=localnode,fitness=fitness(gpTree,localnode,x,y)) for _ in range(nelements)]
    return population

def tournament_sel_array(population,n=2): # tournament selection to decide which individual to crossover



    # Choose n elements at random from the specified row
    fitness_list=[]
    random_elements_indexes = np.random.choice(len(population), size=n, replace=True)
    for el in random_elements_indexes:
        fitness_list.append((population[el]).fitness)


    bestIndex=np.argmin(fitness_list)
    bestValue=fitness_list[bestIndex]
    
    chosen=population[random_elements_indexes[bestIndex]]

    return chosen


def EAlgoithm(nproblem,gptree):
    problem = np.load('/home/adri/universita/magistrale/5_anno_1_sem/computational_intelligence/project_work/CI2024_project-work/data/problem_{}.npz'.format(nproblem))
    x = problem['x']
    y = problem['y']

    ic(x.shape)
    
    mutrate=1
    nelemets=200
    pop=popInizialize(gptree,nelemets,x,y)
    nstep=1000
    lowest_fitness_individual = min(pop, key=lambda x: x.fitness)
    ic(lowest_fitness_individual)
    ic(str(lowest_fitness_individual.genome))
    #ic(lowest_fitness_individual.fitness)

    for el in tqdm(range(nstep)):
       
        """ if min(pop, key=lambda x: x.fitness).fitness==max(pop, key=lambda ind: ind.fitness).fitness:  #reached steady state
            break """
        ris1=tournament_sel_array(pop,2)   # uses tournament selection to select 2 parent for the crossover
        ris2=tournament_sel_array(pop,2)
        
        
        child = gxgp.xover_swap_subtree(ris1.genome,ris2.genome)  # uses partially mapped crossover
        child=Individual(genome=child,fitness=fitness(gptree,child,x,y))
     
        if( np.random.rand()<mutrate):  # has a chance of mutating the child using inversion mutation
            mutedgenome=gxgp.mutation_hoist(child.genome)
            child=Individual(genome=mutedgenome,fitness=fitness(gptree,mutedgenome,x,y))

        
        
        ic(fitness(gptree,child.genome,x,y))
        pop.append(child)  # add child to population
        
                

        lowest_fitness_individual = min(pop, key=lambda x: x.fitness)
        #ic(lowest_fitness_individual.fitness)
        history.append(lowest_fitness_individual.fitness)
        
        maxfit_ind=(max(pop, key=lambda ind: ind.fitness))  
        #ic(maxfit_ind.fitness)
        pop.remove(maxfit_ind) # delete from population the wrost individual
    

    
    lowest_fitness_individual = min(pop, key=lambda x: x.fitness)
    ic(lowest_fitness_individual)
    ic(str(lowest_fitness_individual.genome))
    


    return pop
    

numproblem=2
problem = np.load('/home/adri/universita/magistrale/5_anno_1_sem/computational_intelligence/project_work/CI2024_project-work/data/problem_{}.npz'.format(numproblem))
x = problem['x']
y=problem['y']
ic(y[2])
ic(x[:,2])

dag = gxgp.DagGP(
    operators=[np.add,np.negative,ignore_op,np.multiply,np.sin,np.tanh,np.reciprocal,np.exp,np.exp2,np.pow,np.sinh,np.cosh],#np.round #np.pow, #np.ldexp],
    variables=x.shape[0],
    constants=[],#np.linspace(-2, 2, 500),
)
pop=EAlgoithm(numproblem,dag)
#print(pop)
#3741781815.1038804
#14219046445.359241

""" single=dag.create_individual()

print(str(single))
single=dag.create_individual()


single.draw(True) """

""" multiple=dag.create_individuals()
for el in multiple:
    print(str(el)) """










""" nelements=10000
population = [dag.create_individual(100) for _ in range(nelements)]
from math import inf
minpossible=inf
for el in population:
    print(str(el))
    ris=dag.mse(el, x, y)
    if minpossible>ris:
        minpossible=ris

    
print(minpossible) """
#14219046346.709366
#8443461224.795478
#5652132292.672331
#13073143822.91186
#3241142940.98623



#14218963895.333452
#14218826655.570127
#print(gp.mse())
#mse(individual: Node, X, y, variable_names=None):