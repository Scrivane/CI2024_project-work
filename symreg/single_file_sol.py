import operator
import numpy as np

from icecream import ic
from gxgp import Node
import gxgp
from gxgp import *
from tqdm import tqdm
from dataclasses import dataclass
from matplotlib import pyplot as plt
from itertools import accumulate

import gxgp.random
import time

import sys

sys.path.append('/home/adri/universita/magistrale/5_anno_1_sem/computational_intelligence/project_work/CI2024_project-work')

import s333044
#http://tiny.cc/ci24_github

globaltime=0
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

def fitness(tree,nodefun:Node,x, y):
    start_time = time.perf_counter()


    with np.errstate(all="raise"):          # only for the code inside
        try:
            mse_val = tree.mse(nodefun, x.T, y)   #check this x.T
        except (FloatingPointError, ZeroDivisionError, ValueError):
            return math.inf                 # numerical failure → worst fitness

    # Some operations may finish without raising but still yield nan/inf
    if not np.isfinite(mse_val):
        return math.inf 
    


    tree_length = len(nodefun)
    penalty=5
    res= mse_val  + tree_length *penalty

    penalty_coeff = 0.001
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    global globaltime
    globaltime+=elapsed

    return mse_val + penalty_coeff * (tree_length ** 2)
    

    

    #return res 
    """ try :
        result=tree.mse(nodefun,x,y)
    except err:
        result=math.inf


    return result """




def error(tree,nodefun:Node,x, y):


    with np.errstate(all="raise"):          # only for the code inside
        try:
            mse_val = tree.mse(nodefun, x.T, y)
        except (FloatingPointError, ZeroDivisionError, ValueError):
            return math.inf                 # numerical failure → worst fitness

    # Some operations may finish without raising but still yield nan/inf
    if not np.isfinite(mse_val):
        return math.inf 
    
    return mse_val
    

def popInizialize(gpTree,nelements,x,y):   #probably pass neleemnts to create individuals and just use that  
    #pop.append(Individual(genome=startgreedy(el)))
    #localnode=gpTree.create_individual(30) #was 15
    population=[] #change  i thnk here is the problem, 
    while len(population)<nelements:
        locel=gpTree.create_individual(10)
        #locel=gxgp.random.gxgp_random.choice(localnode)  #maybe non reimbussola
                                  #random.sample(population, k, *, counts=None)
        fitnesslocel=fitness(gpTree,locel,x,y)
        #if (fitnesslocel<np.inf):
        population.append(Individual(genome=locel,fitness=fitnesslocel))
        """    else:
            print("malke") """
        #population = [Individual(genome=localnode,fitness=fitness(gpTree,localnode,x,y)) for _ in range(nelements)]
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

#find a way to limit depth
def EAlgoithm(nproblem,gptree):  #use elitism try to preserve best ones
    problem = np.load('/home/adri/universita/magistrale/5_anno_1_sem/computational_intelligence/project_work/CI2024_project-work/data/problem_{}.npz'.format(nproblem))
    x = problem['x']
    y = problem['y']

    minx = x.min()
    maxx = x.max()
    miny = y.min()
    maxy = y.max()
    minx_nonneg = np.min(x[x >= 0]) if np.any(x >= 0) else None
    maxx_neg = np.max(x[x < 0]) if np.any(x < 0) else None

    # For y
    miny_nonneg = np.min(y[y >= 0]) if np.any(y >= 0) else None
    maxy_neg = np.max(y[y < 0]) if np.any(y < 0) else None

    ic(x.shape)
    
    mutrate=0.05
    nrestarts=5
    nelemets=100#15   #400   #was 200
    
    nstep=10000#10000#1000 #100 #2000   #usa 4000
    """  lowest_fitness_individual = min(pop, key=lambda x: x.fitness)
    ic(lowest_fitness_individual)
    ic(str(lowest_fitness_individual.genome)) """
    #ic(lowest_fitness_individual.fitness)
    lowest_fitness_individual=popInizialize(gptree,1,x,y)[0]
    for _ in range(nrestarts):
        pop=popInizialize(gptree,nelemets,x,y)
        if lowest_fitness_individual not in pop:
            pop.append(lowest_fitness_individual)


        for el in tqdm(range(nstep)):
            """ lowest_fitness_individual = min(pop, key=lambda x: x.fitness)
            best_one=min(pop,key=lambda el: error(gptree,el.genome,x,y)) """
            #apply elitism
            """ if min(pop, key=lambda x: x.fitness).fitness==max(pop, key=lambda ind: ind.fitness).fitness:  #reached steady state
                break """
            

            

            """ values = [el.genome.value for el in pop]
            print(values) """
            ris1=tournament_sel_array(pop,2)   # uses tournament selection to select 2 parent for the crossover
            ris2=tournament_sel_array(pop,2)



            
            
            child = gxgp.xover_swap_subtree(ris1.genome,ris2.genome)  # uses partially mapped crossover
            child=Individual(genome=child,fitness=fitness(gptree,child,x,y))

        
            
        
            mutation_functions = [
            lambda genome: gxgp.mutation_point(genome, gptree),
        lambda genome: gxgp.mutation_hoist(genome),
        lambda genome: gxgp.mutation_permutations(genome)
            ]
            gxgp.random.gxgp_random.random
            if( np.random.rand()<mutrate):  # has a chance of mutating the child using inversion mutation
                mutation_fn = np.random.choice(mutation_functions)
            
                mutedgenome = mutation_fn(child.genome)
                child = Individual(genome=mutedgenome, fitness=fitness(gptree, mutedgenome, x, y))

            
            
        # ic(fitness(gptree,child.genome,x,y))
            pop.append(child)  # add child to population
            
            

            lowest_fitness_individual = min(pop, key=lambda x: x.fitness)

            
            
            #ic(lowest_fitness_individual.fitness)
            history.append(lowest_fitness_individual.fitness)
            
            maxfit_ind=(max(pop, key=lambda ind: ind.fitness))    #todo make negativ efitness and use a minimizer 
            #ic(maxfit_ind.fitness)
            pop.remove(maxfit_ind) # delete from population the wrost individual
        



        
        

        
        lowest_fitness_individual = min(pop, key=lambda x: x.fitness)
        print(lowest_fitness_individual.fitness)

        #ic(str(lowest_fitness_individual.genome))
        """ best_one=min(pop,key=lambda el: error(gptree,el.genome,x,y))
        ic(str(best_one.genome))
        ic(best_one.fitness)
        
        ic(formula)
     """
    best_one=min(pop,key=lambda el: error(gptree,el.genome,x,y))
    formula=best_one.genome.to_np_formula()
    ic(best_one.fitness)

    


    return formula,error(gptree,best_one.genome,x,y)
    
#2 problem 
#3197567994.386814
#3780431286.080886
#3241102705.3421373   mine
#19040148669684.434  ###others solution 
#3281516793.5310025  #others sol
#1.904e+15

numproblem=2
problem = np.load('/home/adri/universita/magistrale/5_anno_1_sem/computational_intelligence/project_work/CI2024_project-work/data/problem_{}.npz'.format(numproblem))
x = problem['x']  #3 righe 5000 colonne 

y=problem['y']
ic(y[2])  #5000 colonne
ic(x[:,2]) # i think x[2] is never used   , 3 righe , 1 colonna, è il secondo elemento

dag = gxgp.DagGP(   #problems with ldex , it's unsupported for some types   np.ldexp,
    #safe
    #operators=[np.pow],

    operators=[np.add,np.negative,np.multiply,np.sin,np.tanh,np.reciprocal,np.exp,np.exp2,np.pow,np.sinh,np.cosh,np.round,np.cosh,np.hypot,np.i0,np.absolute,np.square,np.log1p,np.log2,np.log10,np.log],  #np.acos], #ignore_op#np.round],  #np.round #np.pow, #np.ldexp],
    #operators=[np.log1p,np.exp2,np.i0,np.sin],
    variables=x.shape[0],
    constants=[np.pi,np.e,np.euler_gamma],
   # constants=[np.pi,np.e,np.euler_gamma],#[-1,0,1,np.pi,np.e,np.euler_gamma],#np.linspace(-2, 2, 500),
)

ic(np.var(y))
formula,fit=EAlgoithm(numproblem,dag)
print(formula)
ygen = eval(formula, {"np": np, "x": x})
if (ygen.size==1):
    ygen = ygen * np.ones(5000)

ic(ygen.shape)
ic(y.shape)
ic(x.shape)
ris= sum((a - b) ** 2 for a, b in zip(ygen, y)) / len(y) 
ic(ris)

if ris==fit:
    print("good")
else:
    print("wrong") 



ic(globaltime)
plt.figure(figsize=(14, 8))
plt.plot(
            range(len(history)),
            list(accumulate(history, min)),
            color="red",
        )
_ = plt.scatter(range(len(history)), history, marker=".")
plt.show() 
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