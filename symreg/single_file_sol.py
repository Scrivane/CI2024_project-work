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
""" tree = Node(np.multiply, [Node(np.add, [Node(10), Node('x')]), Node(2)])
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
str(tree) """



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


@dataclass
class Individual:
    genome: Node
    fitness: float = None



def length_penalty(length, threshold=2000, scale=1e6):
    if length <= threshold:
        #print(0.001*length*scale)
        return   0.001*length*scale#scale * 0.001 * length**2
    else:
        base = scale * 0.001 * threshold**2
        blowup = np.exp((length - threshold) / 10) * scale
        return base + blowup

def fitness(tree,nodefun:Node,x, y):
    #start_time = time.perf_counter()


    with np.errstate(all="raise"):          # only for the code inside
        try:
            mse_val = tree.mse(nodefun, x.T, y)   #check this x.T
        except (FloatingPointError, ZeroDivisionError, ValueError):
            return -math.inf                 # numerical failure → worst fitness

    # Some operations may finish without raising but still yield nan/inf
    if not np.isfinite(mse_val) :#or len(nodefun)>2300:
        """ if len(nodefun)>2300:
            
            print("aqui") """
        return -math.inf 
    

    
    


    """ tree_length = len(nodefun)
    penalty=5
    res= mse_val  + tree_length *penalty

    penalty_coeff = 0.001
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    global globaltime
    globaltime+=elapsed """


    #depthpenalty=nodefun.depth   # it's too computationally costly

    if numproblem==5: #useless
        div=10
    elif numproblem==7:
        div=8
        #div=10
    elif numproblem==6:
        #div=10
        div=6
    elif numproblem==3:
        div=0.8
    elif numproblem==2:  #added
        div=0.8
    else:
        div=1 


    return -mse_val-((length_penalty(len(nodefun),1000,maxy*sizey/(5000*2))) /div)  #+ penalty_coeff * (tree_length ** 2)
    

    

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
    unique_string_genomes = set()

    while len(population)<nelements:
        locel=gpTree.create_individual(10)
        #locel=gxgp.random.gxgp_random.choice(localnode)  #maybe non reimbussola
                                  #random.sample(population, k, *, counts=None)
        fitnesslocel=fitness(gpTree,locel,x,y)
        #if (fitnesslocel<np.inf):
        if (str(locel) not in unique_string_genomes):
            unique_string_genomes.add(str(locel))
            population.append(Individual(genome=locel,fitness=fitnesslocel))
        
        else:
            print("malke") 
        #population = [Individual(genome=localnode,fitness=fitness(gpTree,localnode,x,y)) for _ in range(nelements)]
    return population,unique_string_genomes

def tournament_sel_array(population,n=2): # tournament selection to decide which individual to crossover



    # Choose n elements at random from the specified row
    fitness_list=[]
    random_elements_indexes = np.random.choice(len(population), size=n, replace=True)
    for el in random_elements_indexes:
        fitness_list.append((population[el]).fitness)


    bestIndex=np.argmax(fitness_list)
    bestValue=fitness_list[bestIndex]
    
    chosen=population[random_elements_indexes[bestIndex]]

    return chosen

#find a way to limit depth
#Steady-state model


#failed variable mutrate and crossover
""" if len(history) > 5: # Check after 5 steps to have enough history
                # Calculate improvements within the last 5 fitness values
                # A "strict improvement" means `el < previou_el`.
                # Initialize improvements as 0. Each time fitness decreases, it's an improvement.
                improvements = 0
                # Iterate from the second element of the last 5 to compare with its predecessor
              
                for i in range(1, 5): # history[-5:] effectively gives us 5 elements. indices 0-4

                    if history[-5 + i] < history[-5 + i - 1]:
                        improvements += 1
                
                # ic(f"Step {current_step}: mutrate={mutrate:.2f}, crossOverRate={crossOverRate:.2f}, improvements={improvements}")

                if improvements >= 1 :
                    # Good progress: Decrease mutation to exploit, increase crossover for mixing good parts
                    mutrate = min(1, mutrate * 1.01) # Corrected typo: mut_increasement to mut_increase_factor
                    crossOverRate = max(0.05, crossOverRate * 0.99)
                    
                # Check for stagnation (e.g., less than 1 improvement in last 4 comparisons)
                elif improvements < 1:
                    # Stagnating: Increase mutation to explore, decrease crossover to preserve existing structures
                    mutrate = max(0.05, mutrate * 0.99)
                    crossOverRate = min(1, crossOverRate * 1.01)
                    
            
            # Ensure rates stay within bounds after any adjustment
            mutrate += np.random.normal(0, 0.01)
            crossOverRate += np.random.normal(0, 0.01)
            mutrate = np.clip(mutrate, 0.05, 1)
            crossOverRate = np.clip(crossOverRate, 0.05, 1)    """


def EA_Run(nstep,pop,crossOverRate,mutrate,MAX_TREE_LENGTH,gptree):
        run_until_plateaux=False

        unique_string_genomes=set()
        for el in pop:  
            localGenome=el.genome
            unique_string_genomes.add(str(localGenome))
        #pop,loc_unique_str_genome=popInizialize(gptree,nelemets,x,y)
        #unique_string_genomes.update(loc_unique_str_genome)
        best_individual = max(pop, key=lambda x: x.fitness)
        #worst_individual = max(pop, key=lambda x: x.fitness)
        lastImprouvement=-1
        originalnstep=nstep
        while lastImprouvement<originalnstep/3 and (lastImprouvement==-1 or run_until_plateaux==True):
            if lastImprouvement==-1:
                lastImprouvement=0
            print(lastImprouvement)
            print(best_individual.fitness)
            
            for el in tqdm(range(nstep)):
                lastImprouvement+=1

     
                topPoptreshold=0.24  #1500   #pop 1000 =0.16    #pop 2000  0.32
                getting_From_Top=0.8 

                """ topPoptreshold=1   #not using overpopulatin parameters

                getting_From_Top=1 """



                if(crossOverRate==1 or  np.random.rand()<crossOverRate):
                    sorted_pop = sorted(pop, key=lambda x: x.fitness,reverse=True)
                    best_pop=sorted_pop[:int(topPoptreshold*len(sorted_pop))]
                    worst_pop=sorted_pop[int(topPoptreshold*len(sorted_pop)):]
                    """ print("best:"+str(sorted_pop[0].fitness))
                
                    print("worst:"+str(worst_pop[-1].fitness)) """

                    if np.random.rand()<getting_From_Top:

                        ris1=tournament_sel_array(best_pop,2)   # uses tournament selection to select 2 parent for the crossover
                    else:
                        ris1=tournament_sel_array(worst_pop,2)   # uses tournament selection to select 2 parent for the crossover


                    if np.random.rand()<getting_From_Top:

                        ris2=tournament_sel_array(best_pop,2)   # uses tournament selection to select 2 parent for the crossover
                    else:
                        ris2=tournament_sel_array(worst_pop,2)  
                    
                    #ris2=tournament_sel_array(pop,2)
                    genome_child = gxgp.xover_swap_subtree(ris1.genome,ris2.genome)  # uses partially mapped crossover
                    #child_candidate = gxgp.xover_swap_subtree(ris1.genome, ris2.genome)
                    if len(genome_child) > MAX_TREE_LENGTH:
                        #print("generated new one because too long")
                        listonepop,_=popInizialize(gptree,1,x,y)
                        randomNewIndividual=listonepop[0]
                        genome_child = randomNewIndividual.genome
                    
                
                else:
                    genome_child=(tournament_sel_array(pop,1)).genome  #select a random individual that later can be mutated
                if len(genome_child) > 2300:
                    print("aqui")
                    
                    continue
                #child=Individual(genome=child,fitness=fitness(gptree,child,x,y))

            
                
            
                mutation_functions = [
                lambda genome: gxgp.mutation_point(genome, gptree),
                lambda genome: gxgp.mutation_hoist(genome),
                lambda genome: gxgp.mutation_permutations(genome),
                lambda genome: gxgp.mutation_collapse(genome, gptree),
                lambda genome: gxgp.mutation_delete_unirary(genome),
                lambda genome: gxgp.mutation_add_unirary(genome,gptree)
                ]
            
                if( np.random.rand()<mutrate):  # has a chance of mutating the child using a mutation
                    mutation_fn = np.random.choice(mutation_functions)
                
                    genome_child = mutation_fn(genome_child)
                    if len(genome_child) > 2300:
                        print("si")
                        continue
                    #child = Individual(genome=mutedgenome, fitness=fitness(gptree, mutedgenome, x, y))
                    

                
                
                # ic(fitness(gptree,child.genome,x,y))
                if (str(genome_child) not in unique_string_genomes):
                    unique_string_genomes.add(str(genome_child))
                    child=Individual(genome=genome_child, fitness=fitness(gptree, genome_child, x, y))
                    pop.append(child)  # add child to population
                    newIndividual=child

                
                else:  # if it  isn't new
                    
                    listonepop,_=popInizialize(gptree,1,x,y)
                    randomNewIndividual=listonepop[0]
                    while str(randomNewIndividual.genome) in unique_string_genomes:
                        listonepop,_=popInizialize(gptree,1,x,y)
                        randomNewIndividual=listonepop[0]
                    
                    unique_string_genomes.add(str(randomNewIndividual.genome))
                    #child=Individual(genome=randomNewGenome, fitness=fitness(gptree, randomNewGenome, x, y))
                    pop.append(randomNewIndividual)
                    newIndividual=randomNewIndividual

                


                if(newIndividual.fitness>best_individual.fitness):
                    best_individual=newIndividual
                    lastImprouvement=0
                

            

                

                #lowest_fitness_individual = min(pop, key=lambda x: x.fitness)
                localbestfit=best_individual.fitness

                
                
                #ic(lowest_fitness_individual.fitness)
                history.append(localbestfit)
                
                worst_fit_ind=(min(pop, key=lambda ind: ind.fitness))    #todo make negativ efitness and use a minimizer 
                #ic(maxfit_ind.fitness)
                pop.remove(worst_fit_ind) # delete from population the wrost individual
                unique_string_genomes.remove(str(worst_fit_ind.genome))
            
        



        
        


        sorted_pop = sorted(pop, key=lambda x: x.fitness,reverse=True)

        # Get the best and second-best individuals
        lowest_fitness_individual = sorted_pop[0]
        formula=lowest_fitness_individual.genome.to_np_formula()
        print(formula)
        print("fitness:")
        print(lowest_fitness_individual.fitness)
        second_lowest_fitness_individual = sorted_pop[1]
        formula=second_lowest_fitness_individual.genome.to_np_formula()
        
        return [lowest_fitness_individual,second_lowest_fitness_individual]
        




def EAlgoithm(nproblem,gptree,x,y):  #use elitism try to preserve best ones
    """ problem = np.load('/home/adri/universita/magistrale/5_anno_1_sem/computational_intelligence/project_work/CI2024_project-work/data/problem_{}.npz'.format(nproblem))
    x = problem['x']
    y = problem['y'] """
    

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
    final_run_long=True
    MAX_TREE_LENGTH=500
    mutrate=0.055  #0.1#
    minmutrate=0.05
    mincrossover=0.05                              
    #mutrate=1
    nrestarts=6  # was 10 for probl 4 #was 5
    nelemets=1500  #was 2000    #1000#15   #400   #was 200
    crossOverRate=1#0.9
    mut_increasement=1.2
    
    nstep=50000#15000#50000#15000 #12000 #5000#10000#1000 #100 #2000   #usa 4000
    """  lowest_fitness_individual = min(pop, key=lambda x: x.fitness)
    ic(lowest_fitness_individual)
    ic(str(lowest_fitness_individual.genome)) """
    #ic(lowest_fitness_individual.fitness)
    #unique_string_genomes = set()
    pop0,_=popInizialize(gptree,1,x,y)
    
    lowest_fitness_individual=pop0[0]
    best_individuals_each_restart=[]
    for _ in range(nrestarts):
        pop,_=popInizialize(gptree,nelemets,x,y)
        
        top_2_individual=EA_Run(nstep,pop,crossOverRate,mutrate,MAX_TREE_LENGTH,gptree)
        best_individuals_each_restart.extend(top_2_individual)

    

    if final_run_long==True:

    
        unique_string_genomes=set()
        pop=[]
        for individual in best_individuals_each_restart:
            genome_child=individual.genome


            if (str(genome_child) not in unique_string_genomes):
                        unique_string_genomes.add(str(genome_child))
                        child=Individual(genome=genome_child, fitness=fitness(gptree, genome_child, x, y))
                        pop.append(child)  # add child to population

            
            mutation_functions = [
                lambda genome: gxgp.mutation_point(genome, gptree),
                lambda genome: gxgp.mutation_hoist(genome),
                lambda genome: gxgp.mutation_permutations(genome),
                lambda genome: gxgp.mutation_collapse(genome, gptree),
                lambda genome: gxgp.mutation_delete_unirary(genome),
                lambda genome: gxgp.mutation_add_unirary(genome,gptree)
                ]

            for i in range(int(0.4*nelemets/len(best_individuals_each_restart))):
                mutation_fn = np.random.choice(mutation_functions)
                mutated_genome = mutation_fn(genome_child)

                if str(mutated_genome) not in unique_string_genomes:
                    unique_string_genomes.add(str(mutated_genome))
                    mutated_child = Individual(genome=mutated_genome, fitness=fitness(gptree, mutated_genome, x, y))
                    pop.append(mutated_child)
            
        for _ in range(int(0.15*nelemets)):
                p1=tournament_sel_array(best_individuals_each_restart,2)   # uses tournament selection to select 2 parent for the crossover
                p2=tournament_sel_array(best_individuals_each_restart,2)

                crossed = gxgp.xover_swap_subtree(p1.genome, p2.genome)
                if str(crossed) not in unique_string_genomes:
                    unique_string_genomes.add(str(crossed))
                    child = Individual(genome=crossed, fitness=fitness(gptree, crossed, x, y))
                    pop.append(child)

            

        missing_individuals=nelemets-len(pop)
        poploc,_=popInizialize(gptree,missing_individuals,x,y)
        pop.extend(poploc)
        if numproblem==2:
            nstep=nstep*10
        top_2_individual=EA_Run(nstep,pop,crossOverRate,mutrate,MAX_TREE_LENGTH,gptree)
        best_individuals_each_restart.extend(top_2_individual)



                
    """ else:  # if it  isn't new
                    
                    listonepop,_=popInizialize(gptree,1,x,y)
                    randomNewIndividual=listonepop[0]
                    while str(randomNewIndividual.genome) in unique_string_genomes:
                        listonepop,_=popInizialize(gptree,1,x,y)
                        randomNewIndividual=listonepop[0]
                    
                    unique_string_genomes.add(str(randomNewIndividual.genome))
                    #child=Individual(genome=randomNewGenome, fitness=fitness(gptree, randomNewGenome, x, y))
                    pop.append(randomNewIndividual) """

        

    #best_one=min(pop,key=lambda el: error(gptree,el.genome,x,y))
    best_one=min(best_individuals_each_restart,key=lambda el: error(gptree,el.genome,x,y))
    formula=best_one.genome.to_np_formula()
    #ic(best_one.fitness)

    


    return formula,error(gptree,best_one.genome,x,y)
    
#2 problem 
#3197567994.386814
#3780431286.080886
#3241102705.3421373   mine
#9809284786418.186  #using no len penalty
#13597638488567.383
#26585319369516.727
#19040148669684.434  ###others solution 
#3281516793.5310025  #others sol
#1.904e+15

def normalization(y,range=(0,5)):
    y_min = np.min(y)
    y_max = np.max(y)
   

    offset=y_min
    mulCoefficient=(range[1] -range[0])/(y_max-y_min)

    yscaled=(y-offset)*mulCoefficient+range[0]

    return yscaled,mulCoefficient,offset,range[0]



numproblem=2
scaling=False
problem = np.load('/home/adri/universita/magistrale/5_anno_1_sem/computational_intelligence/project_work/CI2024_project-work/data/problem_{}.npz'.format(numproblem))
x = problem['x']  #3 righe 5000 colonne 

y=problem['y']
#fixxx
""" def trying_f5_rescaled(x:np.ndarray,offset,mul,y:np.ndarray):
     yset=np.add(np.exp2(np.tanh(np.square(np.hypot(np.sin(np.hypot(np.hypot(np.pi, x[1]), np.e)), np.tanh(np.add(np.tanh(x[1]), np.sin(x[0]))))))), np.pi)
     yreset=np.add(np.divide(np.subtract(yset , np.min(x)),  mul) , offset)

     ris=sum((a - b) ** 2 for a, b in zip(y, yreset)) / len(y)
     print("real result:")
     print(ris)
     return yreset """

if(scaling==True):

    y_scaled,mulCof,offset,_=normalization(y)
    #trying_f5_rescaled(x,offset,mulCof,y)
    y=y_scaled 
sizey=np.size(y)
maxy=np.max(y)

""" ic(y[2])  #5000 colonne
ic(x[:,2])  """# i think x[2] is never used   , 3 righe , 1 colonna, è il secondo elemento


""" x_T = x.T

# Reshape y to (5000, 1)
y_col = y.reshape(-1, 1)

# Concatenate x and y horizontally: shape (5000, 4)
xy = np.hstack((x_T, y_col))

# Save to file
np.savetxt(f'problem_{numproblem}.txt', xy, delimiter=' ', fmt='%.6f') """
dag = gxgp.DagGP(   #problems with ldex , it's unsupported for some types   np.ldexp,
    #safe
    #operators=[np.pow],
    #operators =[np.mod,np.arctan,np.sinc,np.sqrt,np.cbrt,np.add,np.negative,np.multiply,np.sin,np.tanh,np.reciprocal,np.exp,np.exp2,np.pow,np.sinh,np.round,np.cosh,np.hypot,np.i0,np.absolute,np.square,np.log1p,np.log2,np.log10,np.log], 
    #operators=[np.multiply,np.cbrt,np.sqrt,np.negative,np.add,np.subtract,np.sin,np.cos,np.cosh,np.sinc,np.log10,np.log2,np.log1p,np.round,np.hypot,np.mod,np.log,np.arctan,np.ceil],
    #operators=[np.mod,np.arctan,np.sinc,np.sqrt,np.cbrt,np.add,np.negative,np.multiply,np.sin,np.tanh,np.reciprocal,np.exp,np.exp2,np.pow,np.sinh,np.round,np.cosh,np.hypot,np.i0,np.absolute,np.square,np.log1p,np.log2,np.log10,np.log],  #np.acos], #ignore_op#np.round],  #np.round #np.pow, #np.ldexp],
    operators=[np.mod,np.arctan,np.maximum,np.sqrt,np.cbrt,np.add,np.negative,np.multiply,np.sin,np.tanh,np.reciprocal,np.exp,np.exp2,np.pow,np.sinh,np.round,np.cosh,np.hypot,np.i0,np.absolute,np.square,np.log1p,np.log2,np.log10,np.log],  #changed for p2
    #operators=[np.log1p,np.exp2,np.i0,np.sin],
    variables=x.shape[0],
    constants=[np.pi,np.e,np.euler_gamma],#,  -1.234,1.5,9.9343,-9.0345,2.345,-6.3454],
   # constants=[np.pi,np.e,np.euler_gamma],#[-1,0,1,np.pi,np.e,np.euler_gamma],#np.linspace(-2, 2, 500),
)

#ic(np.var(y))
formula,fit=EAlgoithm(numproblem,dag,x,y)
print("THe formula is :")
print(formula)
ygen = eval(formula, {"np": np, "x": x})
if (ygen.size==1):
    ygen = ygen * np.ones(np.size(y))  #checkl

ris= np.mean((y.astype(np.float64) - ygen.astype(np.float64)) ** 2)#sum((a - b) ** 2 for a, b in zip(ygen, y)) / len(y) 
#ic(ris)

print("min_mse no scaling")
print(fit)
if ris==fit:
    print("good")
else:
    print("la differenxza")
    print(ris)
    print(fit)
    print("wrong") 

if scaling==True:
    denormalizedFromula=f"np.add(np.divide({formula}, {mulCof.item()}), {offset.item()})"
    print(denormalizedFromula)
    y_denormalized=eval(denormalizedFromula, {"np": np, "x": x})
    if (y_denormalized.size==1):
        y_denormalized = y_denormalized *np.ones(np.size(y))  #checkl
    y=problem['y']
    ris= np.mean((y.astype(np.float64) - y_denormalized.astype(np.float64)) ** 2)#sum((a - b) ** 2 for a, b in zip(ygen, y)) / len(y) 

    
    ris= sum((a - b) ** 2 for a, b in zip(y_denormalized, y)) / len(y) 

    print("min_mse using scaling: ")
    print(ris) 




results_file = "gp_results_log.txt"
with open(results_file, "a") as f:
    f.write("="*60 + "\n")

    f.write(f"Problem Number: {numproblem}\n")
    f.write(f"Scaling Used: {scaling}\n")
   
    
    if scaling:
        f.write(f"Formula: {denormalizedFromula}\n")
    else:
        f.write(f"Formula: {formula}\n")

    
    f.write(f"MSE: {ris}\n")
    f.write("="*60 + "\n\n")


#ic(globaltime)
plt.figure(figsize=(14, 8))
plt.plot(
            range(len(history)),
            list(accumulate(history, max)),
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