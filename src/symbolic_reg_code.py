import numpy as np

from gxgp import Node
import gxgp
from gxgp import *
from tqdm import tqdm
from dataclasses import dataclass
from matplotlib import pyplot as plt
from itertools import accumulate





i=1

history=[]

import math


@dataclass
class Individual:  #internally generates a init for this dataclass.
    genome: Node
    fitness: float = None



def length_penalty(length, threshold=400, scale=1e6):
    if length <= threshold:

        return   0.001*length*scale#scale * 0.001 * length**2
    else:
        base = scale * 0.001 * threshold**2
        blowup = np.exp((length - threshold) / 10) * scale
        return base + blowup

def fitness(tree,nodefun:Node,x, y):
    #start_time = time.perf_counter()


    with np.errstate(all="raise"):          # only for the code inside
        try:
            mse_val = tree.mse(nodefun, x.T, y)   
        except (FloatingPointError, ZeroDivisionError, ValueError):
            return -math.inf                 # error= worst fitness

   
    if not np.isfinite(mse_val) :  #infinite result or nan result  = worst fitness 
        return -math.inf 




  
    if numproblem==3: #increase penalty for lengthy individuals
        div=0.8
    elif numproblem==2:  #increase penalty for lengthy individuals
        div=0.8
    else:
        div=1 


    return -mse_val-((length_penalty(len(nodefun),1000,maxy*sizey/(5000*2))) /div)  #+ penalty_coeff * (tree_length ** 2)




def error(tree,nodefun:Node,x, y):


    with np.errstate(all="raise"):          # only for the code inside
        try:
            mse_val = tree.mse(nodefun, x.T, y)
        except (FloatingPointError, ZeroDivisionError, ValueError):
            return math.inf                 # ncan't be compute 

   
    if not np.isfinite(mse_val):
        return math.inf 
    
    return mse_val
    

def popInizialize(gpTree,nelements,x,y):   
    population=[] 
    unique_string_genomes = set()

    while len(population)<nelements:
        locel=gpTree.create_individual(10)
 
        fitnesslocel=fitness(gpTree,locel,x,y)
        if (str(locel) not in unique_string_genomes):
            unique_string_genomes.add(str(locel))
            population.append(Individual(genome=locel,fitness=fitnesslocel))

    return population

def tournament_sel_array(population,n=2): # tournament selection to decide which individual to crossover


    fitness_list=[]
    random_elements_indexes = np.random.choice(len(population), size=n, replace=True)
    for el in random_elements_indexes:
        fitness_list.append((population[el]).fitness)


    bestIndex=np.argmax(fitness_list)
    
    chosen=population[random_elements_indexes[bestIndex]]

    return chosen




def EA_Run(nstep,pop,crossOverRate,mutrate,MAX_TREE_LENGTH,gptree,extra_el_sel=0):  #tournament selection with size 2 , in all runs except in the last one when it's size 4 (increased selective pressure )
        run_until_plateaux=False

        unique_string_genomes=set()
        for el in pop:  
            localGenome=el.genome
            unique_string_genomes.add(str(localGenome))
        
        best_individual = max(pop, key=lambda x: x.fitness)
        lastImprouvement=-1
        originalnstep=nstep
        while lastImprouvement<originalnstep/3 and (lastImprouvement==-1 or run_until_plateaux==True):  #if run_until_plateaux is true , it runs as long as a plateux is not detected 
            if lastImprouvement==-1:
                lastImprouvement=0
            print(lastImprouvement)
            print(best_individual.fitness)
            
            for el in tqdm(range(nstep)):
                lastImprouvement+=1

     
                topPoptreshold=0.24  #1500   #pop 1000 =0.16    #pop 2000  0.32
                getting_From_Top=0.8 




                if(crossOverRate==1 or  np.random.rand()<crossOverRate):
                    sorted_pop = sorted(pop, key=lambda x: x.fitness,reverse=True)
                    best_pop=sorted_pop[:int(topPoptreshold*len(sorted_pop))]
                    worst_pop=sorted_pop[int(topPoptreshold*len(sorted_pop)):]

                    if np.random.rand()<getting_From_Top:

                        ris1=tournament_sel_array(best_pop,2+extra_el_sel)   # uses tournament selection to select 2 parent for the crossover
                    else:
                        ris1=tournament_sel_array(worst_pop,2+extra_el_sel)   # uses tournament selection to select 2 parent for the crossover


                    if np.random.rand()<getting_From_Top:

                        ris2=tournament_sel_array(best_pop,2+extra_el_sel)   # uses tournament selection to select 2 parent for the crossover
                    else:
                        ris2=tournament_sel_array(worst_pop,2+extra_el_sel)  
                    
                    genome_child = gxgp.xover_swap_subtree(ris1.genome,ris2.genome)  
                    if len(genome_child) > MAX_TREE_LENGTH:
                        #generated new one because old one was too long
                        listonepop=popInizialize(gptree,1,x,y)
                        randomNewIndividual=listonepop[0]
                        genome_child = randomNewIndividual.genome
                    
                
                else:
                    genome_child=(tournament_sel_array(pop,1)).genome  #select a random individual that later can be mutated
                if len(genome_child) > 2300:
                    print("aqui")
                    
                    continue
       
            
                mutation_functions = [
                lambda genome: gxgp.mutation_point(genome, gptree),
                lambda genome: gxgp.mutation_hoist(genome),
                lambda genome: gxgp.mutation_permutations(genome),
                lambda genome: gxgp.mutation_collapse(genome, gptree),
                lambda genome: gxgp.mutation_delete_unary(genome),
                lambda genome: gxgp.mutation_add_unary(genome,gptree)
                ]
            
                if( np.random.rand()<mutrate):  # has a chance of mutating the child using a mutation
                    mutation_fn = np.random.choice(mutation_functions)
                    if len(genome_child) > 2300:
                        print("si")
                        continue
                
                    genome_child = mutation_fn(genome_child)
                    
             
                if (str(genome_child) not in unique_string_genomes):
                    unique_string_genomes.add(str(genome_child))
                    child=Individual(genome=genome_child, fitness=fitness(gptree, genome_child, x, y))
                    pop.append(child)  # add child to population
                    newIndividual=child

                
                else:  # if it  isn't new
                    
                    listonepop=popInizialize(gptree,1,x,y)
                    randomNewIndividual=listonepop[0]
                    while str(randomNewIndividual.genome) in unique_string_genomes:
                        listonepop=popInizialize(gptree,1,x,y)
                        randomNewIndividual=listonepop[0]
                    
                    unique_string_genomes.add(str(randomNewIndividual.genome))
                    pop.append(randomNewIndividual)
                    newIndividual=randomNewIndividual

                


                if(newIndividual.fitness>best_individual.fitness):
                    best_individual=newIndividual
                    lastImprouvement=0
                
                localbestfit=best_individual.fitness

                history.append(localbestfit)
                
                worst_fit_ind=(min(pop, key=lambda ind: ind.fitness))    
                pop.remove(worst_fit_ind) # delete from population the wrost individual
                unique_string_genomes.discard(str(worst_fit_ind.genome))
            
        



        
        


        sorted_pop = sorted(pop, key=lambda x: x.fitness,reverse=True)  #reverse true means desc order(first the one with higher fitness)

        # Get the best and second individuals
        lowest_fitness_individual = sorted_pop[0]
        formula=lowest_fitness_individual.genome.to_np_formula()
        print(formula)
        print("fitness:")
        print(lowest_fitness_individual.fitness)
        second_lowest_fitness_individual = sorted_pop[1]
        formula=second_lowest_fitness_individual.genome.to_np_formula()
        
        return [lowest_fitness_individual,second_lowest_fitness_individual]
        




def EAlgoithm(gptree,x,y):  
    
    final_run_long=True
    MAX_TREE_LENGTH=500
    mutrate=0.055 
    nrestarts=6   
    nelemets=1500  
    crossOverRate=1     #trying dynamically adjusting crossover rate and mutation rate did't improved the solutions
    
    nstep=50000

    best_individuals_each_restart=[]
    for _ in range(nrestarts):
        pop=popInizialize(gptree,nelemets,x,y)  #generates nelemets individuals each restart 
        
        top_2_individual=EA_Run(nstep,pop,crossOverRate,mutrate,MAX_TREE_LENGTH,gptree) #return  an array with the top2 individuals 
        best_individuals_each_restart.extend(top_2_individual)  #adds those 2 individuals to the pool of all previus generated champions for each restart 

    

    if final_run_long==True:  #if it's true , does an other run using previus champions, their mutation , their crossover a

    
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
                lambda genome: gxgp.mutation_delete_unary(genome),
                lambda genome: gxgp.mutation_add_unary(genome,gptree)
                ]

            for i in range(int(0.4*nelemets/len(best_individuals_each_restart))):
                mutation_fn = np.random.choice(mutation_functions) #select a random mutation and applies it 
                mutated_genome = mutation_fn(genome_child)

                if str(mutated_genome) not in unique_string_genomes:
                    unique_string_genomes.add(str(mutated_genome))
                    mutated_child = Individual(genome=mutated_genome, fitness=fitness(gptree, mutated_genome, x, y))
                    pop.append(mutated_child)
        nlong=0
        for _ in range(int(0.15*nelemets)):
                p1=tournament_sel_array(best_individuals_each_restart,2)   # uses tournament selection to select 2 parent for the crossover
                p2=tournament_sel_array(best_individuals_each_restart,2)

                crossed = gxgp.xover_swap_subtree(p1.genome, p2.genome)
            
                if str(crossed) not in unique_string_genomes and len(crossed) <= MAX_TREE_LENGTH :
                    nlong+=1
                    unique_string_genomes.add(str(crossed))
                    child = Individual(genome=crossed, fitness=fitness(gptree, crossed, x, y))
                    pop.append(child)

            
        print(nlong)
        missing_individuals=nelemets-len(pop)
        poploc=popInizialize(gptree,missing_individuals,x,y)
        pop.extend(poploc)
        if numproblem==2:  #do more step in the final run
            nstep=nstep*10
        elif numproblem==3:  #it worked
            nstep=nstep*10
        elif numproblem==7: #it worked
            nstep=nstep*10
        
  
        
        top_2_individual=EA_Run(nstep,pop,crossOverRate,mutrate,MAX_TREE_LENGTH,gptree,2) #last run puts more selective pressure when doing parent selection (tournament selection size 4)
        best_individuals_each_restart.extend(top_2_individual)


    best_one=min(best_individuals_each_restart,key=lambda el: error(gptree,el.genome,x,y))
    formula=best_one.genome.to_np_formula()

    


    return formula,error(gptree,best_one.genome,x,y)
    


def normalization(y,range=(0,5)):  #scales the y into the passed range doing min-max scaling
    y_min = np.min(y)
    y_max = np.max(y)
   

    offset=y_min
    mulCoefficient=(range[1] -range[0])/(y_max-y_min)

    yscaled=(y-offset)*mulCoefficient+range[0]

    return yscaled,mulCoefficient,offset,range[0]



numproblem=7
scaling=False
problem = np.load('../data/problem_{}.npz'.format(numproblem)) #depends on where you are running the program , if in the project directory only one dot at the start ,, if in the src directory use two dots
x = problem['x']  #3 righe 5000 colonne 

y=problem['y']


if(scaling==True):

    y_scaled,mulCof,offset,_=normalization(y)
    y=y_scaled 
sizey=np.size(y)
maxy=np.max(y)


if numproblem==2:  #there is maximum function instead of sinc one (we have high values of y here , i belive it's not useful sinc )
            operators=[np.mod,np.arctan,np.maximum,np.sqrt,np.cbrt,np.add,np.negative,np.multiply,np.sin,np.tanh,np.reciprocal,np.exp,np.exp2,np.pow,np.sinh,np.round,np.cosh,np.i0,np.hypot,np.absolute,np.square,np.log1p,np.log2,np.log10,np.log]
else:
        operators=[np.mod,np.arctan,np.sinc,np.sqrt,np.cbrt,np.add,np.negative,np.multiply,np.sin,np.tanh,np.reciprocal,np.exp,np.exp2,np.pow,np.sinh,np.round,np.cosh,np.hypot,np.i0,np.absolute,np.square,np.log1p,np.log2,np.log10,np.log]  #np.acos], #ignore_op#np.round],  #np.round #np.pow, #np.ldexp],


dag = gxgp.DagGP( 
    operators=operators, 
    variables=x.shape[0],
    constants=[np.pi,np.e,np.euler_gamma],
)

formula,fit=EAlgoithm(dag,x,y)
print("THe formula is :")
print(formula)
ygen = eval(formula, {"np": np, "x": x})
if (ygen.size==1):
    ygen = ygen * np.ones(np.size(y))  

ris= np.mean((y.astype(np.float64) - ygen.astype(np.float64)) ** 2)

print("min_mse no scaling")
print(fit)
if ris==fit:
    print("good")
else:  #there is an error if they don't match 
    print("wrong") 

if scaling==True:
    denormalizedFromula=f"np.add(np.divide({formula}, {mulCof.item()}), {offset.item()})"  #denormalizing by inverting the previus minamx scaling directly in the string formula 
    print(denormalizedFromula)
    y_denormalized=eval(denormalizedFromula, {"np": np, "x": x})
    if (y_denormalized.size==1):
        y_denormalized = y_denormalized *np.ones(np.size(y)) 
    y=problem['y']
    ris= np.mean((y.astype(np.float64) - y_denormalized.astype(np.float64)) ** 2)
    
    ris= sum((a - b) ** 2 for a, b in zip(y_denormalized, y)) / len(y) 

    print("min_mse using scaling: ")
    print(ris) 




results_file = "gp_results_log.txt"   #writes all result in a log file
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



plt.figure(figsize=(14, 8))
plt.plot(
            range(len(history)),
            list(accumulate(history, max)),
            color="red",
        )
_ = plt.scatter(range(len(history)), history, marker=".")
plt.show() 
