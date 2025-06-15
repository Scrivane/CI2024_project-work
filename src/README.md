# Project
I worked on this algorithm alone.
The objective of this program is to discover mathematical functions that best minimize the Mean Squared Error (MSE) on the given datasets. The algorithm is based on a modern Genetic Algorithm (GA).

## Genotype representation  


Each individual in the population has an expression tree as its genotype. Each node within a tree can be a constant, a variable, or a mathematical function. The algorithm can evolve formulas by choosing from many NumPy functions (hoping to always find a possible good solution, even if that means running for more epochs), but only a few constants (3 constants: pi, Euler's number, and the Euler–Mascheroni constant). The available variables are provided by each problem.

Each node serves as the root of its respective subtrees, and each node has a list of its subtrees and its successors. Each node can have a maximum of 2 successors

## Fitness 

Each individual has a fitness field. 
The fitness combines the Mean Squared Error (MSE) with a parsimony pressure term (a penalty for lengthy individuals). This penalty is scaled based on the maximum values observed in the target (y) data and the number of data points in the problem. The maximum length of an individual's genome is limited to 500 nodes (including operations, constants, and variables) to avoid excessive bloat, which would have caused Python errors.


## Optimized Individual Evaluation

The program leverages the gxgp library by Professor Squillero, extending its capabilities with custom functions and deleting unused functions. A significant performance enhancement involves a custom evaluation function, which converts a tree into its equivalent NumPy formula string. This string is then executed using Python's eval() function, passing the input x (a NumPy array) as a parameter. This approach allows for faster evaluations.


## Evolutionary set-up 

The algorithm employs a steady-state evolutionary model; during each epoch, a new individual replaces the worst-fitting one.

Parent selection for crossover is performed using tournament selection (size 2, with low selective pressure to maintain diversity). Crossover is always applied. To increase the selective pressure during parent selection in the large population (1500 individuals), Over-selection is utilized. Specifically, 80% of parent selections are made exclusively from the top 24% of individuals in the population. This helps promote faster convergence while maintaining enough diversity from the rest of the population.

The crossover function used is xover_swap_subtree, which was already present in the gxgp library. It selects a random subtree from parent one and replaces it with a random subtree from parent two.

The resulting offspring may mutate at a low rate. Here are the possible mutation functions that can be employed:

* Mutation Point: Alters a node; if it's a leaf (constant or variable), it's replaced by another constant or variable. If it's an internal node, it's replaced by another operand with the same arity.
* Mutation Hoist: Selects a random subtree and makes it the new root of the tree, effectively "cutting" the tree.
* Mutation Permutations: Changes the order of operands for binary function nodes, avoiding commutative functions.
* Mutation Collapse: Replaces an entire subtree with a single constant or variable.
* Mutation Delete Unary: Removes a unary function node, promoting its child directly to its parent's position.
* Mutation Add Unary: Inserts a new unary function node at a random point in the tree, adding depth.

Survivor selection is deterministic. The worst-fitting individual in the population is purged in each epoch. To maintain genetic diversity and prevent premature convergence, the algorithm explicitly avoids adding individuals with formulas identical to those already present in the population.


## Refinement Strategies

For some problems, specific optimization strategies are applied to get better results:

* Restart Mechanism: The algorithm performs 6 independent restarts. Each restart initializes a new population and runs the GA until a maximum number of steps is reached.
* Final Refinement Run: After the restarts, a final refinement run is done. The starting population for this run contains the best two individuals from each previous restart, their mutated versions, and individuals generated through crossover between them, alongside a proportion of randomly initialized individuals. This approach aims to refine the "best individuals" discovered across multiple independent searches.
* "Run Until Plateau" Feature: Instead of a fixed number of steps, the algorithm continues as long as there are improvements in fitness over the last nstep/3 epochs, detecting when the search has stagnated in a local optimum. This is a computationally intensive strategy, so it's not done for every program.
* Long Refinement: The last refinement run has more steps. The idea is that the initial 6 restarts efficiently locate promising local minima, and the final run can perform a more extensive search.



