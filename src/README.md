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
* Final Refinement Run: After the restarts, a final refinement run is done. The starting population for this run contains the best two individuals from each previous restart, their mutated versions, and individuals generated through crossover between them, alongside a proportion of randomly initialized individuals. This approach aims to refine the "best individuals" discovered across multiple independent searches. During this run , it's increased the selective pressure using tournament selection of size 4 ( instead of 2 as previous runs)
* "Run Until Plateau" Feature: Instead of a fixed number of steps, the algorithm continues as long as there are improvements in fitness over the last nstep/3 epochs, detecting when the search has stagnated in a local optimum. This is a computationally intensive strategy, so it's not done for every program.
* Long Refinement: The Final Refinement Run has x10 steps. The idea is that the initial 6 restarts efficiently locate promising local minimums, and the final run can perform a more extensive search.
* Rescaling : For problem 5 , the objective values of y were with a really low scale , so they got rescaled into the range [0,5] evolution began.After the formula was evolved, the y values were scaled back to their original range. This mechanism significantly improved the results.


# Results 
| Problem Number | Nsteps | Nrestarts | Rescaling | Run Until Plateau | Final Refinement Run | Long Refinement | MSE | Formula |
|---|---|---|---|---|---|---|---|---|
| 1 | 10000 | 1 | No | No | No | No | 0 | np.sin(x[0]) |
| 2 | 50000 | 6 | No | No | Yes | Yes | 1548510021431.2053 | np.multiply(np.tanh(np.add(np.multiply(x[0], np.log2(np.absolute(np.exp(np.log1p(np.e))))), np.add(x[1], x[2]))), np.multiply(np.tanh(np.maximum(np.maximum(np.sin(np.maximum(np.add(np.log1p(np.maximum(np.sin(np.arctan(np.sin(np.pi))), x[0])), np.arctan(x[0])), x[0])), np.add(np.sin(np.exp(np.log1p(np.exp(np.maximum(np.arctan(np.add(np.add(np.maximum(np.pi, np.arctan(np.arctan(np.sin(np.arctan(np.add(x[2], np.log1p(np.sin(np.sin(np.add(np.multiply(x[0], np.log2(np.absolute(np.exp(np.log1p(np.e))))), np.add(x[1], x[2]))))))))))), x[2]), x[1])), np.sin(x[0])))))), np.log1p(np.tanh(np.maximum(np.tanh(np.log1p(np.tanh(np.maximum(np.maximum(np.tanh(np.add(np.multiply(x[0], np.log2(np.absolute(np.add(np.maximum(np.add(np.e, x[2]), np.arctan(np.add(x[2], x[0]))), x[1])))), np.add(np.exp2(np.sin(np.exp2(np.multiply(np.maximum(np.add(np.add(np.maximum(np.add(np.sin(np.pi), x[2]), np.add(np.sin(np.sin(x[0])), x[1])), np.sin(x[0])), x[1]), np.add(np.log1p(np.e), np.tanh(np.pi))), np.maximum(x[2], np.arctan(np.exp2(np.sin(np.maximum(x[0], np.sin(x[0])))))))))), np.maximum(np.log1p(np.pi), np.sin(np.log1p(np.e)))))), np.add(np.log1p(np.sin(np.arctan(np.sin(np.add(np.sin(np.arctan(np.add(x[2], np.log1p(np.sin(np.log1p(np.tanh(np.sin(np.arctan(np.add(x[1], x[1])))))))))), x[0]))))), np.add(x[1], np.arctan(np.sin(np.exp(np.log1p(np.e))))))), np.add(np.sin(np.add(np.multiply(x[0], np.log2(np.absolute(np.exp(np.log1p(np.e))))), np.add(x[1], x[2]))), np.arctan(np.add(np.add(np.maximum(np.add(np.maximum(np.pi, x[0]), x[2]), np.add(np.tanh(x[2]), np.arctan(np.sin(np.sin(x[0]))))), x[1]), np.sin(np.arctan(np.add(np.multiply(x[0], np.log2(np.absolute(np.exp(np.log1p(np.e))))), np.add(x[1], x[2]))))))))))), np.add(np.sin(x[0]), np.arctan(np.add(np.add(np.maximum(np.add(np.maximum(np.pi, x[0]), x[2]), np.add(np.tanh(np.sin(np.arctan(np.sin(x[0])))), x[2])), x[1]), np.sin(np.exp(np.arctan(np.add(x[2], x[1])))))))))))), np.add(np.log1p(np.tanh(np.log1p(np.tanh(np.tanh(np.log1p(np.tanh(np.add(np.maximum(np.add(np.maximum(np.pi, x[0]), x[2]), np.add(np.sin(x[0]), x[2])), x[1])))))))), np.maximum(np.sinh(np.add(np.tanh(np.add(np.tanh(np.sin(np.exp(np.log1p(np.exp(np.maximum(np.arctan(np.maximum(np.arctan(np.arctan(np.arctan(np.sin(np.exp(np.log1p(np.e)))))), np.sin(np.maximum(np.sin(x[0]), np.maximum(np.add(x[0], np.log2(np.absolute(np.add(np.maximum(np.add(np.maximum(np.pi, np.maximum(np.sin(np.arctan(np.sin(np.add(np.sin(np.arctan(np.add(x[2], np.log1p(np.sin(np.log1p(np.tanh(np.pi))))))), x[0])))), x[0])), x[2]), x[0]), x[1])))), x[0]))))), np.sin(np.maximum(np.arctan(np.log1p(np.tanh(np.sin(np.arctan(np.negative(np.cosh(np.pi))))))), np.sin(np.maximum(np.sin(x[0]), np.maximum(np.add(x[0], np.log2(np.absolute(np.add(np.maximum(np.add(np.maximum(np.arctan(np.maximum(x[2], np.arctan(np.exp2(x[2])))), x[1]), x[2]), np.add(np.sin(np.sin(np.arctan(np.sin(np.arctan(np.add(x[2], np.log1p(np.sin(np.sin(np.exp2(np.arctan(np.add(x[0], x[1])))))))))))), x[0])), x[1])))), x[0]))))))))))), np.sin(np.exp(np.arctan(np.add(x[2], x[1])))))), np.sin(np.maximum(np.add(np.log1p(np.sin(np.sin(np.log1p(np.sin(np.sin(np.arctan(np.sin(np.add(np.sin(np.arctan(np.add(x[2], np.log1p(np.sin(np.log1p(np.tanh(np.sin(np.arctan(np.add(x[1], x[1])))))))))), x[0]))))))))), np.arctan(np.arctan(np.sin(np.arctan(np.add(x[2], x[1])))))), np.sin(np.maximum(np.maximum(np.exp(np.log1p(np.exp(np.maximum(np.sin(np.maximum(np.exp(np.log1p(np.maximum(np.sin(np.arctan(np.add(x[0], np.log2(np.absolute(np.add(np.maximum(np.add(np.maximum(np.pi, x[0]), x[2]), np.add(np.sin(x[0]), x[2])), x[1])))))), x[0]))), np.arctan(x[0]))), np.arctan(np.sin(np.arctan(np.add(x[2], x[1])))))))), np.add(np.exp2(np.sin(x[0])), np.exp2(np.tanh(np.sin(np.exp(np.maximum(np.arctan(np.add(np.add(np.maximum(np.pi, np.exp2(np.arctan(np.add(x[2], x[1])))), x[2]), x[1])), np.sin(np.sin(x[0]))))))))), np.tanh(x[2]))))))), np.sin(np.maximum(np.add(np.log1p(np.maximum(np.sin(np.arctan(np.sin(np.add(np.sin(np.arctan(np.add(x[2], np.log1p(np.sin(np.log1p(np.tanh(np.sin(np.arctan(np.add(x[1], x[1])))))))))), x[0])))), np.multiply(x[0], np.log2(np.absolute(np.exp(np.log1p(np.exp(np.log1p(np.e))))))))), np.arctan(np.add(x[2], x[1]))), x[0])))))), np.sinh(np.hypot(np.negative(np.cosh(np.pi)), np.negative(np.cosh(np.pi)))))) |
| 3 | 50000 | 6 | No | No | Yes | Yes | 0.4122990064737059 | np.add(np.add(np.add(np.add(np.add(np.exp2(np.sqrt(np.add(np.negative(np.multiply(np.hypot(x[0], np.sin(np.e)), np.log1p(np.tanh(np.negative(np.pi))))), np.hypot(x[0], np.euler_gamma)))), np.negative(x[2])), np.negative(x[2])), np.negative(x[2])), np.negative(np.cbrt(x[2]))), np.multiply(np.negative(x[1]), np.square(x[1]))) |
| 4 | 50000 | 6 | No | Yes | Yes | No | 0.09592352523610304 | np.multiply(np.sinc(np.multiply(x[1], np.multiply(np.exp(np.negative(np.sin(np.sin(np.add(np.pi, np.euler_gamma))))), np.log10(np.negative(np.sin(np.sin(np.add(np.pi, np.euler_gamma)))))))), np.hypot(np.multiply(x[1], np.pi), np.hypot(np.add(np.add(np.add(np.pi, np.i0(x[1])), np.euler_gamma), np.add(np.pi, np.exp(np.euler_gamma))), np.pi))) |
| 5 | 15000 | 6 | Yes | Yes | Yes | No | 5.512898055429115e-19 | np.add(np.divide(np.add(np.sin(np.hypot(np.power(np.hypot(np.power(np.hypot(np.power(x[0], np.negative(np.sin(x[0]))), np.euler_gamma), np.negative(np.sin(np.round(x[1])))), np.arctan(x[0])), np.negative(np.sin(x[1]))), np.arctan(np.pi))), np.add(np.tanh(np.i0(np.tanh(np.power(np.exp2(x[1]), np.sin(x[0]))))), np.pi)), 174318497.75735644), -2.8520706810421616e-08) |
| 6 | 50000 | 6 | No | No | Yes | No | 7.145589969138924e-07 | np.add(np.multiply(np.arctan(np.cbrt(np.euler_gamma)), np.add(x[1], np.negative(x[0]))), x[1]) |
| 7 | 50000 | 6 | No | No | Yes | Yes | 54.844555000111505 | np.hypot(np.multiply(np.add(np.add(np.add(np.hypot(x[0], x[0]), np.hypot(np.multiply(np.remainder(x[1], np.multiply(x[0], np.cosh(np.euler_gamma))), np.e), x[0])), np.cosh(np.remainder(x[1], x[0]))), np.multiply(np.hypot(np.e, np.add(np.multiply(np.square(np.cosh(np.sin(np.sin(np.multiply(x[1], np.cosh(np.cosh(x[1]))))))), np.multiply(x[1], np.remainder(np.cbrt(np.multiply(np.hypot(x[0], np.add(np.hypot(np.multiply(x[1], np.remainder(x[1], np.multiply(x[0], np.cosh(np.sin(np.e))))), x[0]), np.e)), np.multiply(x[1], np.exp2(np.hypot(x[1], np.log10(np.euler_gamma)))))), x[0]))), np.e)), np.multiply(x[1], x[0]))), np.multiply(np.hypot(np.sin(np.multiply(x[1], np.square(np.cosh(np.euler_gamma)))), np.remainder(x[0], np.multiply(x[1], np.cosh(np.euler_gamma)))), np.exp2(np.hypot(np.remainder(x[0], np.multiply(np.remainder(x[1], np.multiply(np.remainder(x[0], np.multiply(np.remainder(x[1], np.multiply(x[0], np.cosh(np.sin(x[1])))), np.cosh(np.euler_gamma))), np.cosh(np.sin(np.sin(np.e))))), np.cosh(np.arctan(np.arctan(np.arctan(np.sin(np.e))))))), np.log10(np.hypot(np.remainder(x[1], x[0]), np.remainder(np.remainder(x[1], np.multiply(x[0], np.cosh(np.sin(x[1])))), x[0]))))))), np.euler_gamma) |
| 8 | 50000 | 6 | No | No | Yes | No | 166579.13617635533 | np.add(np.multiply(np.add(np.add(x[5], np.round(np.cbrt(np.negative(np.exp2(np.cbrt(np.negative(np.exp2(np.hypot(np.round(np.cbrt(np.negative(np.square(np.arctan(np.add(np.multiply(np.square(np.multiply(x[5], np.add(x[5], x[5]))), np.e), np.cbrt(np.remainder(np.arctan(np.negative(np.hypot(np.euler_gamma, np.square(np.remainder(np.arctan(np.negative(np.exp2(np.hypot(x[4], np.pi)))), np.negative(np.add(x[5], x[5]))))))), np.multiply(np.multiply(x[5], np.add(x[5], x[5])), np.add(x[5], x[5])))))))))), np.pi))))))))), x[5]), np.multiply(np.hypot(np.e, x[5]), np.multiply(x[5], np.add(x[5], x[5])))), np.add(np.multiply(np.square(np.multiply(x[5], np.add(x[5], x[5]))), x[5]), np.negative(np.square(np.hypot(np.exp2(np.hypot(np.hypot(np.euler_gamma, np.pi), x[4])), np.round(x[5])))))) |






