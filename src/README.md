# MY WONDERFUL SYMREG PROGRAM

The objective of the program was find the function which could best minimize mse on example data.
In the program modern GA flow is applied.
It's used prof Squillero gxgp library , with some added functions to support different types of mutations and other changes to speed up evaluation using numpy. A great performance improvement is using a custom evaluate function that transform the function saved in the node into the relative numpy formula string and then runs it having as parameter x.  
The program select 2 individual to crossover using a tournament selection(size 2 , low selective preassuere) always apply crossover, then the generated child is mutated with a fixed low mutation rate . Using a random mutation function between  :
mutation_point : changes a node of my tree, if it's a leef it changes it into an other constant or variable , it it's not it changes it with a function with the same arity .
mutation_hoist: cuts the tree , select a random subtree and give it as a result
mutation_permutations: changes order of operands in function node with arity =2 , avoiding commutative functions 
mutation_collapse(genome, gptree): deletes a subtree and , makes it become  a constant or a varible
mutation_delete_unirary(genome): deletes a node that had a unary functions 
mutation_add_unirary(genome,gptree): adds a level in a random point of the tree adding a unary functions

It then applies survivor selection by deleting the worst fit individual and purging it .
Fitness in the program is a combination of the mse and a penalty for lengthy individuals(to limit bloat), the penalty is scaled according to the maximum values in the y and the number of elements there multiplied by a custom parameter for each problem to try squeezing better results. 
The program does 5 restarts and then a final run using as population the best 2 individuals of each run ,, some of their mutations and individual resukt as crossover oìbetween them , plus random individuaks as always.
It's defined also a parameter called run until plateaux , that avoid limiting the number of step of each rerun and just goes on as long as there is an improvement in the last nstep/3 (when the  program assumes there is a plateaux).
I added a great number of possible numpy functions (hopying to get always a possible solution even if that means running for more epochs), there a just a few constants for the program to decide from (only the 3 constants: pi, Euler's number and Euler–Mascheroni constant).









Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

...

```python
problem = np.load('problem_X.npz')
x = problem['x']
y = problem['y']
x.shape

def fX(x: np.ndarray) -> np.ndarray: 
    ...

```
