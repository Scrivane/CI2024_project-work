#   *        Giovanni Squillero's GP Toolbox
#  / \       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2   +      A no-nonsense GP in pure Python
#    / \
#  10   11   Distributed under MIT License

from typing import Collection

from .node import Node
from .random import gxgp_random
from .utils import arity
import numpy as np


__all__ = ['DagGP']

non_zero_func = ["reciprocal"]
non_negative_func = ["log", "log10", "log2", "log1p", "pow"]  #togli pow
high_func = ["exp", "exp2", "pow", "sinh", "cosh", "i0", "square"]
class DagGP:
    def __init__(self, operators: Collection, variables: int | Collection, constants: int | Collection):
        self._operators = list(operators)
        if isinstance(variables, int):
            self._variables = [Node(DagGP.default_variable(i)) for i in range(variables)]
        else:
            self._variables = [Node(t) for t in variables]
        if isinstance(constants, int):
            self._constants = [Node(gxgp_random.random()) for i in range(constants)]
        else:
            self._constants = [Node(t) for t in constants]

    def create_individual(self, n_nodes=7): #//fix here it's never x2
        pool = self._variables * (1 + len(self._constants) // len(self._variables)) + self._constants
  

        individual = None
        while individual is None or len(individual) < n_nodes:
            op = gxgp_random.choice(self._operators)
            localpool=pool.copy()  #copy
            """ if op.__name__ in non_zero_func:
                localpool= [el for el in localpool if  el.value!=0 ] 
            if op.__name__ in non_negative_func:
                localpool=[el for el in localpool if  el.value>=0 ] 
            
            if op.__name__ in high_func:
                localpool=[el for el in localpool if  el.value<=20 and el.value>=-20 ]  """
                
            params = gxgp_random.choices(pool, k=arity(op))
            individual = Node(op, params)
            """ if ("x2" in individual.long_name):
                print(individual) """
            pool.append(individual)
        return individual
    
    def create_individuals(self, n_nodes=7):  #all different individuals
        pool = self._variables * (1 + len(self._constants) // len(self._variables)) + self._constants
        individual = None
        for operand in self._operators:
            k=arity(operand)
            if k==1:
                for var in self._variables:
                    individual = Node(operand, [var])
                    pool.append(individual)
            
            elif k==2:
                for var1 in self._variables:
                    for var2 in self._variables:
                        individual = Node(operand, [var1,var2])
                        pool.append(individual)




        """     individual = Node(op, params)
        while individual is None or len(individual) < n_nodes:
            op = gxgp_random.choice(self._operators)
            params = gxgp_random.choices(pool, k=arity(op))
            individual = Node(op, params)
            pool.append(individual) """
        return pool

    @staticmethod
    def default_variable(i: int) -> str:
        return f'x{i}'

    @staticmethod
    def evaluate(individual: Node, X, variable_names=None):  ##change
        formula=individual.to_np_formula()
        y_pred = eval(formula, {"np": np, "x": X.T})
        if np.isscalar(y_pred) or  y_pred.shape == ():
            #y_pred = np.full(X.shape[0], y_pred)
            y_pred = y_pred * np.ones(X.shape[0])  

       
        

        return y_pred
    

    @staticmethod
    def evaluate2(individual: Node, X, variable_names=None):  ##change
        """ if variable_names:
            names = variable_names
        else:
            names = [DagGP.default_variable(i) for i in range(len(X[0]))] """


        formula=individual.to_np_formula()

        
        y_pred = eval(formula, {"np": np, "x": X.T})
        if np.isscalar(y_pred) or  y_pred.shape == ():
            #y_pred = np.full(X.shape[0], y_pred)
            y_pred = y_pred * np.ones(X.shape[0])  

       
        

        return y_pred

    @staticmethod
    def plot_evaluate(individual: Node, X, variable_names=None):
        import matplotlib.pyplot as plt

        y_pred = DagGP.evaluate(individual, X, variable_names)
        plt.figure()
        plt.title(individual.long_name)
        plt.scatter([x[0] for x in X], y_pred)

        return y_pred

    @staticmethod
    def mse(individual: Node, X, y, variable_names=None):
        y_pred = DagGP.evaluate2(individual, X, variable_names)
        #y_pred = DagGP.evaluate(individual, X, variable_names)

        ris=np.mean((y.astype(np.float64) - y_pred.astype(np.float64)) ** 2)
        #ris=sum((a - b) ** 2 for a, b in zip(y, y_pred)) / len(y)
        return  ris






