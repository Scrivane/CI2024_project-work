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

    def create_individual(self, n_nodes=7): 
        pool = self._variables * (1 + len(self._constants) // len(self._variables)) + self._constants
  

        individual = None
        while individual is None or len(individual) < n_nodes:
            op = gxgp_random.choice(self._operators)

            params = gxgp_random.choices(pool, k=arity(op))
            individual = Node(op, params)
        
            pool.append(individual)
        return individual
    


    @staticmethod
    def default_variable(i: int) -> str:
        return f'x{i}'



    @staticmethod
    def evaluate2(individual: Node, X, variable_names=None): 
        


        formula=individual.to_np_formula()

        
        y_pred = eval(formula, {"np": np, "x": X.T})
        if np.isscalar(y_pred) or  y_pred.shape == ():
            y_pred = y_pred * np.ones(X.shape[0])  

       
        

        return y_pred



    @staticmethod
    def mse(individual: Node, X, y, variable_names=None):
        y_pred = DagGP.evaluate2(individual, X, variable_names)

        ris=np.mean((y.astype(np.float64) - y_pred.astype(np.float64)) ** 2)
        return  ris






