#   *        Giovanni Squillero's GP Toolbox
#  / \       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2   +      A no-nonsense GP in pure Python
#    / \
#  10   11   Distributed under MIT License

import numbers
import warnings
from typing import Callable


from .utils import arity

__all__ = ['Node']


class Node:   #using property here allows to call those method as a parameter doing for example .subtree instead of .subtree()
    _func: Callable
    _successors: tuple['Node']
    _arity: int
    _str: str

    def __init__(self, node=None, successors=None, *, name=None):
        if callable(node):

            def _f(*_args, **_kwargs):
                return node(*_args)

            self._func = _f
            self._successors = tuple(successors)
            self._arity = arity(node)
            assert self._arity is None or len(tuple(successors)) == self._arity, (
                "Panic: Incorrect number of children."
                + f" Expected {len(tuple(successors))} found {arity(node)}"
            )
            self._leaf = False
            assert all(isinstance(s, Node) for s in successors), "Panic: Successors must be `Node`"
            self._successors = tuple(successors)
            if name is not None:
                self._str = name
            elif node.__name__ == '<lambda>':
                self._str = 'λ'
            else:
                self._str = node.__name__
        elif isinstance(node, numbers.Number):
            self._func = eval(f'lambda **_kw: {node}')
            self._successors = tuple()
            self._arity = 0
            self._str = f'{node:g}'
        elif isinstance(node, str):
            self._func = eval(f'lambda *, {node}, **_kw: {node}')
            self._successors = tuple()
            self._arity = 0
            self._str = str(node)
        else:
            assert False

    def __call__(self, **kwargs):
        return self._func(*[c(**kwargs) for c in self._successors], **kwargs)

    def __str__(self):
        return self.long_name

    def __len__(self):
        return 1 + sum(len(c) for c in self._successors)

    @property
    def value(self):
        return self()

    @property
    def arity(self):
        return self._arity

    @property
    def successors(self):
        return list(self._successors)

    @successors.setter
    def successors(self, new_successors):
        assert len(new_successors) == len(self._successors)
        self._successors = tuple(new_successors)

    @property
    def is_leaf(self):
        return not self._successors

    @property
    def short_name(self):
        return self._str

    @property
    def long_name(self):
        if self.is_leaf:
            return self.short_name
        else:
            return f'{self.short_name}(' + ', '.join(c.long_name for c in self._successors) + ')'

    @property
    def subtree(self):
        result = set()
        _get_subtree(result, self)
        return result
    
    def to_np_formula(self):  #custom function to go from string representation , to numpy string representation (useful for evaluating the function)
        stringa=self.long_name
        newstringa=""
        openbracket=0
        for i in range(0,len(stringa)):
            char=stringa[i]

            newstringa=newstringa+stringa[i]
            if openbracket==1:
                newstringa=newstringa+"]"
                openbracket=0

            if( char=="x" and stringa[i+1].isdigit()):  
                newstringa=newstringa+"["
                openbracket=1
            

        isfun=0
        stringa=newstringa
        newstringa=""
        for i in range(0,len(stringa)):
            char=stringa[i]
            if( char.isalpha() and isfun==0 and stringa[i+1]!="["):
                isfun=1
                newstringa=newstringa+"np."
            elif(char.isalpha()==False and ( ( i!=len(stringa)-1 and stringa[i+1]!='p') or ( i<len(stringa)-2 and stringa[i+1]=='p' and stringa[i+2]=='o'))  ):
                isfun=0
            newstringa=newstringa+stringa[i]
        

        newstringa = newstringa.replace("3.14159", "np.pi")
        newstringa = newstringa.replace("0.577216", "np.euler_gamma")
        newstringa = newstringa.replace("2.71828", "np.e")
    
        
        return newstringa
  



def _get_subtree(bunch: set, node: Node):
    bunch.add(node)
    for c in node._successors:
        _get_subtree(bunch, c)
