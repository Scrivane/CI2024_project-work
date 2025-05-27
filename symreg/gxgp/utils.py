#   *        Giovanni Squillero's GP Toolbox
#  / \       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2   +      A no-nonsense GP in pure Python
#    / \
#  10   11   Distributed under MIT License

import inspect
from typing import Callable

__all__ = ['arity']

##old aririty
""" def arity(f: Callable) -> int:
    #Return the number of expected parameter or None if variable
    if inspect.getfullargspec(f).varargs is not None:
        return None
    else:
        return len(inspect.getfullargspec(f).args) """



import  numpy as np
def arity(f: Callable) -> int:
    if isinstance(f, np.ufunc):
        return f.nin

            

    """ if inspect.getfullargspec(f).varargs is not None:
            return None
    else:
            return len(inspect.getfullargspec(f).args) """
    
    try:
        if inspect.getfullargspec(f).varargs is not None:
            return None
        else:
            return len(inspect.getfullargspec(f).args)
        
    except TypeError:
        ris=inspect.signature(f)
        parameters=ris.parameters.values()
        todelete=0
        for p  in parameters:
            type=p.kind
            default=p.default
        
            
            if default==None or default==0:
                todelete+=1 
            
        return len(parameters)-todelete 
        


       

        
    




