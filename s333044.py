# Copyright © 2024 Giovanni Squillero <giovanni.squillero@polito.it>
# https://github.com/squillero/computational-intelligence
# Free under certain conditions — see the license for details.

import numpy as np

# All numpy's mathematical functions can be used in formulas
# see: https://numpy.org/doc/stable/reference/routines.math.html


# Notez bien: No need to include f0 -- it's just an example!
def f0(x: np.ndarray) -> np.ndarray:
    return x[0] + np.sin(x[1]) / 5


def f1(x: np.ndarray) -> np.ndarray: ...



def testf2(x) -> np.ndarray: # [np.float64(5420086.098834277), np.float64(2934417.028615387)]
     for i in range(x.shape[0]):
          print(x[i])
    
     x=[4.52991777,-0.18656382,-1.60071107]

     print(np.multiply(np.add(np.multiply(np.sinh(np.add(np.tanh(1), 1)), 
              np.add(np.exp(np.sinh(np.pi)), np.exp(np.e))), np.sinh(x[0])), 
              np.add(np.multiply(np.reciprocal(np.exp(np.tanh(np.negative(np.exp(np.sinh(x[1])))))), 
              np.add(x[0], x[0])), np.negative(np.reciprocal(x[2])))) )
     
     print(np.float64(5420086.098834277))
     
     x=[1.90003334,4.07797331,3.23380643]
     print(np.multiply(np.add(np.multiply(np.sinh(np.add(np.tanh(1), 1)), 
              np.add(np.exp(np.sinh(np.pi)), np.exp(np.e))), np.sinh(x[0])), 
              np.add(np.multiply(np.reciprocal(np.exp(np.tanh(np.negative(np.exp(np.sinh(x[1])))))), 
              np.add(x[0], x[0])), np.negative(np.reciprocal(x[2])))) )
     print(np.float64(2934417.028615387))

     return   np.multiply(np.add(np.multiply(np.sinh(np.add(np.tanh(1), 1)), 
              np.add(np.exp(np.sinh(np.pi)), np.exp(np.e))), np.sinh(x[0])), 
              np.add(np.multiply(np.reciprocal(np.exp(np.tanh(np.negative(np.exp(np.sinh(x[1])))))), 
              np.add(x[0], x[0])), np.negative(np.reciprocal(x[2])))) 

def f2(x: np.ndarray) -> np.ndarray:




    """ ic| best_one.fitness: np.float64(18549826191420.848)
ic| formula: ('np.multiply(np.exp2(np.e), '
              'np.add(np.multiply(np.multiply(np.add(np.add(np.tanh(x[0]), x[0]), x[0]), '
              'np.exp(np.exp(np.sinh(np.cosh(-1))))), np.add(np.pi, np.pi)), '
              'np.add(np.add(np.tanh(np.sin(np.exp2(np.add(np.e, '
              'np.round(np.negative(np.e)))))), np.round(np.negative(np.e))), '
              'np.multiply(np.add(np.multiply(np.reciprocal(np.exp(np.add(x[2], x[2]))), '
              'x[0]), x[2]), x[0]))))')
ic| ygen.shape: (5000,)
ic| y.shape: (5000,)
ic| x.shape: (3, 5000)
ic| ris: np.float64(18549826191418.91) """


    """     ic| best_one.fitness: np.float64(25498062664772.316)
    ic| formula: ('np.multiply(np.add(np.exp(np.exp2(np.pi)), np.add(x[0], x[2])), '
                'np.add(np.add(np.add(0, np.tanh(np.add(0, '
                'np.tanh(np.multiply(np.reciprocal(np.sinh(np.add(np.reciprocal(np.reciprocal(np.multiply(np.tanh(1), '
                'np.add(np.tanh(np.add(np.exp(np.exp2(np.pi)), '
                'np.add(np.add(np.add(np.cosh(np.negative(x[1])), np.sin(x[0])), x[2]), '
                'x[2]))), x[0])))), np.sin(x[0])))), np.add(np.add(x[0], '
                'np.sin(np.add(np.add(np.add(np.e, x[0]), x[2]), x[0]))), np.pi)))))), '
                'np.sinh(np.add(np.e, x[0]))), '
                'np.sinh(np.add(np.reciprocal(np.reciprocal(np.multiply(np.tanh(1), '
                'np.add(x[0], np.round(x[0]))))), np.sin(x[0])))))')
    ic| ygen.shape: (5000,)
    ic| y.shape: (5000,)
    ic| x.shape: (3, 5000)
    ic| ris: np.float64(25498062664766.99) """
   

    #9809284786418.186  no len penalty , alway crossover 0.05 di mutrate

    return np.multiply(np.hypot(np.sinh(np.pi), np.exp(np.pi)), np.add(np.add(np.add(np.multiply(np.hypot(np.absolute(np.add(np.log1p(np.pi), np.sinh(np.pi))), np.hypot(np.absolute(np.add(x[0], np.log1p(np.tanh(x[1])))), np.absolute(np.sinh(np.pi)))), np.absolute(np.add(x[0], np.log1p(np.tanh(np.sinh(np.exp2(np.pi))))))), np.absolute(np.i0(np.absolute(np.add(x[0], np.log1p(np.tanh(x[1]))))))), np.absolute(np.exp2(x[2]))), np.add(np.multiply(np.tanh(np.add(np.multiply(np.tanh(np.add(x[0], np.log1p(np.tanh(x[1])))), np.absolute(np.sinh(np.sinh(np.pi)))), np.add(np.multiply(np.hypot(np.hypot(np.pi, np.absolute(np.exp2(np.sinh(np.pi)))), np.hypot(np.absolute(np.absolute(np.absolute(np.hypot(np.pi, x[2])))), np.absolute(np.exp2(np.pi)))), np.e), np.add(np.add(np.exp2(x[2]), np.add(np.exp2(np.sinh(np.pi)), np.add(np.multiply(np.multiply(np.hypot(np.absolute(np.exp2(np.sinh(np.pi))), x[0]), np.sinh(np.pi)), x[0]), np.add(np.add(np.multiply(np.log1p(np.tanh(np.sinh(np.pi))), x[0]), np.add(np.exp2(np.sinh(np.pi)), np.multiply(np.absolute(np.absolute(np.absolute(np.sinh(np.exp2(np.pi))))), np.sinh(x[1])))), np.add(np.add(np.multiply(np.sinh(np.pi), np.sinh(np.log1p(np.tanh(np.exp2(np.pi))))), np.add(np.sinh(np.pi), x[1])), np.add(np.add(np.multiply(np.hypot(np.hypot(np.exp2(np.pi), np.absolute(np.multiply(np.exp2(np.hypot(np.pi, np.hypot(x[1], np.sinh(np.pi)))), np.pi))), np.exp2(np.add(x[0], np.log1p(np.tanh(x[1]))))), x[2]), np.hypot(np.absolute(np.exp2(np.hypot(np.pi, np.multiply(np.absolute(np.i0(np.absolute(np.sin(np.exp2(np.sinh(np.pi)))))), np.sinh(np.e))))), x[1])), np.add(np.multiply(np.hypot(np.hypot(np.absolute(np.exp2(np.hypot(np.absolute(np.add(x[0], np.log1p(np.tanh(x[1])))), np.absolute(np.sinh(np.pi))))), np.absolute(np.exp2(np.sinh(np.pi)))), np.absolute(np.exp2(x[0]))), x[0]), np.multiply(np.hypot(np.absolute(np.exp2(np.absolute(np.absolute(np.exp2(x[0]))))), np.exp(np.pi)), np.square(np.pi))))))))), np.add(np.hypot(np.pi, np.multiply(np.absolute(np.i0(np.absolute(np.sin(np.exp2(np.sinh(np.pi)))))), np.sinh(np.e))), np.add(np.multiply(np.hypot(np.hypot(np.absolute(np.absolute(np.add(x[0], np.log1p(np.tanh(np.sinh(np.exp2(np.pi))))))), np.absolute(np.hypot(np.absolute(np.multiply(np.exp2(np.hypot(np.pi, np.hypot(np.pi, np.sinh(np.pi)))), np.pi)), np.absolute(np.exp2(np.hypot(np.pi, np.hypot(np.pi, np.sinh(np.pi)))))))), np.exp2(np.sinh(np.pi))), x[2]), np.add(np.log1p(np.pi), np.sinh(np.exp2(np.pi))))))))), np.absolute(np.sinh(np.hypot(np.pi, np.hypot(np.pi, np.sinh(np.pi)))))), np.add(np.multiply(np.hypot(np.hypot(np.absolute(np.hypot(np.pi, np.pi)), np.absolute(np.exp2(np.pi))), np.hypot(np.absolute(np.exp2(np.pi)), np.absolute(np.absolute(np.sinh(np.pi))))), np.e), np.add(np.add(np.exp2(np.pi), np.add(np.exp2(np.add(x[0], np.log1p(np.tanh(x[1])))), np.add(np.multiply(np.multiply(np.hypot(np.absolute(np.exp2(np.sinh(np.pi))), x[0]), np.sinh(np.e)), x[0]), np.add(np.add(np.multiply(np.exp2(np.pi), x[0]), np.add(np.square(x[2]), np.absolute(np.absolute(np.absolute(np.exp2(np.pi)))))), np.add(np.add(np.multiply(np.sinh(np.pi), np.sinh(np.log1p(np.tanh(x[2])))), np.add(np.add(np.multiply(np.hypot(np.absolute(np.add(np.log1p(np.pi), np.sinh(np.pi))), np.hypot(np.exp2(x[1]), np.log1p(np.tanh(np.hypot(np.pi, np.absolute(np.exp2(np.sinh(np.pi)))))))), x[2]), np.add(np.multiply(np.hypot(np.absolute(np.add(np.log1p(np.pi), np.sinh(np.pi))), np.hypot(np.absolute(np.add(x[0], np.log1p(np.tanh(x[1])))), np.absolute(np.sinh(np.pi)))), np.absolute(np.add(x[0], np.log1p(np.tanh(np.sinh(np.exp2(np.pi))))))), np.absolute(np.i0(np.absolute(np.add(x[0], np.log1p(np.tanh(x[1])))))))), x[1])), np.add(np.add(np.add(np.multiply(np.hypot(np.absolute(np.add(np.log1p(np.pi), np.sinh(x[2]))), np.hypot(np.absolute(np.add(x[0], np.log1p(np.tanh(x[1])))), np.absolute(np.sinh(np.pi)))), x[2]), np.absolute(np.i0(np.absolute(np.add(x[0], np.log1p(np.tanh(x[1]))))))), np.absolute(np.exp2(x[0]))), np.add(np.add(np.multiply(np.hypot(np.absolute(np.add(np.log1p(np.pi), np.sinh(np.pi))), np.hypot(np.absolute(np.add(np.pi, np.log1p(np.tanh(x[1])))), np.multiply(np.absolute(np.exp2(np.pi)), np.sinh(x[1])))), x[2]), np.absolute(np.i0(np.absolute(np.add(x[0], np.log1p(np.tanh(x[1]))))))), np.absolute(np.exp2(x[0]))))))))), np.add(np.hypot(np.pi, np.multiply(np.absolute(np.i0(np.absolute(np.add(x[0], np.log1p(np.tanh(x[2])))))), x[0])), np.add(np.add(np.square(x[2]), np.absolute(np.hypot(np.pi, np.sinh(np.pi)))), np.pi)))))))


    #return np.add(np.add(np.add(np.add(np.add(np.round(np.add(x[1],np.add(np.exp(np.exp(0)), x[0]))), np.pi), np.round(np.add(np.sin(np.tanh(np.cosh(np.tanh(np.pi)))),np.add(np.exp(x[1]), np.add(np.add(x[0], np.e), np.sinh(np.add(np.add(x[0], x[0]), x[0]))))))), np.pi), np.round(np.add(x[1], np.add(np.exp(np.add(x[0],np.multiply(np.pi, np.pi))), np.add(np.add(np.round(x[2]), np.exp(np.add(x[0], np.multiply(np.pi, np.pi)))), np.negative(np.multiply(x[0], np.exp(np.add(x[0], x[0]))))))))), np.pi)
    #26631472916904.3
    #15558346254051.248
          #all below are false
     

    """ return   np.multiply(np.add(np.multiply(np.sinh(np.add(np.tanh(1), 1)), 
              np.add(np.exp(np.sinh(np.pi)), np.exp(np.e))), np.sinh(x[0])), 
              np.add(np.multiply(np.reciprocal(np.exp(np.tanh(np.negative(np.exp(np.sinh(x[1])))))), 
              np.add(x[0], x[0])), np.negative(np.reciprocal(x[2]))))  #137169039.58563918
     return np.multiply(np.add(np.multiply(np.add(np.exp(x[0]), np.add(-1, x[2])),   #make that constant are translatable 
              np.add(np.exp(x[0]), np.multiply(np.reciprocal(x[1]), 0))), np.exp(x[0])), 
              np.round(np.add(np.multiply(np.add(np.sin(np.sin(x[1])), 
              np.add(np.add(np.cosh(np.sinh(np.power(0.577216, np.multiply(3.14159, 
              3.14159)))), np.power(np.exp(np.multiply(np.round(x[2]), 1)), 
              np.tanh(np.power(1, np.round(x[2]))))), x[1])), np.cosh(np.multiply(x[1], 
              3.14159))), np.exp(x[0])))) """                                                 #2726315836.663211
    # 
    # return np.multiply(np.sin(np.tanh(np.exp2(np.tanh(np.exp2(x[0]))))), np.exp2(np.exp2(x[0]))) 
                                                                                                #3236255985.3856373
                                                                                                 #3238539087.1239223
    #return np.multiply(np.round(np.multiply(x[0], np.power(np.multiply(x[0], x[0]),x[0]))), x[1])  #eror of 3241102705.3421373
    #return np.multiply(np.sin(np.tanh(np.round(x[1]))), np.exp2(np.exp2(x[0])))
    #return np.multiply(np.round(np.multiply(np.power(np.multiply(x[0], x[0]), x[0]), x[0])), x[1])
    #return np.multiply(np.power(np.exp(np.cosh(x[1])), np.add(x[0], np.multiply(x[0], np.power(np.exp(np.negative(x[1])), x[1])))), np.tanh(np.cosh(np.add(x[0], np.exp(np.sinh(np.negative(x[1])))))))
   # return np.multiply(np.power(np.exp(np.sinh(np.negative(np.tanh(np.cosh(np.multiply(x[1], x[0])))))), np.add(x[1], np.tanh(np.cosh(x[0])))), np.power(np.multiply(x[0], x[0]), np.add(x[0], x[1])))

def f3(x: np.ndarray) -> np.ndarray: 
     #always crossover , mutation of 0.05
     #8.006929603960224   mse   using scaled npenalty and hardlimi ton depth after crossover of 1000 (otherwishe it would generate  anew one )
     return np.add(np.multiply(np.negative(x[1]), np.multiply(x[1], x[1])), np.hypot(np.add(np.multiply(np.negative(np.hypot(np.hypot(x[0], x[0]), np.pi)), np.hypot(np.hypot(np.log1p(np.hypot(np.negative(np.exp2(np.add(np.multiply(np.negative(np.hypot(np.hypot(x[0], x[0]), np.hypot(np.hypot(x[0], x[0]), np.pi))), np.log1p(np.exp2(np.log1p(np.exp2(x[2]))))), np.multiply(np.pi, x[2])))), np.log1p(np.pi))), np.negative(x[0])), x[0])), np.multiply(np.pi, x[2])), np.exp2(np.negative(np.exp2(x[2])))))




     #14.466307622905152
     return np.add(np.multiply(np.multiply(x[1], x[1]), np.negative(x[1])), np.hypot(np.add(np.negative(np.hypot(np.add(np.negative(np.hypot(np.pi, x[0])), x[2]), x[2])), x[2]), np.hypot(np.add(np.negative(np.hypot(np.negative(np.add(np.negative(np.hypot(np.pi, x[0])), x[2])), x[2])), x[2]), np.multiply(np.hypot(x[0], np.hypot(x[0], np.add(np.negative(np.hypot(np.hypot(x[0], x[0]), x[2])), x[2]))), x[0]))))
     
     
     #mse 175.38503573732595
     return np.add(np.hypot(np.power(np.exp(np.sin(np.hypot(x[1], np.round(np.cosh(np.round(np.e)))))), x[1]), np.add(np.hypot(np.add(np.hypot(np.i0(x[0]), np.hypot(np.exp(np.sinh(np.log1p(np.exp(np.exp(np.sin(np.hypot(x[1], np.e))))))), np.hypot(np.add(np.power(np.absolute(x[1]), np.add(np.log1p(np.exp(np.sin(x[1]))), np.log1p(np.exp(np.sinh(np.log1p(np.exp(np.sin(np.log1p(np.log1p(np.power(np.exp(np.sin(np.hypot(x[1], np.e))), x[1]))))))))))), np.i0(x[0])), np.i0(x[0])))), np.log(np.power(np.power(np.exp(np.sin(np.hypot(x[1], np.cosh(np.round(np.pi))))), x[1]), np.exp(np.sinh(np.log1p(np.exp(np.log1p(np.log1p(np.i0(x[0])))))))))), np.hypot(np.power(np.exp(np.sin(np.hypot(x[1], np.e))), x[1]), np.i0(x[0]))), np.log(np.power(np.exp(np.sin(np.hypot(x[1], np.e))), x[1])))), np.log(np.power(np.power(np.power(np.exp(np.sin(np.sin(x[1]))), x[1]), x[1]), x[1])))


def f4(x: np.ndarray) -> np.ndarray: #as before


     #min_mse
     #0.3802252366701618


     return np.add(np.sin(np.add(np.absolute(x[1]), np.hypot(np.cosh(np.log2(np.round(np.round(np.hypot(np.cosh(np.log2(np.euler_gamma)), np.cosh(np.log2(np.euler_gamma))))))), np.log2(np.euler_gamma)))), np.round(np.add(np.sin(np.add(np.absolute(x[1]), np.round(np.hypot(np.cosh(np.log2(np.euler_gamma)), np.log2(np.euler_gamma))))), np.round(np.add(np.sin(np.i0(np.round(np.add(np.absolute(x[1]), np.log1p(np.euler_gamma))))), np.add(np.sin(np.add(np.i0(np.round(x[1])), np.euler_gamma)), np.round(np.add(np.sin(np.add(np.i0(np.round(np.absolute(x[1]))), np.log1p(np.euler_gamma))), np.round(np.i0(np.round(np.add(np.sin(np.log1p(np.tanh(np.sin(np.add(np.i0(np.round(x[1])), np.log1p(np.add(np.absolute(x[1]), np.log1p(np.tanh(np.sin(np.euler_gamma)))))))))), np.round(np.round(np.hypot(np.cosh(np.log2(np.euler_gamma)), np.log2(np.euler_gamma))))))))))))))))
     
     #4.653280870773529
     return np.multiply(np.cosh(np.pi), np.power(np.reciprocal(np.i0(np.cosh(x[1]))), np.log1p(np.cosh(x[1]))))


def f5(x: np.ndarray) -> np.ndarray: 
     #min_mse
     #5.572809570693574e-18

     return np.multiply(np.sin(np.pi), np.negative(x[0]))


def f6(x: np.ndarray) -> np.ndarray: ...


def f7(x: np.ndarray) -> np.ndarray: ...


def f8(x: np.ndarray) -> np.ndarray: ...
