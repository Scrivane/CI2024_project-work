# Copyright © 2024 Giovanni Squillero <giovanni.squillero@polito.it>
# https://github.com/squillero/computational-intelligence
# Free under certain conditions — see the license for details.

import numpy as np

# All numpy's mathematical functions can be used in formulas
# see: https://numpy.org/doc/stable/reference/routines.math.html


# Notez bien: No need to include f0 -- it's just an example!
def f0(x: np.ndarray) -> np.ndarray:
    return x[0] + np.sin(x[1]) / 5


def f1(x: np.ndarray) -> np.ndarray: 
     #0 mse , 1 restart , 1000 steps , no final run 
     return np.sin(x[0])



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

def f2(x: np.ndarray) -> np.ndarray:  #could benefit more steps
     #nrestarts 5 , last run nstep 500000
     #1548510021431.2053

     return np.multiply(np.tanh(np.add(np.multiply(x[0], np.log2(np.absolute(np.exp(np.log1p(np.e))))), np.add(x[1], x[2]))), np.multiply(np.tanh(np.maximum(np.maximum(np.sin(np.maximum(np.add(np.log1p(np.maximum(np.sin(np.arctan(np.sin(np.pi))), x[0])), np.arctan(x[0])), x[0])), np.add(np.sin(np.exp(np.log1p(np.exp(np.maximum(np.arctan(np.add(np.add(np.maximum(np.pi, np.arctan(np.arctan(np.sin(np.arctan(np.add(x[2], np.log1p(np.sin(np.sin(np.add(np.multiply(x[0], np.log2(np.absolute(np.exp(np.log1p(np.e))))), np.add(x[1], x[2]))))))))))), x[2]), x[1])), np.sin(x[0])))))), np.log1p(np.tanh(np.maximum(np.tanh(np.log1p(np.tanh(np.maximum(np.maximum(np.tanh(np.add(np.multiply(x[0], np.log2(np.absolute(np.add(np.maximum(np.add(np.e, x[2]), np.arctan(np.add(x[2], x[0]))), x[1])))), np.add(np.exp2(np.sin(np.exp2(np.multiply(np.maximum(np.add(np.add(np.maximum(np.add(np.sin(np.pi), x[2]), np.add(np.sin(np.sin(x[0])), x[1])), np.sin(x[0])), x[1]), np.add(np.log1p(np.e), np.tanh(np.pi))), np.maximum(x[2], np.arctan(np.exp2(np.sin(np.maximum(x[0], np.sin(x[0])))))))))), np.maximum(np.log1p(np.pi), np.sin(np.log1p(np.e)))))), np.add(np.log1p(np.sin(np.arctan(np.sin(np.add(np.sin(np.arctan(np.add(x[2], np.log1p(np.sin(np.log1p(np.tanh(np.sin(np.arctan(np.add(x[1], x[1])))))))))), x[0]))))), np.add(x[1], np.arctan(np.sin(np.exp(np.log1p(np.e))))))), np.add(np.sin(np.add(np.multiply(x[0], np.log2(np.absolute(np.exp(np.log1p(np.e))))), np.add(x[1], x[2]))), np.arctan(np.add(np.add(np.maximum(np.add(np.maximum(np.pi, x[0]), x[2]), np.add(np.tanh(x[2]), np.arctan(np.sin(np.sin(x[0]))))), x[1]), np.sin(np.arctan(np.add(np.multiply(x[0], np.log2(np.absolute(np.exp(np.log1p(np.e))))), np.add(x[1], x[2]))))))))))), np.add(np.sin(x[0]), np.arctan(np.add(np.add(np.maximum(np.add(np.maximum(np.pi, x[0]), x[2]), np.add(np.tanh(np.sin(np.arctan(np.sin(x[0])))), x[2])), x[1]), np.sin(np.exp(np.arctan(np.add(x[2], x[1])))))))))))), np.add(np.log1p(np.tanh(np.log1p(np.tanh(np.tanh(np.log1p(np.tanh(np.add(np.maximum(np.add(np.maximum(np.pi, x[0]), x[2]), np.add(np.sin(x[0]), x[2])), x[1])))))))), np.maximum(np.sinh(np.add(np.tanh(np.add(np.tanh(np.sin(np.exp(np.log1p(np.exp(np.maximum(np.arctan(np.maximum(np.arctan(np.arctan(np.arctan(np.sin(np.exp(np.log1p(np.e)))))), np.sin(np.maximum(np.sin(x[0]), np.maximum(np.add(x[0], np.log2(np.absolute(np.add(np.maximum(np.add(np.maximum(np.pi, np.maximum(np.sin(np.arctan(np.sin(np.add(np.sin(np.arctan(np.add(x[2], np.log1p(np.sin(np.log1p(np.tanh(np.pi))))))), x[0])))), x[0])), x[2]), x[0]), x[1])))), x[0]))))), np.sin(np.maximum(np.arctan(np.log1p(np.tanh(np.sin(np.arctan(np.negative(np.cosh(np.pi))))))), np.sin(np.maximum(np.sin(x[0]), np.maximum(np.add(x[0], np.log2(np.absolute(np.add(np.maximum(np.add(np.maximum(np.arctan(np.maximum(x[2], np.arctan(np.exp2(x[2])))), x[1]), x[2]), np.add(np.sin(np.sin(np.arctan(np.sin(np.arctan(np.add(x[2], np.log1p(np.sin(np.sin(np.exp2(np.arctan(np.add(x[0], x[1])))))))))))), x[0])), x[1])))), x[0]))))))))))), np.sin(np.exp(np.arctan(np.add(x[2], x[1])))))), np.sin(np.maximum(np.add(np.log1p(np.sin(np.sin(np.log1p(np.sin(np.sin(np.arctan(np.sin(np.add(np.sin(np.arctan(np.add(x[2], np.log1p(np.sin(np.log1p(np.tanh(np.sin(np.arctan(np.add(x[1], x[1])))))))))), x[0]))))))))), np.arctan(np.arctan(np.sin(np.arctan(np.add(x[2], x[1])))))), np.sin(np.maximum(np.maximum(np.exp(np.log1p(np.exp(np.maximum(np.sin(np.maximum(np.exp(np.log1p(np.maximum(np.sin(np.arctan(np.add(x[0], np.log2(np.absolute(np.add(np.maximum(np.add(np.maximum(np.pi, x[0]), x[2]), np.add(np.sin(x[0]), x[2])), x[1])))))), x[0]))), np.arctan(x[0]))), np.arctan(np.sin(np.arctan(np.add(x[2], x[1])))))))), np.add(np.exp2(np.sin(x[0])), np.exp2(np.tanh(np.sin(np.exp(np.maximum(np.arctan(np.add(np.add(np.maximum(np.pi, np.exp2(np.arctan(np.add(x[2], x[1])))), x[2]), x[1])), np.sin(np.sin(x[0]))))))))), np.tanh(x[2]))))))), np.sin(np.maximum(np.add(np.log1p(np.maximum(np.sin(np.arctan(np.sin(np.add(np.sin(np.arctan(np.add(x[2], np.log1p(np.sin(np.log1p(np.tanh(np.sin(np.arctan(np.add(x[1], x[1])))))))))), x[0])))), np.multiply(x[0], np.log2(np.absolute(np.exp(np.log1p(np.exp(np.log1p(np.e))))))))), np.arctan(np.add(x[2], x[1]))), x[0])))))), np.sinh(np.hypot(np.negative(np.cosh(np.pi)), np.negative(np.cosh(np.pi))))))
     
     #nrestarts=10
     #nsteps = 5000
     #min_mse no scaling
     #8645320196154.205 
     
     #return np.multiply(np.add(np.hypot(np.cbrt(np.exp2(np.hypot(np.hypot(np.multiply(np.tanh(np.e), np.square(np.add(np.add(np.cbrt(np.absolute(np.add(np.multiply(np.sin(np.add(np.hypot(np.hypot(np.multiply(np.tanh(np.exp2(np.exp2(np.cosh(np.negative(np.e))))), np.square(np.exp2(np.e))), x[2]), np.multiply(x[0], np.i0(np.e))), np.e)), np.add(np.exp2(np.e), np.round(x[1]))), np.cbrt(np.cosh(np.e))))), np.cbrt(x[2])), np.e))), x[1]), np.cbrt(np.square(np.add(np.sin(x[0]), np.e)))))), np.cbrt(np.e)), np.multiply(np.multiply(x[0], np.pi), np.absolute(np.hypot(np.sinh(np.cosh(np.negative(np.e))), np.cbrt(np.square(np.hypot(np.hypot(np.cbrt(np.euler_gamma), x[1]), np.cbrt(np.cbrt(np.e))))))))), np.multiply(np.square(np.e), np.add(np.hypot(np.hypot(np.multiply(np.tanh(np.cosh(np.exp(np.multiply(np.sin(np.add(np.pi, np.cbrt(x[2]))), np.add(np.sin(np.add(np.sin(x[0]), np.e)), np.round(x[1])))))), np.hypot(np.hypot(np.multiply(np.tanh(np.add(np.hypot(np.hypot(np.multiply(np.tanh(np.exp2(np.exp2(np.cosh(np.negative(np.e))))), np.square(np.exp2(np.e))), x[2]), np.multiply(x[0], np.i0(np.e))), np.e)), np.cbrt(np.e)), np.tanh(x[0])), np.hypot(np.cbrt(np.tanh(np.add(np.multiply(np.tanh(np.add(np.arctan(np.add(np.multiply(np.sin(np.add(np.pi, np.cbrt(x[2]))), np.add(np.e, np.round(x[1]))), np.cbrt(x[2]))), np.cbrt(np.tanh(np.add(np.cbrt(x[1]), np.cbrt(np.square(np.exp2(np.e)))))))), np.square(np.hypot(np.hypot(np.cbrt(np.cbrt(np.round(np.tanh(np.tanh(np.add(np.arctan(np.add(np.multiply(np.sin(np.add(np.pi, np.cbrt(x[2]))), np.add(np.e, np.e)), np.cbrt(x[2]))), np.cbrt(np.tanh(np.add(np.cbrt(x[1]), np.cbrt(np.square(np.exp2(np.e)))))))))))), x[1]), np.cbrt(np.cbrt(np.cbrt(np.e)))))), np.cbrt(np.add(np.cbrt(x[1]), np.cbrt(x[2])))))), np.hypot(np.hypot(np.multiply(np.tanh(np.add(np.cbrt(np.sin(x[0])), np.cbrt(np.tanh(np.add(np.cbrt(x[1]), np.cbrt(x[2])))))), np.square(np.exp2(np.e))), x[1]), np.cbrt(np.hypot(np.hypot(np.multiply(np.hypot(np.hypot(np.multiply(np.tanh(np.add(np.cbrt(np.sin(np.square(np.arctan(np.hypot(np.exp(np.sinh(np.arctan(np.multiply(np.cbrt(np.tanh(np.add(np.cbrt(np.sin(np.square(np.arctan(x[0])))), np.cbrt(np.tanh(np.add(np.multiply(np.tanh(np.add(np.arctan(np.add(np.multiply(np.sin(np.add(np.pi, np.cbrt(x[2]))), np.add(np.e, np.round(x[1]))), np.cbrt(x[2]))), np.cbrt(np.tanh(np.add(np.cbrt(x[1]), np.cbrt(np.square(np.exp2(np.e)))))))), np.square(np.hypot(np.hypot(np.cbrt(x[1]), x[1]), np.cbrt(np.cbrt(np.cbrt(np.e)))))), np.cbrt(np.add(np.cbrt(x[1]), np.cbrt(x[2]))))))))), np.i0(np.e))))), np.sinh(np.cbrt(x[2]))))))), np.cbrt(np.tanh(np.add(np.multiply(np.tanh(np.add(np.arctan(np.add(np.multiply(np.sin(np.add(np.pi, np.cbrt(x[2]))), np.add(np.e, np.round(x[1]))), np.cbrt(x[2]))), np.cbrt(np.tanh(np.add(np.cbrt(x[1]), np.cbrt(np.square(np.exp2(np.e)))))))), np.square(np.hypot(np.hypot(np.cbrt(np.cbrt(np.round(np.tanh(np.tanh(np.add(np.arctan(np.add(np.multiply(np.sin(np.add(np.pi, np.cbrt(x[2]))), np.add(np.e, np.round(x[1]))), np.cbrt(x[2]))), np.cbrt(np.tanh(np.add(np.cbrt(x[1]), np.cbrt(np.square(np.exp2(np.euler_gamma)))))))))))), x[1]), np.cbrt(np.euler_gamma)))), np.cbrt(np.add(np.cbrt(np.e), np.cbrt(x[2])))))))), np.square(np.exp2(np.e))), np.hypot(np.euler_gamma, np.cbrt(np.cbrt(np.cbrt(x[1]))))), np.cbrt(x[1])), np.square(np.hypot(np.hypot(np.multiply(np.tanh(np.add(np.cbrt(np.sin(x[0])), np.cbrt(np.tanh(np.add(np.cbrt(x[1]), np.cbrt(x[2])))))), np.square(np.cosh(np.negative(np.e)))), x[1]), np.cbrt(np.hypot(np.hypot(np.multiply(np.e, np.square(np.add(np.pi, np.cbrt(np.cbrt(np.round(x[0])))))), x[1]), np.pi))))), x[1]), np.pi)))))), np.cbrt(np.e)), np.cbrt(np.hypot(np.euler_gamma, np.cbrt(np.cbrt(np.cbrt(x[1])))))), np.multiply(np.tanh(np.exp(np.add(np.multiply(np.sin(np.add(np.hypot(np.hypot(np.multiply(np.tanh(np.exp2(np.exp2(np.e))), np.square(np.exp2(np.e))), x[2]), np.multiply(x[0], np.absolute(np.e))), np.e)), np.add(np.exp2(np.e), np.round(np.e))), np.cbrt(np.cosh(np.negative(np.e)))))), np.hypot(np.multiply(np.tanh(np.exp(np.multiply(np.sin(np.add(x[0], np.cbrt(x[2]))), np.add(np.sin(np.cbrt(x[2])), np.round(x[1]))))), np.square(np.exp2(np.round(np.e)))), np.log2(np.tanh(np.euler_gamma)))))))


def f3(x: np.ndarray) -> np.ndarray: 
     #nrestart nrestarts=10   , no run until plateaux
     #   nstep=50000
     """ min_mse no scaling
     3.5762604488319454 """


     return np.add(np.add(np.multiply(np.add(np.sinc(np.e), np.negative(np.square(x[1]))), x[1]), np.square(np.hypot(x[0], x[0]))), np.add(np.add(np.negative(x[2]), np.negative(x[2])), np.absolute(np.absolute(np.negative(np.multiply(np.cbrt(np.reciprocal(np.multiply(np.negative(np.arctan(np.negative(np.multiply(np.cbrt(np.negative(np.sinc(np.e))), np.square(np.arctan(x[2])))))), np.square(np.sinc(np.e))))), np.square(np.arctan(np.remainder(np.sin(np.sinc(np.e)), x[2])))))))))
     
     
     
     #always crossover , mutation of 0.05
     #1/3 ble
     #8.006929603960224   mse   using scaled npenalty and hardlimi ton depth after crossover of 1000 (otherwishe it would generate  anew one )
     return np.add(np.multiply(np.negative(x[1]),
                                np.multiply(x[1], x[1])), np.hypot(np.add(np.multiply(np.negative(np.hypot(np.hypot(x[0], x[0]), np.pi)), 
                                                                                                        np.hypot(np.hypot(np.log1p(np.hypot(
                                                                                                             np.negative(
                                                                                                                  np.exp2(np.add(np.multiply(np.negative(np.hypot(np.hypot(x[0], x[0]), 
                                                                                                                                                                  np.hypot(np.hypot(x[0], x[0]), np.pi))), 
                                                                                                                                                                  np.log1p(np.exp2(np.log1p(np.exp2(x[2]))))), 
                                                                                                                                                                  np.multiply(np.pi, x[2])))), np.log1p(np.pi))),
                                                                                                                                                                    np.negative(x[0])), x[0])), np.multiply(np.pi, x[2])), 
                                                                                                                                                                    np.exp2(np.negative(np.exp2(x[2])))))




     """ np.add(np.add(np.add(np.negative(np.sin(np.negative(x[1]))), np.round(np.hypot(np.multiply(np.hypot(np.multiply(x[0], x[0]), np.hypot(np.remainder(np.negative(np.e), x[1]), x[0])), np.arctan(np.multiply(np.pi, x[0]))), x[2]))), np.add(np.negative(x[2]), np.add(np.negative(x[2]), np.round(np.hypot(np.multiply(np.hypot(np.multiply(np.pi, x[0]), np.e), np.arctan(np.sqrt(np.cosh(np.negative(np.sin(np.exp(np.arctan(np.hypot(np.multiply(np.pi, x[0]), x[0]))))))))), np.remainder(np.negative(x[1]), np.negative(np.cosh(np.tanh(x[2]))))))))), np.add(np.negative(np.sinh(x[1])), np.multiply(np.hypot(np.hypot(x[1], x[1]), np.hypot(x[1], x[1])), np.negative(np.add(x[1], np.tanh(x[2]))))))
     min_mse no scaling
     16.07115297792266 """


     #14.466307622905152
     return np.add(np.multiply(np.multiply(x[1], x[1]), np.negative(x[1])), np.hypot(np.add(np.negative(np.hypot(np.add(np.negative(np.hypot(np.pi, x[0])), x[2]), x[2])), x[2]), np.hypot(np.add(np.negative(np.hypot(np.negative(np.add(np.negative(np.hypot(np.pi, x[0])), x[2])), x[2])), x[2]), np.multiply(np.hypot(x[0], np.hypot(x[0], np.add(np.negative(np.hypot(np.hypot(x[0], x[0]), x[2])), x[2]))), x[0]))))
     
     
     #mse 175.38503573732595
     return np.add(np.hypot(np.power(np.exp(np.sin(np.hypot(x[1], np.round(np.cosh(np.round(np.e)))))), x[1]), np.add(np.hypot(np.add(np.hypot(np.i0(x[0]), np.hypot(np.exp(np.sinh(np.log1p(np.exp(np.exp(np.sin(np.hypot(x[1], np.e))))))), np.hypot(np.add(np.power(np.absolute(x[1]), np.add(np.log1p(np.exp(np.sin(x[1]))), np.log1p(np.exp(np.sinh(np.log1p(np.exp(np.sin(np.log1p(np.log1p(np.power(np.exp(np.sin(np.hypot(x[1], np.e))), x[1]))))))))))), np.i0(x[0])), np.i0(x[0])))), np.log(np.power(np.power(np.exp(np.sin(np.hypot(x[1], np.cosh(np.round(np.pi))))), x[1]), np.exp(np.sinh(np.log1p(np.exp(np.log1p(np.log1p(np.i0(x[0])))))))))), np.hypot(np.power(np.exp(np.sin(np.hypot(x[1], np.e))), x[1]), np.i0(x[0]))), np.log(np.power(np.exp(np.sin(np.hypot(x[1], np.e))), x[1])))), np.log(np.power(np.power(np.power(np.exp(np.sin(np.sin(x[1]))), x[1]), x[1]), x[1])))


def f4(x: np.ndarray) -> np.ndarray: #as before  migliorabile
     """ min_mse no scaling  using run until plateaux
     0.09592352523610304 """

     return np.multiply(np.sinc(np.multiply(x[1], np.multiply(np.exp(np.negative(np.sin(np.sin(np.add(np.pi, np.euler_gamma))))), np.log10(np.negative(np.sin(np.sin(np.add(np.pi, np.euler_gamma)))))))), np.hypot(np.multiply(x[1], np.pi), np.hypot(np.add(np.add(np.add(np.pi, np.i0(x[1])), np.euler_gamma), np.add(np.pi, np.exp(np.euler_gamma))), np.pi)))


     #min_mse
     #0.3802252366701618


     return np.add(np.sin(np.add(np.absolute(x[1]), np.hypot(np.cosh(np.log2(np.round(np.round(np.hypot(np.cosh(np.log2(np.euler_gamma)), np.cosh(np.log2(np.euler_gamma))))))), np.log2(np.euler_gamma)))), np.round(np.add(np.sin(np.add(np.absolute(x[1]), np.round(np.hypot(np.cosh(np.log2(np.euler_gamma)), np.log2(np.euler_gamma))))), np.round(np.add(np.sin(np.i0(np.round(np.add(np.absolute(x[1]), np.log1p(np.euler_gamma))))), np.add(np.sin(np.add(np.i0(np.round(x[1])), np.euler_gamma)), np.round(np.add(np.sin(np.add(np.i0(np.round(np.absolute(x[1]))), np.log1p(np.euler_gamma))), np.round(np.i0(np.round(np.add(np.sin(np.log1p(np.tanh(np.sin(np.add(np.i0(np.round(x[1])), np.log1p(np.add(np.absolute(x[1]), np.log1p(np.tanh(np.sin(np.euler_gamma)))))))))), np.round(np.round(np.hypot(np.cosh(np.log2(np.euler_gamma)), np.log2(np.euler_gamma))))))))))))))))
     
     #4.653280870773529
     return np.multiply(np.cosh(np.pi), np.power(np.reciprocal(np.i0(np.cosh(x[1]))), np.log1p(np.cosh(x[1]))))
def trying_f5_rescaled(x:np.ndarray,offset,mul,y:np.ndarray):
     yset=np.add(np.exp2(np.tanh(np.square(np.hypot(np.sin(np.hypot(np.hypot(np.pi, x[1]), np.e)), np.tanh(np.add(np.tanh(x[1]), np.sin(x[0]))))))), np.pi)
     yreset=(yset - np.min(x)) / mul + offset

     ris=sum((a - b) ** 2 for a, b in zip(y, yset)) / len(y)
     print("real result:")
     print(ris)
     return yreset
 

     

def f5(x: np.ndarray) -> np.ndarray: #is zero so it's good

     #using rescaling 15000 steps and run until the plateaux 
     #MSE: 5.512898055429115e-19
     return np.add(np.divide(np.add(np.sin(np.hypot(np.power(np.hypot(np.power(np.hypot(np.power(x[0], np.negative(np.sin(x[0]))), np.euler_gamma), np.negative(np.sin(np.round(x[1])))), np.arctan(x[0])), np.negative(np.sin(x[1]))), np.arctan(np.pi))), np.add(np.tanh(np.i0(np.tanh(np.power(np.exp2(x[1]), np.sin(x[0]))))), np.pi)), 174318497.75735644), -2.8520706810421616e-08)


     #using rescaling  15000 steps
     #6.229062672792087e-19
     return np.add(np.divide(np.add(np.add(np.cbrt(np.log1p(np.cbrt(np.log1p(np.round(np.cbrt(np.reciprocal(np.power(np.hypot(np.round(np.euler_gamma), np.log10(np.i0(x[1]))), x[0])))))))), np.pi), np.remainder(np.reciprocal(np.power(np.hypot(np.round(np.euler_gamma), np.log10(np.reciprocal(np.power(np.hypot(np.round(np.euler_gamma), np.log10(np.i0(np.power(np.hypot(np.round(np.round(np.euler_gamma)), np.log10(np.i0(x[1]))), x[0])))), x[0])))), x[0])), np.power(np.pi, np.e))), 174318497.75735644), -2.8520706810421616e-08)

     #rescaling ,with final long run 5 43start 10000 steps
     #fitness= mse_val+(length_penalty(len(nodefun),1000,maxy*sizey/(50000*2))) /div
     """     min_mse: 
     1.3036406406056244e-18 """
     return np.add(np.divide(np.multiply(np.power(np.power(np.tanh(np.e), x[0]), np.round(np.reciprocal(np.multiply(np.cosh(np.power(np.power(np.tanh(np.e), x[0]), np.round(np.reciprocal(np.multiply(np.cosh(np.tanh(np.round(np.round(np.euler_gamma)))), np.power(np.multiply(np.cosh(np.reciprocal(np.cosh(np.log2(np.e)))), np.euler_gamma), np.round(np.reciprocal(np.multiply(np.cosh(np.tanh(np.cosh(np.log2(np.e)))), np.power(np.multiply(np.cosh(np.reciprocal(np.cosh(np.log2(np.cosh(np.log2(x[0])))))), np.euler_gamma), x[1])))))))))), np.power(np.multiply(np.cosh(np.reciprocal(np.round(np.add(np.cosh(np.log2(np.cosh(np.log2(np.cosh(np.log2(x[0])))))), np.euler_gamma)))), np.euler_gamma), x[1]))))), np.add(np.cosh(np.log2(np.e)), np.round(np.round(np.e)))), 174318497.75735644), -2.8520706810421616e-08)






     #trying rescaling
     #with low oenalty on fitness for long and no final long step and 20 restarts adn 5000 speps each run

     #min_mse 1.4526349340721968e-18
     return np.add(np.divide(np.add(np.pi, np.multiply(np.absolute(np.cosh(np.euler_gamma)), np.exp2(np.sin(np.log1p(np.round(np.hypot(np.add(np.round(np.hypot(np.power(np.multiply(np.pi, np.sin(np.log1p(np.euler_gamma))), np.round(np.add(np.log2(np.tanh(np.hypot(np.power(np.multiply(np.pi, np.sin(np.log2(np.tanh(np.hypot(np.add(np.round(np.hypot(np.add(np.tanh(np.cosh(np.log1p(x[1]))), x[1]), np.cosh(np.log1p(np.pi)))), np.cosh(np.hypot(np.cosh(np.pi), np.euler_gamma))), np.add(np.euler_gamma, np.pi)))))), np.e), np.log10(np.add(np.hypot(np.square(np.log1p(np.e)), np.exp2(np.e)), np.round(np.power(np.cosh(np.log1p(x[1])), np.cosh(np.log1p(x[1]))))))))), np.multiply(np.cosh(np.log10(np.add(np.add(np.log2(np.tanh(np.hypot(np.power(np.multiply(np.pi, np.sin(np.log1p(x[1]))), np.round(np.add(np.log2(np.e), np.multiply(np.cosh(np.euler_gamma), np.e)))), np.add(np.euler_gamma, np.pi)))), np.multiply(np.cosh(np.log10(np.add(np.hypot(np.square(np.log1p(np.e)), np.exp2(np.e)), np.round(np.power(np.cosh(np.log1p(x[1])), x[0]))))), np.e)), np.round(np.power(np.cosh(np.log1p(x[1])), x[0]))))), np.e)))), np.add(np.euler_gamma, np.pi))), np.cosh(np.hypot(np.cosh(np.log1p(np.pi)), np.euler_gamma))), np.hypot(np.cosh(np.log1p(np.pi)), np.euler_gamma)))))))), 174318497.75735644), -2.8520706810421616e-08)




     #with low oenalty on fitness for long 

     #.1 *10^-8
     return np.multiply(np.pi, np.hypot(np.round(np.tanh(np.tanh(np.add(np.power(np.tanh(np.round(np.sin(np.round(np.sin(x[0]))))), np.round(np.sin(np.round(np.sin(x[0]))))), np.hypot(np.power(np.add(np.power(np.tanh(np.log(np.multiply(np.pi, np.hypot(np.round(np.tanh(np.round(x[1]))), np.power(np.tanh(np.tanh(np.log(np.e))), np.euler_gamma))))), np.cosh(x[1])), np.round(np.power(np.round(np.tanh(x[0])), np.round(np.euler_gamma)))), np.round(np.cosh(x[1]))), np.power(np.cosh(np.sin(np.tanh(np.add(np.power(np.power(np.sin(np.sin(x[0])), np.round(np.e)), np.round(np.power(np.sin(x[0]), np.round(np.power(np.sin(np.e), np.round(np.cosh(x[1]))))))), np.round(np.power(np.sin(np.round(np.sin(x[0]))), np.round(np.cosh(np.round(np.euler_gamma))))))))), np.round(np.e))))))), np.power(np.cosh(np.sin(np.tanh(np.add(np.power(np.power(np.sin(x[0]), np.round(np.i0(np.power(np.tanh(np.cosh(x[1])), np.round(np.euler_gamma))))), np.round(np.power(np.sin(x[0]), np.round(np.power(np.sin(np.sin(np.tanh(np.round(np.round(np.power(np.euler_gamma, np.round(np.power(np.sin(x[1]), np.round(np.cosh(x[1])))))))))), np.round(np.e)))))), np.round(np.power(np.add(np.power(np.tanh(np.log(np.multiply(np.pi, np.hypot(np.round(np.tanh(np.log(np.e))), np.add(np.power(np.power(np.sin(x[0]), np.round(np.tanh(np.tanh(np.add(np.power(np.tanh(np.round(np.sin(np.round(np.sin(x[0]))))), np.round(np.sin(np.round(np.sin(x[0]))))), np.hypot(np.power(np.add(np.power(np.tanh(np.log(np.multiply(np.pi, np.hypot(np.round(np.tanh(np.round(np.tanh(np.euler_gamma)))), np.power(np.tanh(np.tanh(np.i0(np.euler_gamma))), np.euler_gamma))))), np.cosh(x[1])), np.round(np.power(np.round(np.tanh(np.euler_gamma)), np.round(np.euler_gamma)))), np.round(np.cosh(x[1]))), np.power(np.cosh(np.sin(np.tanh(np.add(np.power(np.power(np.sin(np.sin(x[0])), np.round(np.e)), np.round(np.round(np.euler_gamma))), np.round(np.power(np.sin(np.round(np.sin(x[0]))), np.round(np.cosh(np.round(np.i0(np.euler_gamma)))))))))), np.round(np.e)))))))), np.round(np.tanh(np.add(np.power(np.round(np.tanh(np.sin(np.tanh(np.round(np.round(np.e)))))), np.round(np.power(np.e, np.round(np.power(np.sin(np.e), np.round(np.tanh(np.cosh(x[1])))))))), np.round(np.e))))), np.round(np.power(np.add(np.power(np.tanh(np.log(np.multiply(np.pi, np.hypot(np.round(np.tanh(np.round(np.e))), np.power(np.cosh(np.round(np.power(np.tanh(np.log(np.multiply(np.pi, np.hypot(np.round(np.e), np.power(np.cosh(np.round(np.tanh(np.power(np.round(np.sin(x[0])), np.round(np.round(np.tanh(np.round(np.e)))))))), np.power(np.tanh(np.round(x[0])), np.round(x[0]))))))), np.cosh(x[1])))), np.power(np.tanh(np.round(x[0])), np.round(x[0]))))))), np.cosh(x[1])), np.round(np.round(np.tanh(np.euler_gamma)))), np.round(np.round(np.e))))))))), np.cosh(x[1])), np.round(np.power(np.tanh(np.round(x[0])), np.round(x[0])))), np.round(np.e))))))), np.euler_gamma)))
     #2.529315024476992e-18
     #np.add(np.exp2(np.tanh(np.square(np.hypot(np.sin(np.hypot(np.hypot(np.pi, x[1]), np.e)), np.tanh(np.add(np.tanh(x[1]), np.sin(x[0]))))))), np.pi)



     #min_mse
     #5.572809570693574e-18

     return np.multiply(np.sin(np.pi), np.negative(x[0]))


def f6(x: np.ndarray) -> np.ndarray:   #migliorabile

     # popsize 1500 , using suggested overselections parameters for overpopulation  #and no run_until_plateaux
     #7.145589969138924e-07

     return np.add(np.multiply(np.arctan(np.cbrt(np.euler_gamma)), np.add(x[1], np.negative(x[0]))), x[1])





     #popsize 1000 , using suggested overselections parameters for overpopulation  #and no run_until_plateaux
     """ min_mse no scaling
     0.0012425233639984457 """


     return np.add(np.multiply(np.log10(np.log10(np.log(np.absolute(np.add(np.cosh(np.log1p(np.exp2(np.tanh(np.e)))), np.pi))))), x[0]), np.multiply(x[1], np.log1p(np.add(np.cosh(np.tanh(np.tanh(np.cosh(np.pi)))), np.pi)))) 
     


     ###changed using pop 2000 and mor eoperands 
     """ min_mse no scaling
     0.003157777848447691 """
     return np.add(np.remainder(np.multiply(x[1], np.e), x[1]), np.add(x[1], np.multiply(np.exp2(np.remainder(np.euler_gamma, np.arctan(np.multiply(np.log2(np.tanh(np.arctan(np.sin(np.cosh(np.euler_gamma))))), np.pi)))), np.negative(x[0]))))
     """      min_mse no scaling
     0.00251533612410414 """
     return np.add(np.add(np.multiply(np.log10(np.euler_gamma), x[0]), np.multiply(x[1], np.tanh(np.exp(np.euler_gamma)))), np.add(np.multiply(np.log10(np.square(np.euler_gamma)), x[0]), np.multiply(x[1], np.sin(np.sin(np.tanh(np.pi))))))


     #long run using the while loop , div 10 fitness no scaling 
     """ min_mse no scaling
     0.03557585723525194 """



     return np.add(x[1], np.multiply(np.hypot(x[0], np.log10(np.i0(np.add(x[1], np.multiply(np.hypot(x[0], np.hypot(np.add(x[1], np.multiply(np.hypot(np.tanh(x[1]), x[1]), np.tanh(np.negative(np.tanh(x[0]))))), np.hypot(np.multiply(np.pi, np.tanh(np.tanh(np.tanh(np.pi)))), np.pi))), np.negative(np.tanh(np.tanh(np.negative(np.add(x[0], np.add(np.multiply(np.pi, np.negative(np.tanh(np.tanh(x[0])))), x[1]))))))))))), np.negative(np.tanh(np.negative(np.add(x[1], np.multiply(np.hypot(np.tanh(np.add(x[1], np.hypot(np.tanh(np.pi), np.add(x[1], np.tanh(np.tanh(x[0])))))), np.multiply(x[1], np.tanh(x[0]))), np.negative(np.tanh(x[0])))))))))
     
     
     #long run using the while loop , div 1 fitness no scaling 
     #0.05452263550185847

     return np.add(np.multiply(np.tanh(np.add(np.tanh(x[0]), np.add(np.absolute(np.multiply(x[1], np.exp2(np.tanh(np.tanh(x[1]))))), np.add(np.tanh(x[1]), np.negative(x[0]))))), np.log1p(np.absolute(np.add(np.sin(x[0]), np.add(np.absolute(np.multiply(x[1], np.exp2(np.tanh(x[1])))), np.add(np.tanh(x[1]), np.negative(x[0]))))))), np.add(np.tanh(np.negative(x[0])), np.log1p(np.tanh(x[1]))))



     """ min_mse
          0.18331680059707295 """

     return np.multiply(np.add(np.tanh(np.multiply(x[0], np.log2(np.tanh(np.exp2(np.log10(np.euler_gamma)))))), x[1]), np.exp2(np.exp2(np.log10(np.log10(np.hypot(np.hypot(np.square(np.hypot(np.log2(np.tanh(np.tanh(np.i0(x[0])))), np.i0(np.log2(np.tanh(np.euler_gamma))))), np.euler_gamma), np.i0(x[0])))))))


def f7(x: np.ndarray) -> np.ndarray: 
     """ min_mse no scaling   using run until plateaux
     64.10785620740353 """

     return np.hypot(np.pi, np.multiply(np.add(np.log2(np.power(np.exp(x[1]), np.sin(x[0]))), np.hypot(np.multiply(np.hypot(np.exp(np.pi), np.power(np.exp(x[1]), x[0])), np.multiply(np.sinc(np.hypot(np.multiply(np.sinc(np.multiply(np.pi, np.tanh(np.tanh(np.multiply(np.sinc(np.hypot(np.i0(np.negative(np.log2(np.power(np.exp(x[1]), np.sin(np.log(np.hypot(np.i0(x[1]), np.power(np.hypot(x[0], np.exp(np.pi)), np.sin(np.log(np.hypot(np.i0(np.negative(np.log2(np.power(np.exp(x[1]), np.sin(np.log(np.hypot(x[1], np.power(np.hypot(np.arctan(np.pi), np.exp(np.pi)), np.sin(np.log(np.hypot(x[0], np.arctan(np.round(np.arctan(np.exp(np.pi))))))))))))))), np.arctan(x[1])))))))))))), np.multiply(np.arctan(np.exp(np.hypot(np.i0(np.hypot(x[0], np.absolute(np.sin(x[0])))), np.sin(x[1])))), np.absolute(np.sin(x[1]))))), np.round(np.pi)))))), x[0]), np.round(np.i0(np.negative(np.log2(np.power(np.exp(x[1]), np.sin(np.log(np.hypot(x[1], np.power(np.hypot(np.i0(x[1]), x[1]), np.sin(np.log(np.hypot(np.hypot(np.i0(x[1]), x[1]), np.arctan(x[0]))))))))))))))), np.hypot(x[1], np.power(np.hypot(np.sinc(np.exp(np.pi)), np.multiply(np.negative(np.log2(np.power(np.exp(np.negative(np.hypot(np.i0(x[1]), x[1]))), np.sin(np.power(np.exp(x[1]), np.sin(x[0])))))), np.sinc(np.multiply(np.pi, np.multiply(np.sinc(np.hypot(np.i0(np.negative(np.log2(np.power(np.exp(x[0]), np.sin(np.log(np.hypot(np.arctan(np.absolute(np.round(np.arctan(x[1])))), np.power(np.hypot(x[1], np.exp(np.pi)), np.sin(np.log(np.hypot(np.i0(np.negative(np.log2(np.power(np.exp(x[1]), np.sin(np.log(np.hypot(np.hypot(x[0], x[0]), np.power(np.hypot(np.exp(x[1]), np.exp(np.pi)), np.sin(np.log(np.hypot(x[0], np.arctan(np.round(np.arctan(np.exp(np.pi))))))))))))))), np.arctan(x[1])))))))))))), np.multiply(np.arctan(np.exp(np.hypot(np.i0(x[1]), np.arctan(x[1])))), np.sin(x[0])))), np.negative(np.log2(np.power(np.exp(np.negative(np.log2(np.power(np.exp(np.hypot(x[0], x[0])), np.arctan(np.pi))))), np.sin(np.power(np.exp(x[1]), np.sin(x[0]))))))))))), np.arctan(np.exp(np.pi)))))), np.round(np.arctan(np.log2(np.sin(np.absolute(np.arctan(x[1])))))))), np.hypot(np.power(np.exp(x[1]), np.sin(np.log2(np.power(np.exp(x[0]), np.sin(np.log(np.hypot(np.arctan(x[1]), np.power(np.hypot(np.arctan(np.exp(np.log2(np.exp(x[1])))), np.exp(np.pi)), np.sin(np.log(np.hypot(np.i0(np.negative(np.log2(np.power(np.exp(x[1]), np.sin(x[1]))))), np.arctan(np.i0(x[1]))))))))))))), np.power(np.exp(x[1]), x[0]))))

     """ even better 
     np.hypot(np.pi, np.multiply(np.add(np.log2(np.power(np.exp(x[1]), np.sin(x[0]))), np.hypot(np.multiply(np.hypot(np.exp(np.pi), np.power(np.exp(x[1]), x[0])), np.multiply(np.sinc(np.hypot(np.multiply(np.sinc(np.multiply(np.pi, np.tanh(np.tanh(np.multiply(np.sinc(np.hypot(np.i0(np.negative(np.log2(np.power(np.exp(x[1]), np.sin(np.log(np.hypot(np.i0(x[1]), np.power(np.hypot(x[0], np.exp(np.pi)), np.sin(np.log(np.hypot(np.i0(np.negative(np.log2(np.power(np.exp(x[1]), np.sin(np.log(np.hypot(x[1], np.power(np.hypot(np.arctan(np.pi), np.exp(np.pi)), np.sin(np.log(np.hypot(x[0], np.arctan(np.round(np.arctan(np.exp(np.pi))))))))))))))), np.arctan(x[1])))))))))))), np.multiply(np.arctan(np.exp(np.hypot(np.i0(np.hypot(x[0], np.absolute(np.sin(x[0])))), np.sin(x[1])))), np.absolute(np.sin(x[1]))))), np.round(np.pi)))))), x[0]), np.round(np.i0(np.negative(np.log2(np.power(np.exp(x[1]), np.sin(np.log(np.hypot(x[1], np.power(np.hypot(np.i0(x[1]), x[1]), np.sin(np.log(np.hypot(np.hypot(np.i0(x[1]), x[1]), np.arctan(x[0]))))))))))))))), np.hypot(x[1], np.power(np.hypot(np.sinc(np.exp(np.pi)), np.multiply(np.negative(np.log2(np.power(np.exp(np.negative(np.hypot(np.i0(x[1]), x[1]))), np.sin(np.power(np.exp(x[1]), np.sin(x[0])))))), np.sinc(np.multiply(np.pi, np.multiply(np.sinc(np.hypot(np.i0(np.negative(np.log2(np.power(np.exp(x[0]), np.sin(np.log(np.hypot(np.arctan(np.absolute(np.round(np.arctan(x[1])))), np.power(np.hypot(np.arctan(np.pi), np.exp(np.pi)), np.sin(np.log(np.hypot(np.i0(np.negative(np.log2(np.power(np.exp(x[1]), np.sin(np.log(np.hypot(np.hypot(x[0], x[0]), np.power(np.hypot(np.exp(x[1]), np.exp(np.pi)), np.sin(np.log(np.hypot(x[0], np.arctan(np.round(np.arctan(np.exp(np.pi))))))))))))))), np.arctan(x[1])))))))))))), np.multiply(np.arctan(np.exp(np.hypot(np.i0(x[1]), np.arctan(x[1])))), np.sin(x[0])))), np.negative(np.log2(np.power(np.exp(np.negative(np.log2(np.power(np.exp(np.hypot(x[0], x[0])), np.arctan(np.pi))))), np.sin(np.power(np.exp(x[1]), np.sin(x[0]))))))))))), np.arctan(np.exp(np.pi)))))), np.round(np.arctan(np.log2(np.sin(np.absolute(np.sin(x[1])))))))), np.hypot(np.power(np.exp(x[1]), np.sin(np.log2(np.power(np.exp(x[0]), np.sin(np.log(np.hypot(np.arctan(x[1]), np.power(np.hypot(np.arctan(np.exp(np.log2(np.exp(x[1])))), np.exp(np.pi)), np.sin(np.log(np.hypot(np.i0(np.negative(np.log2(np.power(np.exp(x[1]), np.sin(x[1]))))), np.arctan(np.i0(x[1]))))))))))))), np.power(np.exp(x[1]), x[0]))))
     #should be good , calc later
     return np.hypot(np.multiply(np.power(np.i0(np.sinh(np.cbrt(x[1]))), np.hypot(np.multiply(np.add(x[1], x[0]), x[0]), np.sin(x[0]))), np.add(np.cbrt(np.sin(np.sin(x[0]))), x[1])), np.hypot(np.multiply(np.power(np.i0(np.multiply(np.tanh(np.multiply(np.multiply(np.tanh(np.sin(np.cosh(np.multiply(np.multiply(np.tanh(np.sin(x[0])), np.sin(np.hypot(np.round(np.add(np.cbrt(x[1]), x[1])), np.add(np.multiply(np.tanh(np.multiply(np.tanh(np.multiply(np.tanh(np.cbrt(x[0])), np.e)), np.sin(np.cosh(np.tanh(np.sin(np.cosh(np.negative(np.multiply(np.multiply(np.tanh(np.sin(np.sin(x[1]))), np.sin(np.sin(x[1]))), np.cosh(np.multiply(np.tanh(np.multiply(np.tanh(x[0]), np.sin(x[1]))), np.e))))))))))), np.e), x[0])))), np.cosh(np.multiply(np.tanh(np.tanh(np.multiply(np.tanh(np.multiply(np.e, np.e)), np.sin(np.cosh(np.sin(np.cosh(np.multiply(np.multiply(np.tanh(np.sin(x[0])), np.sin(np.hypot(np.round(np.add(np.cbrt(np.tanh(np.add(np.cbrt(np.sin(np.add(x[1], x[0]))), x[0]))), x[1])), np.add(np.multiply(np.tanh(np.sin(np.sin(x[1]))), np.e), x[0])))), np.cosh(np.multiply(np.tanh(np.tanh(np.tanh(x[0]))), np.e)))))))))), np.e)))))), np.sin(np.cosh(np.multiply(np.multiply(np.tanh(np.sin(x[1])), np.sin(np.hypot(np.round(np.add(np.cbrt(np.round(np.tanh(np.tanh(np.sin(np.sin(np.tanh(np.sin(x[1])))))))), x[0])), np.add(np.multiply(np.tanh(np.tanh(np.multiply(np.tanh(np.multiply(np.multiply(np.hypot(np.round(x[0]), np.add(np.sin(np.multiply(x[1], np.cosh(np.round(np.add(np.sin(x[1]), x[0]))))), np.sin(np.sin(x[1])))), x[0]), np.sin(np.tanh(np.sin(np.cbrt(np.add(np.cbrt(x[1]), x[1]))))))), np.tanh(np.add(np.cbrt(np.multiply(np.tanh(x[1]), np.e)), x[1]))))), np.e), x[0])))), np.cosh(np.multiply(np.tanh(np.multiply(np.tanh(x[0]), np.sin(x[1]))), np.e)))))), np.cbrt(x[1]))), np.e)), np.hypot(np.round(np.add(np.sin(x[1]), x[0])), np.add(np.multiply(np.tanh(np.multiply(np.tanh(np.tanh(np.multiply(x[0], np.sin(np.tanh(np.multiply(x[0], np.sin(np.add(np.cbrt(np.round(np.tanh(np.tanh(np.sin(np.sin(np.tanh(np.sin(x[0])))))))), x[1])))))))), np.e)), np.cosh(np.multiply(np.tanh(np.tanh(np.multiply(x[0], np.sin(np.tanh(np.multiply(x[0], np.sin(np.add(np.cbrt(np.round(np.tanh(np.tanh(np.sin(np.sin(np.tanh(np.sin(x[1])))))))), x[1])))))))), np.e))), x[0]))), np.cosh(np.multiply(np.multiply(np.tanh(np.tanh(np.multiply(np.add(x[1], x[0]), np.add(np.multiply(np.tanh(np.sin(x[1])), np.e), x[0])))), np.sin(x[1])), np.cosh(np.multiply(np.tanh(np.multiply(np.tanh(np.tanh(np.multiply(np.round(np.sin(np.sin(x[1]))), np.add(np.multiply(np.sin(x[0]), np.e), x[0])))), np.sin(np.multiply(np.tanh(np.multiply(np.tanh(np.multiply(x[0], np.sin(np.tanh(np.multiply(x[0], np.sin(np.add(np.sin(x[1]), x[1]))))))), np.sin(np.cosh(np.multiply(np.multiply(np.tanh(np.tanh(np.multiply(np.tanh(np.cbrt(x[0])), np.e))), np.sin(np.hypot(np.round(np.add(np.cbrt(np.tanh(np.sin(np.sin(x[0])))), x[1])), np.add(np.multiply(np.tanh(np.sin(np.multiply(x[0], np.sin(np.tanh(np.multiply(x[0], np.sin(np.add(np.sin(x[1]), x[1])))))))), np.e), x[0])))), np.e))))), np.e)))), np.e))))), np.add(np.add(np.multiply(np.sin(np.cbrt(np.multiply(np.tanh(np.sin(x[1])), np.e))), np.e), x[0]), x[0])))
     #long run using the while loop , div 10 fitness no scaling 
     #186.37582307610336


     return np.hypot(np.power(np.tanh(np.tanh(np.tanh(np.tanh(np.tanh(np.exp(x[1])))))), np.round(np.add(np.add(np.round(np.tanh(x[1])), np.tanh(np.add(np.add(np.add(np.add(np.power(np.tanh(np.exp(x[1])), np.round(np.add(np.add(np.add(np.add(np.power(np.exp(x[1]), np.round(np.tanh(np.add(np.add(np.add(np.add(np.power(np.exp(x[1]), np.round(np.tanh(x[1]))), x[0]), x[0]), np.round(x[0])), x[0])))), x[1]), x[0]), x[0]), x[0]))), x[0]), x[0]), np.add(np.add(np.add(np.round(np.tanh(x[1])), np.tanh(np.add(np.power(np.exp(x[0]), x[0]), x[0]))), np.tanh(np.tanh(np.tanh(np.tanh(np.tanh(np.round(np.tanh(np.round(np.tanh(np.tanh(np.add(np.tanh(np.power(np.exp(x[1]), np.round(x[1]))), x[0])))))))))))), np.tanh(x[1]))), x[0]))), np.tanh(x[0])))), np.hypot(np.multiply(np.multiply(np.power(np.exp(x[1]), x[0]), x[0]), x[1]), np.hypot(np.add(np.power(np.exp(np.tanh(x[1])), np.add(np.add(np.add(np.round(np.tanh(np.tanh(np.tanh(np.tanh(x[0]))))), np.tanh(np.add(np.add(np.add(np.add(np.power(np.exp(np.add(np.add(np.add(np.power(np.exp(x[1]), np.round(np.tanh(np.add(np.add(np.add(np.add(np.power(np.exp(x[1]), np.round(np.tanh(x[1]))), x[1]), x[0]), x[0]), x[0])))), np.round(x[0])), x[0]), x[0])), x[0]), np.tanh(x[1])), x[0]), x[0]), np.tanh(x[0])))), np.tanh(x[0])), np.tanh(x[0]))), np.tanh(np.power(np.exp(np.tanh(x[1])), x[0]))), np.hypot(np.round(np.power(np.exp(np.round(x[1])), np.round(np.round(x[0])))), np.hypot(np.tanh(np.exp(np.round(x[0]))), np.hypot(np.round(np.power(np.tanh(np.tanh(np.tanh(np.tanh(np.exp(x[1]))))), np.round(np.add(np.add(np.round(np.tanh(x[1])), np.tanh(np.add(np.add(np.add(np.add(np.power(np.tanh(np.exp(x[1])), np.round(np.add(np.add(np.add(np.add(np.power(np.exp(x[1]), np.round(np.round(np.tanh(np.add(np.add(np.add(np.add(np.power(np.exp(x[1]), np.round(np.tanh(x[1]))), x[0]), x[0]), x[0]), x[0]))))), x[0]), x[0]), x[0]), x[0]))), x[0]), x[0]), np.add(np.add(np.add(np.round(np.tanh(np.tanh(x[1]))), np.tanh(np.add(np.power(np.exp(x[0]), x[0]), x[0]))), np.tanh(np.tanh(np.tanh(np.tanh(np.tanh(x[1])))))), np.tanh(np.add(np.add(np.add(np.add(np.power(np.exp(x[1]), np.round(np.tanh(x[1]))), x[0]), x[0]), x[1]), x[0])))), x[0]))), np.tanh(x[0]))))), np.tanh(np.tanh(x[0]))))))))
      """

def f8(x: np.ndarray) -> np.ndarray: #good
     #min_mse
     #
     #166579.13617635533
     return np.add(np.multiply(np.add(np.add(x[5], np.round(np.cbrt(np.negative(np.exp2(np.cbrt(np.negative(np.exp2(np.hypot(np.round(np.cbrt(np.negative(np.square(np.arctan(np.add(np.multiply(np.square(np.multiply(x[5], np.add(x[5], x[5]))), np.e), np.cbrt(np.remainder(np.arctan(np.negative(np.hypot(np.euler_gamma, np.square(np.remainder(np.arctan(np.negative(np.exp2(np.hypot(x[4], np.pi)))), np.negative(np.add(x[5], x[5]))))))), np.multiply(np.multiply(x[5], np.add(x[5], x[5])), np.add(x[5], x[5])))))))))), np.pi))))))))), x[5]), np.multiply(np.hypot(np.e, x[5]), np.multiply(x[5], np.add(x[5], x[5])))), np.add(np.multiply(np.square(np.multiply(x[5], np.add(x[5], x[5]))), x[5]), np.negative(np.square(np.hypot(np.exp2(np.hypot(np.hypot(np.euler_gamma, np.pi), x[4])), np.round(x[5]))))))

    #run until plateaux , no restarts , no scaling
     #mse 371401.4112070473
     return np.add(np.multiply(np.multiply(np.add(x[5], np.sin(np.hypot(np.sin(x[5]), np.exp(np.arctan(np.hypot(np.absolute(np.exp(np.sinh(np.sinh(np.sinh(np.sinh(np.sin(np.hypot(np.sinh(np.sinh(np.hypot(np.sinh(np.sin(np.hypot(np.e, np.hypot(np.hypot(np.sqrt(np.arctan(np.e)), x[4]), np.exp(np.sqrt(np.sinh(np.arctan(np.e)))))))), np.sin(np.hypot(np.sin(x[4]), np.exp(np.e)))))), np.hypot(x[5], np.arctan(np.absolute(np.exp(np.arctan(x[5]))))))))))))), np.sinh(np.hypot(np.absolute(np.exp(np.arctan(x[5]))), np.sinh(np.sinh(np.sin(np.hypot(np.sinh(np.sin(np.sqrt(np.e))), np.hypot(x[5], np.exp(np.sqrt(np.arctan(np.e)))))))))))))))), np.cosh(np.absolute(np.hypot(np.sin(np.hypot(np.sinh(np.arctan(np.exp(np.cosh(x[5])))), np.hypot(np.hypot(np.sin(np.exp(np.cbrt(x[5]))), np.exp(np.arctan(np.hypot(x[5], np.sinh(np.exp(x[5])))))), np.exp(np.sqrt(np.sinh(np.hypot(np.sinh(np.sin(np.hypot(np.sinh(np.arctan(np.exp(np.cosh(np.e)))), np.hypot(np.hypot(np.sin(np.sinh(np.e)), x[4]), np.exp(np.sqrt(np.sinh(np.arctan(np.e)))))))), np.sinh(np.arctan(np.arctan(np.e)))))))))), np.hypot(np.sin(np.hypot(np.sinh(np.arctan(np.exp(np.cosh(x[5])))), np.hypot(np.hypot(np.sin(np.e), np.exp(np.arctan(np.hypot(x[5], np.sinh(np.exp(x[5])))))), np.exp(np.hypot(np.sinh(np.sin(np.hypot(np.sinh(np.arctan(np.exp(np.exp(np.cbrt(x[5]))))), np.hypot(np.hypot(np.sin(x[5]), x[4]), np.exp(np.sqrt(np.sinh(np.arctan(np.hypot(np.hypot(np.e, np.hypot(x[5], np.exp(x[5]))), np.exp(np.sqrt(np.e))))))))))), np.sinh(np.arctan(np.arctan(np.e)))))))), np.hypot(np.absolute(np.absolute(np.absolute(np.exp(np.sinh(np.sinh(np.sinh(np.sin(np.sinh(np.sinh(np.sin(np.sinh(np.sinh(np.sin(np.hypot(np.arctan(np.hypot(x[5], np.hypot(np.sinh(np.exp(np.arctan(np.exp(np.cosh(np.e))))), x[5]))), np.hypot(x[5], np.arctan(np.exp(np.sinh(np.hypot(np.sinh(np.sin(np.hypot(np.sinh(np.arctan(np.exp(np.e))), np.hypot(np.hypot(np.sin(np.e), x[4]), np.exp(np.sqrt(np.sinh(np.arctan(np.e)))))))), np.sinh(np.hypot(np.sinh(np.sin(np.hypot(np.hypot(np.sinh(np.arctan(np.e)), np.hypot(x[5], np.exp(x[5]))), np.hypot(np.hypot(np.sin(x[5]), x[4]), np.exp(np.sqrt(np.sinh(np.arctan(np.e)))))))), np.sinh(np.arctan(np.sin(np.hypot(np.sin(x[5]), np.exp(np.arctan(np.hypot(np.absolute(np.exp(x[5])), np.sinh(np.hypot(np.absolute(np.sin(np.exp(x[5]))), x[5]))))))))))))))))))))))))))))))), np.exp(np.sinh(np.arctan(np.e))))))))), np.cosh(x[5])), np.sin(np.sinh(x[5])))

     #restarts without elitims and last run to mix evertythibg up :
     #500888.3890449591

     return np.add(np.multiply(x[5], np.hypot(np.hypot(np.hypot(np.add(np.exp(np.exp(np.euler_gamma)), np.multiply(np.negative(np.hypot(np.exp(np.absolute(x[5])), np.add(np.exp(np.absolute(x[5])), np.absolute(x[5])))), np.exp(np.e))), np.absolute(np.multiply(np.multiply(x[5], x[5]), np.hypot(x[5], x[5])))), np.exp(np.add(np.log(np.e), np.absolute(x[5])))), x[5])), np.negative(np.hypot(np.exp(np.absolute(np.exp(np.exp(np.euler_gamma)))), np.exp(np.log(np.exp(np.exp(np.exp(np.euler_gamma))))))))

     """ min_mse
     574668.6816519526 """
     return np.multiply(x[5], np.hypot(np.square(np.cosh(np.euler_gamma)), np.multiply(x[5], np.hypot(np.multiply(np.negative(np.add(np.multiply(x[5], np.absolute(np.multiply(x[5], np.round(np.add(np.e, np.e))))), np.multiply(np.negative(np.hypot(np.multiply(np.reciprocal(np.square(np.cosh(np.log10(np.hypot(np.square(np.negative(np.square(np.cosh(np.log10(np.hypot(np.square(np.negative(np.add(np.multiply(np.euler_gamma, np.round(np.absolute(x[3]))), np.multiply(np.negative(np.hypot(np.multiply(np.hypot(np.square(np.round(np.round(np.absolute(np.multiply(x[5], np.round(np.add(np.e, np.e))))))), x[5]), np.negative(np.hypot(np.pi, x[3]))), np.euler_gamma)), np.square(np.square(np.log10(np.euler_gamma))))))), x[5])))))), np.add(np.multiply(x[5], np.square(np.log10(np.euler_gamma))), np.multiply(np.multiply(np.euler_gamma, x[1]), np.reciprocal(np.add(np.multiply(np.square(np.log10(np.hypot(np.pi, x[3]))), x[5]), np.multiply(np.negative(np.square(np.log10(np.euler_gamma))), np.square(np.square(x[5])))))))))))), np.negative(np.hypot(x[5], np.hypot(np.multiply(np.reciprocal(np.square(np.cosh(np.log10(np.hypot(np.square(np.negative(np.square(np.cosh(np.log10(np.hypot(np.square(np.negative(np.negative(np.add(np.multiply(np.euler_gamma, np.round(np.absolute(x[3]))), np.multiply(np.negative(np.hypot(np.multiply(np.hypot(np.square(np.round(np.round(np.absolute(np.multiply(x[5], np.round(np.add(np.e, np.e))))))), np.multiply(x[5], np.multiply(np.negative(np.pi), x[5]))), np.negative(np.hypot(np.pi, x[3]))), x[5])), np.square(np.square(np.log10(np.euler_gamma)))))))), x[5])))))), np.add(np.multiply(x[5], np.square(np.log10(np.euler_gamma))), np.multiply(x[1], np.reciprocal(np.absolute(np.multiply(x[5], np.round(np.add(np.e, np.e)))))))))))), np.add(np.e, np.e)), x[3])))), np.square(np.square(np.log10(np.square(np.multiply(x[5], np.square(np.log10(np.euler_gamma))))))))), np.square(np.square(np.log10(np.square(np.log10(np.euler_gamma)))))))), x[5]), np.add(np.multiply(np.euler_gamma, np.round(np.absolute(x[3]))), np.log10(np.euler_gamma))))))


     #restarts_without elitims
     #min_mse

     #733125.2102081886
     return np.add(np.add(np.multiply(x[5], x[5]), np.add(np.add(np.multiply(np.hypot(x[5], np.add(np.add(np.multiply(np.hypot(x[5], x[5]), np.multiply(np.hypot(x[5], x[5]), np.multiply(np.hypot(x[5], x[5]), x[5]))), x[5]), np.multiply(np.hypot(x[5], np.add(np.multiply(np.hypot(x[5], x[5]), np.multiply(np.hypot(x[5], x[5]), x[5])), np.add(np.sin(np.hypot(x[5], np.e)), np.add(x[5], np.multiply(np.sin(x[5]), x[5]))))), x[5]))), x[5]), np.multiply(np.hypot(x[5], x[5]), x[5])), x[5])), np.hypot(x[5], x[5]))

     #1213407.179733729
     return np.multiply(np.cosh(np.negative(x[5])), np.multiply(np.exp2(np.sinh(np.reciprocal(np.sin(np.e)))), x[5]))


     

     return np.multiply(np.add(np.sinh(x[5]), np.sin(np.add(np.pi, np.i0(np.i0(np.i0(np.tanh(np.round(np.round(np.round(np.pi)))))))))), np.round(np.absolute(np.add(np.round(np.absolute(np.add(np.round(np.power(np.multiply(np.tanh(np.absolute(np.add(np.sinh(np.add(x[5], np.sin(np.add(np.pi, np.i0(np.i0(np.i0(np.sin(np.round(np.i0(np.i0(np.i0(np.sin(x[5]))))))))))))), np.sin(np.add(np.pi, np.i0(np.i0(np.i0(np.sin(np.pi))))))))), np.pi), np.add(np.multiply(np.tanh(np.absolute(np.add(np.sinh(np.add(x[5], np.sin(np.add(np.pi, np.i0(np.i0(np.i0(np.sin(x[5])))))))), np.euler_gamma))), np.pi), np.i0(np.i0(np.sin(x[5])))))), np.add(np.pi, np.pi)))), np.power(np.multiply(np.tanh(np.tanh(np.power(np.add(np.sin(x[5]), np.i0(np.sin(np.sin(np.pi)))), np.multiply(np.add(np.sin(x[5]), np.i0(np.sin(np.sin(np.pi)))), np.pi)))), np.pi), np.round(np.round(np.pi)))))))




     """ min_mse
     13383082.027662007 """
     1213407.179733729

     return np.hypot(np.hypot(np.hypot(np.power(np.i0(np.pi), x[5]), np.hypot(np.hypot(np.multiply(np.absolute(np.power(np.pi, x[5])), np.log2(np.hypot(np.power(np.hypot(x[5], np.pi), x[5]), np.power(np.hypot(np.log2(np.round(np.log1p(np.pi))), np.log1p(np.hypot(np.hypot(x[5], np.multiply(np.absolute(np.power(np.hypot(np.log2(np.i0(x[5])), np.hypot(x[5], np.log2(np.round(np.log1p(np.e))))), x[5])), np.power(np.hypot(x[5], np.hypot(x[5], np.log2(np.round(np.log1p(np.pi))))), x[5]))), np.log1p(np.power(np.hypot(x[5], x[5]), x[5]))))), np.hypot(np.i0(np.pi), np.hypot(np.log1p(np.hypot(np.hypot(np.i0(np.i0(np.pi)), np.i0(np.hypot(np.pi, np.hypot(np.log1p(np.pi), np.i0(np.pi))))), np.hypot(np.pi, np.i0(np.log1p(np.i0(np.hypot(np.e, np.i0(np.pi)))))))), np.i0(np.pi))))))), np.power(np.hypot(np.e, np.i0(np.pi)), x[5])), np.power(np.pi, x[5]))), np.power(np.i0(np.log2(np.i0(np.pi))), x[5])), np.multiply(np.absolute(np.power(np.pi, x[5])), np.log2(np.multiply(np.absolute(np.power(np.hypot(np.e, np.i0(np.pi)), x[5])), np.log2(np.hypot(x[5], np.power(np.hypot(np.log1p(np.i0(np.log1p(np.log1p(np.i0(np.log1p(np.pi)))))), np.i0(x[5])), np.hypot(np.log1p(np.pi), np.hypot(np.log1p(np.hypot(np.hypot(x[5], np.power(np.absolute(np.power(np.hypot(np.hypot(np.pi, np.i0(np.pi)), np.pi), x[5])), x[5])), np.pi)), x[5])))))))))
