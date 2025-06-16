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



def f2(x: np.ndarray) -> np.ndarray: 
     #nrestarts 5 , last run nstep 500000
     #1548510021431.2053

     return np.multiply(np.tanh(np.add(np.multiply(x[0], np.log2(np.absolute(np.exp(np.log1p(np.e))))), np.add(x[1], x[2]))), np.multiply(np.tanh(np.maximum(np.maximum(np.sin(np.maximum(np.add(np.log1p(np.maximum(np.sin(np.arctan(np.sin(np.pi))), x[0])), np.arctan(x[0])), x[0])), np.add(np.sin(np.exp(np.log1p(np.exp(np.maximum(np.arctan(np.add(np.add(np.maximum(np.pi, np.arctan(np.arctan(np.sin(np.arctan(np.add(x[2], np.log1p(np.sin(np.sin(np.add(np.multiply(x[0], np.log2(np.absolute(np.exp(np.log1p(np.e))))), np.add(x[1], x[2]))))))))))), x[2]), x[1])), np.sin(x[0])))))), np.log1p(np.tanh(np.maximum(np.tanh(np.log1p(np.tanh(np.maximum(np.maximum(np.tanh(np.add(np.multiply(x[0], np.log2(np.absolute(np.add(np.maximum(np.add(np.e, x[2]), np.arctan(np.add(x[2], x[0]))), x[1])))), np.add(np.exp2(np.sin(np.exp2(np.multiply(np.maximum(np.add(np.add(np.maximum(np.add(np.sin(np.pi), x[2]), np.add(np.sin(np.sin(x[0])), x[1])), np.sin(x[0])), x[1]), np.add(np.log1p(np.e), np.tanh(np.pi))), np.maximum(x[2], np.arctan(np.exp2(np.sin(np.maximum(x[0], np.sin(x[0])))))))))), np.maximum(np.log1p(np.pi), np.sin(np.log1p(np.e)))))), np.add(np.log1p(np.sin(np.arctan(np.sin(np.add(np.sin(np.arctan(np.add(x[2], np.log1p(np.sin(np.log1p(np.tanh(np.sin(np.arctan(np.add(x[1], x[1])))))))))), x[0]))))), np.add(x[1], np.arctan(np.sin(np.exp(np.log1p(np.e))))))), np.add(np.sin(np.add(np.multiply(x[0], np.log2(np.absolute(np.exp(np.log1p(np.e))))), np.add(x[1], x[2]))), np.arctan(np.add(np.add(np.maximum(np.add(np.maximum(np.pi, x[0]), x[2]), np.add(np.tanh(x[2]), np.arctan(np.sin(np.sin(x[0]))))), x[1]), np.sin(np.arctan(np.add(np.multiply(x[0], np.log2(np.absolute(np.exp(np.log1p(np.e))))), np.add(x[1], x[2]))))))))))), np.add(np.sin(x[0]), np.arctan(np.add(np.add(np.maximum(np.add(np.maximum(np.pi, x[0]), x[2]), np.add(np.tanh(np.sin(np.arctan(np.sin(x[0])))), x[2])), x[1]), np.sin(np.exp(np.arctan(np.add(x[2], x[1])))))))))))), np.add(np.log1p(np.tanh(np.log1p(np.tanh(np.tanh(np.log1p(np.tanh(np.add(np.maximum(np.add(np.maximum(np.pi, x[0]), x[2]), np.add(np.sin(x[0]), x[2])), x[1])))))))), np.maximum(np.sinh(np.add(np.tanh(np.add(np.tanh(np.sin(np.exp(np.log1p(np.exp(np.maximum(np.arctan(np.maximum(np.arctan(np.arctan(np.arctan(np.sin(np.exp(np.log1p(np.e)))))), np.sin(np.maximum(np.sin(x[0]), np.maximum(np.add(x[0], np.log2(np.absolute(np.add(np.maximum(np.add(np.maximum(np.pi, np.maximum(np.sin(np.arctan(np.sin(np.add(np.sin(np.arctan(np.add(x[2], np.log1p(np.sin(np.log1p(np.tanh(np.pi))))))), x[0])))), x[0])), x[2]), x[0]), x[1])))), x[0]))))), np.sin(np.maximum(np.arctan(np.log1p(np.tanh(np.sin(np.arctan(np.negative(np.cosh(np.pi))))))), np.sin(np.maximum(np.sin(x[0]), np.maximum(np.add(x[0], np.log2(np.absolute(np.add(np.maximum(np.add(np.maximum(np.arctan(np.maximum(x[2], np.arctan(np.exp2(x[2])))), x[1]), x[2]), np.add(np.sin(np.sin(np.arctan(np.sin(np.arctan(np.add(x[2], np.log1p(np.sin(np.sin(np.exp2(np.arctan(np.add(x[0], x[1])))))))))))), x[0])), x[1])))), x[0]))))))))))), np.sin(np.exp(np.arctan(np.add(x[2], x[1])))))), np.sin(np.maximum(np.add(np.log1p(np.sin(np.sin(np.log1p(np.sin(np.sin(np.arctan(np.sin(np.add(np.sin(np.arctan(np.add(x[2], np.log1p(np.sin(np.log1p(np.tanh(np.sin(np.arctan(np.add(x[1], x[1])))))))))), x[0]))))))))), np.arctan(np.arctan(np.sin(np.arctan(np.add(x[2], x[1])))))), np.sin(np.maximum(np.maximum(np.exp(np.log1p(np.exp(np.maximum(np.sin(np.maximum(np.exp(np.log1p(np.maximum(np.sin(np.arctan(np.add(x[0], np.log2(np.absolute(np.add(np.maximum(np.add(np.maximum(np.pi, x[0]), x[2]), np.add(np.sin(x[0]), x[2])), x[1])))))), x[0]))), np.arctan(x[0]))), np.arctan(np.sin(np.arctan(np.add(x[2], x[1])))))))), np.add(np.exp2(np.sin(x[0])), np.exp2(np.tanh(np.sin(np.exp(np.maximum(np.arctan(np.add(np.add(np.maximum(np.pi, np.exp2(np.arctan(np.add(x[2], x[1])))), x[2]), x[1])), np.sin(np.sin(x[0]))))))))), np.tanh(x[2]))))))), np.sin(np.maximum(np.add(np.log1p(np.maximum(np.sin(np.arctan(np.sin(np.add(np.sin(np.arctan(np.add(x[2], np.log1p(np.sin(np.log1p(np.tanh(np.sin(np.arctan(np.add(x[1], x[1])))))))))), x[0])))), np.multiply(x[0], np.log2(np.absolute(np.exp(np.log1p(np.exp(np.log1p(np.e))))))))), np.arctan(np.add(x[2], x[1]))), x[0])))))), np.sinh(np.hypot(np.negative(np.cosh(np.pi)), np.negative(np.cosh(np.pi))))))

def f3(x: np.ndarray) -> np.ndarray: 
     #mse 0.4122990064737059
     #nrestart nrestarts=6   , no run until plateaux
     #   nstep=50000
     #lastrun nstep =500000


     return np.add(np.add(np.add(np.add(np.add(np.exp2(np.sqrt(np.add(np.negative(np.multiply(np.hypot(x[0], np.sin(np.e)), np.log1p(np.tanh(np.negative(np.pi))))), np.hypot(x[0], np.euler_gamma)))), np.negative(x[2])), np.negative(x[2])), np.negative(x[2])), np.negative(np.cbrt(x[2]))), np.multiply(np.negative(x[1]), np.square(x[1])))
   

def f4(x: np.ndarray) -> np.ndarray: 
     """ min_mse no scaling  using run until plateaux
     0.09592352523610304 """

     return np.multiply(np.sinc(np.multiply(x[1], np.multiply(np.exp(np.negative(np.sin(np.sin(np.add(np.pi, np.euler_gamma))))), np.log10(np.negative(np.sin(np.sin(np.add(np.pi, np.euler_gamma)))))))), np.hypot(np.multiply(x[1], np.pi), np.hypot(np.add(np.add(np.add(np.pi, np.i0(x[1])), np.euler_gamma), np.add(np.pi, np.exp(np.euler_gamma))), np.pi)))

     

def f5(x: np.ndarray) -> np.ndarray: #is zero so it's good

     #using rescaling 15000 steps and run until the plateaux 
     #MSE: 5.512898055429115e-19
     return np.add(np.divide(np.add(np.sin(np.hypot(np.power(np.hypot(np.power(np.hypot(np.power(x[0], np.negative(np.sin(x[0]))), np.euler_gamma), np.negative(np.sin(np.round(x[1])))), np.arctan(x[0])), np.negative(np.sin(x[1]))), np.arctan(np.pi))), np.add(np.tanh(np.i0(np.tanh(np.power(np.exp2(x[1]), np.sin(x[0]))))), np.pi)), 174318497.75735644), -2.8520706810421616e-08)


def f6(x: np.ndarray) -> np.ndarray:  

     # popsize 1500 , using suggested overselections parameters for overpopulation  #and no run_until_plateaux
     #7.145589969138924e-07

     return np.add(np.multiply(np.arctan(np.cbrt(np.euler_gamma)), np.add(x[1], np.negative(x[0]))), x[1])


def f7(x: np.ndarray) -> np.ndarray: 
     #min_mse no scaling 50000 steps , last run 500000 steps
     #mse 54.844555000111505

     return np.hypot(np.multiply(np.add(np.add(np.add(np.hypot(x[0], x[0]), np.hypot(np.multiply(np.remainder(x[1], np.multiply(x[0], np.cosh(np.euler_gamma))), np.e), x[0])), np.cosh(np.remainder(x[1], x[0]))), np.multiply(np.hypot(np.e, np.add(np.multiply(np.square(np.cosh(np.sin(np.sin(np.multiply(x[1], np.cosh(np.cosh(x[1]))))))), np.multiply(x[1], np.remainder(np.cbrt(np.multiply(np.hypot(x[0], np.add(np.hypot(np.multiply(x[1], np.remainder(x[1], np.multiply(x[0], np.cosh(np.sin(np.e))))), x[0]), np.e)), np.multiply(x[1], np.exp2(np.hypot(x[1], np.log10(np.euler_gamma)))))), x[0]))), np.e)), np.multiply(x[1], x[0]))), np.multiply(np.hypot(np.sin(np.multiply(x[1], np.square(np.cosh(np.euler_gamma)))), np.remainder(x[0], np.multiply(x[1], np.cosh(np.euler_gamma)))), np.exp2(np.hypot(np.remainder(x[0], np.multiply(np.remainder(x[1], np.multiply(np.remainder(x[0], np.multiply(np.remainder(x[1], np.multiply(x[0], np.cosh(np.sin(x[1])))), np.cosh(np.euler_gamma))), np.cosh(np.sin(np.sin(np.e))))), np.cosh(np.arctan(np.arctan(np.arctan(np.sin(np.e))))))), np.log10(np.hypot(np.remainder(x[1], x[0]), np.remainder(np.remainder(x[1], np.multiply(x[0], np.cosh(np.sin(x[1])))), x[0]))))))), np.euler_gamma)


def f8(x: np.ndarray) -> np.ndarray: #good
     #min_mse
     #no run until plateaux , nstep 50000, with final refinement
     
     #166579.13617635533
     return np.add(np.multiply(np.add(np.add(x[5], np.round(np.cbrt(np.negative(np.exp2(np.cbrt(np.negative(np.exp2(np.hypot(np.round(np.cbrt(np.negative(np.square(np.arctan(np.add(np.multiply(np.square(np.multiply(x[5], np.add(x[5], x[5]))), np.e), np.cbrt(np.remainder(np.arctan(np.negative(np.hypot(np.euler_gamma, np.square(np.remainder(np.arctan(np.negative(np.exp2(np.hypot(x[4], np.pi)))), np.negative(np.add(x[5], x[5]))))))), np.multiply(np.multiply(x[5], np.add(x[5], x[5])), np.add(x[5], x[5])))))))))), np.pi))))))))), x[5]), np.multiply(np.hypot(np.e, x[5]), np.multiply(x[5], np.add(x[5], x[5])))), np.add(np.multiply(np.square(np.multiply(x[5], np.add(x[5], x[5]))), x[5]), np.negative(np.square(np.hypot(np.exp2(np.hypot(np.hypot(np.euler_gamma, np.pi), x[4])), np.round(x[5]))))))
