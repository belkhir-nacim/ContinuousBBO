from benchmark.CEC2005_real import *
import numpy as np
from benchmark.BBOB import Function_BBOB
from benchmark.mpm import MultiplePeaksModel2

import benchmark.BBOB

dim = 50
obj_fun = F24(dim)
x = np.random.uniform(-100,100,dim)

y = obj_fun(x)
print(y)




# obj_fun1 = Function_BBOB(2,1,1)
# y1 = obj_fun1(np.random.uniform(-5,5,dim))
# print(y1)
