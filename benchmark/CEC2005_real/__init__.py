"""
This set of 25 test problems was compiled for the Special Session on
Real-Parameter Optimization at the Congress on Evolutionary Computation (CEC),
Edinburgh, UK, 2-5 Sept. 2005. The mathematical definitions are given in
[Suganthan2005]_. Additionally, implementations in C, Java, and Matlab were
provided for the participants. This code is a reimplementation of the Java
code in pure Python.

The correctness of this implementation was verified with the test data
distributed with the original code. In accordance with the special
session's guidelines, it is guaranteed for this data that
:math:`|y - \\hat{y}|/y \\leq 10^{-12}` for reference objective values
:math:`y` and values :math:`\\hat{y}` calculated here.

Problems F3, F7, F8, F10, F11, F14, and F16-F25 are only defined for 2, 10,
30, and 50 dimensions. The other problems accept an arbitrary number of
variables between 1 and 100 (inclusive).

References
----------
.. [Suganthan2005] P. N. Suganthan, N. Hansen, J. J. Liang, K. Deb,
    Y.-P. Chen, A. Auger, and S. Tiwari. Problem Definitions and Evaluation
    Criteria for the CEC 2005 Special Session on Real-Parameter
    Optimization. Technical Report, Nanyang Technological University,
    Singapore, May 2005 and KanGAL Report #2005005, IIT Kanpur, India.
    http://web.mysites.ntu.edu.sg/epnsugan/PublicSite/Shared%20Documents/CEC2005/Tech-Report-May-30-05.pdf

"""
from benchmark.CEC2005_real.unimodal import F1, F2, F3, F4, F5
from benchmark.CEC2005_real.basic_multimodal import F6, F7, F8, F9, F10, F11, F12
from benchmark.CEC2005_real.expanded_multimodal import F13, F14
from benchmark.CEC2005_real.f15 import F15
from benchmark.CEC2005_real.f16 import F16
from benchmark.CEC2005_real.f17 import F17
from benchmark.CEC2005_real.f18 import F18
from benchmark.CEC2005_real.f19 import F19
from benchmark.CEC2005_real.f20 import F20
from benchmark.CEC2005_real.f21 import F21
from benchmark.CEC2005_real.f22 import F22
from benchmark.CEC2005_real.f23 import F23
from benchmark.CEC2005_real.f24 import F24
from benchmark.CEC2005_real.f25 import F25


from math import pi
from numpy import inf
dim_cec2005_restricted = [2, 10, 30, 50]

def cec_bound(fid):
    if fid in [1, 2, 3, 4, 5,6, 14]:
        return (-100,100)
    if fid == 7:
        return (-inf,inf)
    if fid == 8:
        return (-32,32)
    if fid == 11:
        return (-0.5,0.5)
    if fid == 12:
        return (-pi,pi)
    if fid == 13:
        return (-3,1)
    if fid in [9, 10, 15, 16, 17, 18,19,20,21,22,23,24,25]:
        return (-5,5)

def cec_dim(fid,dim):
    if fid in [3,7,8,10,11,14,16,17,18,19,20,21,22,23,24,25]:
        return [2,10,30,50]
    else:
        return range(2,1000)

def cec_assert_dim(fid,dim):
    if fid in [3,7,8,10,11,14,16,17,18,19,20,21,22,23,24,25]:
        if dim in [2,10,30,50]:
            return True
        else:
            return False
    else:
        if dim>1:
            return True

class CEC2005(list):
    """The CEC 2005 problem collection.

    The collection was defined in [Suganthan2005]_. This class inherits from
    :class:`list` and fills itself with 25 problems.

    """
    def __init__(self, num_variables, **kwargs):
        """Constructor.

        Parameters
        ----------
        num_variables : int
            The number of decision variables for the problems. Only 2, 10,
            30, and 50 are admissible for all problems.
        kwargs
            Arbitrary keyword arguments, passed through to the constructors
            of the single problems.

        """
        problems = [None] * 25
        problems[0] = F1(num_variables, **kwargs)
        problems[1] = F2(num_variables, **kwargs)
        problems[2] = F3(num_variables, **kwargs)
        problems[3] = F4(num_variables, **kwargs)
        problems[4] = F5(num_variables, **kwargs)
        problems[5] = F6(num_variables, **kwargs)
        problems[6] = F7(num_variables, **kwargs)
        problems[7] = F8(num_variables, **kwargs)
        problems[8] = F9(num_variables, **kwargs)
        problems[9] = F10(num_variables, **kwargs)
        problems[10] = F11(num_variables, **kwargs)
        problems[11] = F12(num_variables, **kwargs)
        problems[12] = F13(num_variables, **kwargs)
        problems[13] = F14(num_variables, **kwargs)
        problems[14] = F15(num_variables, **kwargs)
        problems[15] = F16(num_variables, **kwargs)
        problems[16] = F17(num_variables, **kwargs)
        problems[17] = F18(num_variables, **kwargs)
        problems[18] = F19(num_variables, **kwargs)
        problems[19] = F20(num_variables, **kwargs)
        problems[20] = F21(num_variables, **kwargs)
        problems[21] = F22(num_variables, **kwargs)
        problems[22] = F23(num_variables, **kwargs)
        problems[23] = F24(num_variables, **kwargs)
        problems[24] = F25(num_variables, **kwargs)
        list.__init__(self, problems)


def instanciate(fid, num_variables,**kwargs):
    assert (cec_assert_dim(fid,num_variables))
    if fid == 1:
        return F1(num_variables, **kwargs)
    if fid == 2:
        return F2(num_variables, **kwargs)
    if fid == 3:
        return F3(num_variables, **kwargs)
    if fid == 4:
        return F4(num_variables, **kwargs)
    if fid == 5:
        return F5(num_variables, **kwargs)
    if fid == 6:
        return F6(num_variables, **kwargs)
    if fid == 7:
        return F7(num_variables, **kwargs)
    if fid == 8:
        return F8(num_variables, **kwargs)
    if fid == 9:
        return F9(num_variables, **kwargs)
    if fid == 10:
        return F10(num_variables, **kwargs)
    if fid == 11:
        return F11(num_variables, **kwargs)
    if fid == 12:
        return F12(num_variables, **kwargs)
    if fid == 13:
        return F13(num_variables, **kwargs)
    if fid == 14:
        return F14(num_variables, **kwargs)
    if fid == 15:
        return F15(num_variables, **kwargs)
    if fid == 16:
        return F16(num_variables, **kwargs)
    if fid == 17:
        return F17(num_variables, **kwargs)
    if fid == 18:
        return F18(num_variables, **kwargs)
    if fid == 19:
        return F19(num_variables, **kwargs)
    if fid == 20:
        return F20(num_variables, **kwargs)
    if fid == 21:
        return F21(num_variables, **kwargs)
    if fid == 22:
        return F22(num_variables, **kwargs)
    if fid == 23:
        return F23(num_variables, **kwargs)
    if fid == 24:
        return F24(num_variables, **kwargs)
    if fid == 25:
        return F25(num_variables, **kwargs)

