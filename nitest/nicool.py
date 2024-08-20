"""
Test of the documentation problems

"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike

class TestClass(object):
    """
        The docstring of the class
    """
    def __init__(self,
                a: int = 0,
                b: ArrayLike = 1) -> None:
        """
            Introducing a test class

        Args:
            a : More info about a. This should display in the doc!
            b : An array

        """
        self.a = a
        self.b = b
    def amethod(self,
                u: int = 0,
                z: ArrayLike = None):
        """
        Args:
            u : Just an excuse to use u
            z : So much info about z

        """
        pass

def testfunction(anobj: TestClass = None):
    """
    Args:
        anobj: using the test class

    """
    pass

from dataclasses import dataclass, field
import numpy.typing
from numpy.typing import ArrayLike, NDArray
from types import ModuleType
import astropy.table
# from astropy.table import Table
Table = astropy.table.Table
# from astropy.table import Table

# ArrayLike = np.typing.ArrayLike

def testfunction2(anobj: TestClass = None,
                testmod: ModuleType = None):
    """
    Args:
        anobj: using the test class
        testmod : Your favorite module

    """
    pass

@dataclass
class mydata(object):
    """
    This is my data class

    Args:
        anin : One integer for you
        barray : This is now an array
        data_table : Your table
    """
    anin: int = 0
    barray: ArrayLike = None
    data_table: Table = None # field(default_factory=Table)
