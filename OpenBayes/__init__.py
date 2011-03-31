#Python __init__.py file
#setup.py
__all__ = ['bayesnet', 'distributions', 'inference', 'potentials', \
           'table', 'graph', 'OpenBayesXBN', 'BNController']

# this will only import the class names defined in the __all__ parameter of each
# file :
from bayesnet import *
from inference import *
from distributions import *
from potentials import *
from table import *
from graph import *
#from OpenBayesXBN import *
from BNController import *
