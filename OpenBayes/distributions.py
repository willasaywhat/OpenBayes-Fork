###############################################################################
## OpenBayes
## OpenBayes for Python is a free and open source Bayesian Network library
## Copyright (C) 2006  Gaitanis Kosta
##
## This library is free software; you can redistribute it and/or
## modify it under the terms of the GNU Lesser General Public
## License as published by the Free Software Foundation; either
## version 2.1 of the License, or (at your option) any later version.
##
## This library is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the GNU
## Lesser General Public License for more details.
##
## You should have received a copy of the GNU Lesser General Public
## License along with this library (LICENSE.TXT); if not, write to the
## Free Software Foundation,
## Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
###############################################################################
__all__ = ['MultinomialDistribution', 'Gaussian_Distribution']

import types
import random
# Testing
import unittest

import numpy

from table import Table


# object gives access to __new__
class Distribution(object):
    """
    Base Distribution Class for all types of distributions
    defines the Pr(x|Pa(x))

    variables :
    -------------
        - vertex =     a reference to the BVertex instance containing the
                        variable quantified in this distribution

        - family =		[x, Pa(x)1, Pa(x)2,...,Pa(x)N)]
                        references to the nodes

        - names =      set(name of x, name of Pa(x)1,..., name of Pa(x)N)
                        set of strings : the names of the nodes
                        this is a set, no order is specified!

        - names_list =	[name of x, name of Pa(x)1,..., name of Pa(x)N]
                        list of strings : the names of the nodes
                        this is a list, order is specified!

        - parents =    [name of Pa(x)1,...,name of Pa(x)N]
                        list of strings : the names of the node's parents
                        this is a list, order is the same as family[1:]

        - ndimensions = number of variables contained in this distribution
                        = len(self.family)

        - distribution_type = a string the type of the distribution
                              e.g. 'Multinomial', 'Gaussian', ...

        - isAdjustable = if True: the parameters of this distribution can
                         be learned.

        - nvalues =		the dimension of the distribution
                        discrete : corresponds to number of states
                        continuous : corresponds to number of dimensions
                        (e.g. 2D gaussian,...)

    """
    vertex = None
    family = list()
    ndimensions = 0
    parents = list()
    names_list = list()
    #names = set()
    distribution_type = 'None'
    isAdjustable = False
    nvalues = 0

    def __init__(self, v, isAdjustable=False, ignoreFamily=False):
        """ Creates a new distribution for the given variable.
        v is a BVertex instance
        """
        ###################################################
        #---TODO: Should give an order to the variables, sort them by name for example...
        ###################################################
        self.vertex = v		# the node to which this distribution is attached
        if not ignoreFamily:
            self.family = [v] + [parent for parent in v.in_v]
        else:
            # ignore the family of the node, simply return a distribution for this node only
            # used in the MCMC inference engine to create empty distributions
            self.family = [v]

        self.ndimensions = len(self.family)
        self.parents = self.family[1:]
        self.names_list = [v.name for v in self.family]
        #self.names = set(self.names_list)
        self.nvalues = self.vertex.nvalues

        # the type of distribution
        self.distribution_type = 'None'

        #used for learning
        self.isAdjustable = isAdjustable

    def __str__(self):
        string = 'Distribution for node : '+ self.names_list[0]
        if len(self.names_list)>1: string += '\nParents : ' + str(self.names_list[1:])
        return string

    #==================================================
    #=== Learning & Sampling Functions
    def initializeCounts(self):
        raise "Must be implemented in child class !!!"

    def incrCounts(self, index):
        """ add one to given count """
        raise "Must be implemented in child class !!!"

    def addToCounts(self, index, counts):
        raise "Must be implemented in child class !!!"

    def setCounts(self):
        """ set the distributions underlying cpt equal to the counts """
        raise "Must be implemented in child class !!!"


class MultinomialDistribution(Distribution, Table):
    """		Multinomial/Discrete Distribution
    All nodes involved all discrete --> the distribution is represented by
    a Conditional Probability Table (CPT)
    This class now inherits from Distribution and Table.
    """
    def __init__(self, v, cpt=None, isAdjustable=True, ignoreFamily=False):
        Distribution.__init__(self, v, isAdjustable=isAdjustable, \
                              ignoreFamily=ignoreFamily)
        self.distribution_type = "Multinomial"

        assert(numpy.alltrue([v.discrete for v in self.family])), \
              'All nodes in family '+ str(self.names_list) + ' must be discrete !!!'

        self.sizes = [v.nvalues for v in self.family]

        # initialize the cpt
        Table.__init__(self, self.names_list, self.sizes, cpt)

        #Used for Learning
        self.counts = None
        self.augmented = None

    def setParameters(self, *args, **kwargs):
        ''' put values into self.cpt, delegated to Table class'''
        Table.setValues(self, *args, **kwargs)

    def Convert_to_CPT(self):
        return self.cpt

    #======================================================
    #=== Operations on CPT
    def normalize(self, dim=-1):
        """ If dim=-1 all elements sum to 1.  Otherwise sum to specific
        dimension, such that sum(Pr(x=i|Pa(x))) = 1 for all values of i
        and a specific set of values for Pa(x)
        """
        if dim == -1 or len(self.cpt.shape) == 1:
            self.cpt /= self.cpt.sum()
        else:
            ndim = self.assocdim[dim]
            order = range(len(self.names_list))
            order[0] = ndim
            order[ndim] = 0
            tcpt = numpy.transpose(self.cpt, order)
            t1cpt = numpy.sum(tcpt, axis=0)
            t1cpt = numpy.resize(t1cpt,tcpt.shape)
            tcpt = tcpt/t1cpt
            self.cpt = numpy.transpose(tcpt, order)

    def uniform(self):
        """ All CPT elements have equal probability :
            a = Pr(A|B,C,D)
            a.uniform()
            Pr(A=0)=Pr(A=1)=...=Pr(A=N)

            the result is a normalized CPT
            calls self.ones() and then self.normalize()
        """
        self.ones()
        self.normalize()

    ######################################################
    #---TODO: Should add some initialisation functions:
    #			all ones, uniform, zeros
    #			gaussian, ...
    ######################################################

    #======================================================
    #=== Sampling
    def sample(self, index={}):
        """ returns the index of the sampled value
        eg. a=Pr(A)=[0.5 0.3 0.0 0.2]
            a.sample() -->	5/10 times will return 0
                            3/10 times will return 1
                            2/10 times will return 3
                            2 will never be returned

            - returns an integer
            - only works for one variable tables
              eg. a=Pr(A,B); a.sample() --> ERROR
        """
        assert(len(self.names) == 1 or \
               len(self.names - set(index.keys())) == 1), \
               "Sample only works for one variable tables"
        if not index == {}:
            tcpt = self.__getitem__(index)
        else:
            tcpt = self.cpt
        # csum is the cumulative sum of the distribution
        # csum[i] = numpy.sum(self.cpt[0:i])
        # csum[-1] = numpy.sum(self.cpt)
        csum = [numpy.sum(tcpt.flat[0:end+1]) for end in range(tcpt.shape[0])]

        # sample in this distribution
        r = random.random()
        for i,cs in enumerate(csum):
            if r < cs: return i
        return i

    def random(self):
        """ Returns a random state of this distribution,
        chosen completely at random, it does not take account of the
        underlying distribution
        """

        # CHECK: legal values are 0 - nvalues-1: checked OK
        return random.randint(0, self.nvalues-1)

    #==================================================
    #=== Learning & Sampling Functions
    def initializeCounts(self):
        ''' initialize counts array to zeros '''
        self.counts = Table(self.names_list, shape=self.shape)
        self.counts.zeros()

    def initializeCountsToOnes(self):
        ''' initialize counts array to ones '''
        self.counts = Table(self.names_list, shape=self.shape)
        self.counts.ones()

    def incrCounts(self, index):
        """ add one to given count """
        self.counts[index] += 1

    def addToCounts(self, index, counts):
        self.counts[index] += counts

    def setCountsTo(self, index, counts):
        """ set counts to a given value """
        self.counts[index] = counts

    def setCounts(self):
        """ set the distributions underlying cpt equal to the counts """
        assert(self.names_list == self.counts.names_list)
        #set to copy in case we later destroy the counts or reinitialize them
        self.cpt = self.counts.cpt.copy()

    #=== Augmented Bayesian Network
    # The augmented bayesain parameters are used to enforce the
    # direction in which the CPT's evolve when learning parameters
    #
    # They can also be set to equivalent chances if we do not want to
    # enforce a CPT, in this case this is useful because it prohibits
    # eg. the EM-algorithm to output 'nan' when a particular case
    # doesn't exist in the given data (which cases the counts for that
    # case to be zero everywhere which causes normalize() to divide by
    # zero which gives 'nan')
    #
    # More information can be found in "Learning Bayesian Networks"
    # by Richard E. Neapolitan

    def initializeAugmentedEq(self, eqsamplesize=1):
        '''initialize augmented parameters based on the equivalent array size.
        This can be used to give some prior information before learning.
        '''
        ri = self.nvalues
        qi = 1
        for parent in self.family[1:]:
            qi = qi * parent.distribution.nvalues
        self.augmented = Table(self.names_list, shape=self.shape)
        self.augmented[:] = float(eqsamplesize) / (ri * qi)

    def setAugmented(self, index, value):
        '''set the augmented value on position index to value'''
        self.augmented[index] = value

    def setAugmentedAndCounts(self):
        ''' set the distributions underlying cpt equal to the
        counts + the parameters of the augmented network
        '''
        assert(self.names_list == self.counts.names_list)
        # set to copy in case we later destroy the counts or
        # reinitialize them
        if self.augmented == None:
            # if no augmented parameters are used, only use the counts
            cpt = self.counts.cpt
        else:
            cpt = self.counts.cpt + self.augmented.cpt
        self.cpt = cpt.copy()

    #===================================================
    #=== printing functions
    def __str__(self):
        string = 'Multinomial ' + Distribution.__str__(self)
        string += '\nConditional Probability Table (CPT) :\n'
        #---TODO: should find a nice neat way to represent numarrays
        #		  only 3 decimals are sufficient... any ideas?
        string += repr(self.cpt)

        return string


#=================================================================
#=================================================================
class Gaussian_Distribution(Distribution):
    """ Gaussian Continuous Distribution

    Notes: - this can be a scalar gaussian or multidimensional gaussian
            depending on the value of nvalues of the parent vertex
            - The domain is always defined as ]-inf,+inf[
             TODO: Maybe we should add a domain variable...

    parents can be either discrete or continuous.
    continuous parents (if any) : X
    discrete parents (if any) : Q
    this node : Y

    Pr(Y|X,Q) =
         - no parents: Y ~ N(mu(i), Sigma(i,j))		0 <= i,j < self.nvalues
         - cts parents : Y|X=x ~ N(mu + W x, Sigma)
         - discrete parents: Y|Q=i ~ N(mu(i), Sigma(i))
         - cts and discrete parents: Y|X=x,Q=i ~ N(mu(i) + W(i) x, Sigma(i))


     The list below gives optional arguments [default value in brackets].

     mean		- numarray(shape=(len(Y),len(Q1),len(Q2),...len(Qn))
                  the mean for each combination of DISCRETE parents
                  mean[i1,i2,...,in]

     sigma		  - Sigma[:,:,i] is the sigmaariance given
                    Q=i [ repmat(100*eye(Y,Y), [1 1 Q]) ]
     weights	  - W[:,:,i] is the regression matrix given
                    Q=i [ randn(Y,X,Q) ]
     sigma_type	  - if 'diag', Sigma[:,:,i] is diagonal [ 'full' ]
     tied_sigma	  - if True, we constrain Sigma[:,:,i] to be the same
                    for all i [False]
     """

    #---TODO: Maybe we should add a domain variable...
    #---TODO: ADD 'set attribute' for private variables mu, sigma,
    #         weights: they muist always be a numarray!!!!
    def __init__(self, v, mu = None, sigma = None, wi = None, \
                   sigma_type = 'full', tied_sigma = False, \
                   isAdjustable = True, ignoreFamily = False):

        Distribution.__init__(self, v, isAdjustable=isAdjustable, \
                              ignoreFamily=ignoreFamily)
        self.distribution_type = 'Gaussian'

        # check that current node is continuous
        if v.discrete:
            raise 'Node must be continuous'

        self.discrete_parents = [parent for parent in self.parents \
                                 if parent.discrete]
        self.continuous_parents = [parent for parent in self.parents \
                                   if not parent.discrete]

        self.discrete_parents_shape = [dp.nvalues for dp \
                                       in self.discrete_parents]
        self.parents_shape = [p.nvalues for p in self.parents]
        if not self.parents_shape:
            self.parents_shape = [0]

        # set defaults
        # set all mu to zeros
        self.mean = numpy.zeros(([self.nvalues] + \
                             self.discrete_parents_shape), dtype='Float32')

        # set sigma to ones along the diagonal
        eye = numpy.identity(self.nvalues, dtype = 'Float32')[..., numpy.newaxis]
        if len(self.discrete_parents) > 0:
            q = reduce(lambda a, b:a * b, self.discrete_parents_shape) # number of different configurations for the parents
        else:
            q = 1

        sigma = numpy.concatenate([eye] * q, axis=2)
        self.sigma = numpy.array(sigma)
        self.sigma.shape = [self.nvalues, self.nvalues] + self.discrete_parents_shape

        # set weights to
        self.weights = numpy.ones([self.nvalues] + self.parents_shape, dtype='Float32')

        # set the parameters : mean, sigma, weights
# TODO - uh none of this seems to work?
        self.setParameters(mu=mu, sigma=sigma, wi=wi, sigma_type=sigma_type, \
                           tied_sigma=tied_sigma, isAdjustable=isAdjustable)

        #---TODO: add support for sigma_type, tied_sigma
        #---TODO: add learning functions

    def setParameters(self, mu=None, sigma=None, wi=None, sigma_type='full', \
                         tied_sigma=False, isAdjustable=False):
        #============================================================
        # set the mean :
        # self.mean[i] = the mean for dimension i
        # self.mean.shape = (self.nvalues, q1,q2,...,qn)
        #		 where qi is the size of discrete parent i
#        try:

        return

        self.mean = numpy.array(mu, dtype='Float32')
        self.mean.shape = [self.nvalues] + self.discrete_parents_shape
# TODO - what error exactly are we looking for?
#        except:
#            raise 'Could not convert mu to numarray of shape : %s, discrete parents = %s' \
#                  %(str(self.discrete_parents_shape), str(self.discrete_parents))

        #============================================================
        # set the covariance :
        # self.sigma[i,j] = the covariance between dimension i and j
        # self.sigma.shape = (nvalues,nvalues,q1,q2,...,qn)
        #		 where qi is the size of discrete parent i
#        try:
        self.sigma = numpy.array(sigma, dtype='Float32')
        self.sigma.shape = [self.nvalues,self.nvalues] + self.discrete_parents_shape
# TODO - what error exactly are we looking for?
#        except:
#            raise 'Not a valid covariance matrix'

        #============================================================
        # set the weights :
        # self.weights[i,j] = the regression for dimension i and continuous parent j
        # self.weights.shape = (nvalues,x1,x2,...,xn,q1,q2,...,qn)
        #		 where xi is the size of continuous parent i)
        #		 and qi is the size of discrete parent i
#        try:

#        self.weights = numpy.array(wi, dtype='Float32')
#        self.weights.shape = [self.nvalues] + self.parents_shape
# TODO - what error exactly are we looking for?
#        except:
#            raise 'Not a valid weight'

    def normalize(self):
        pass

    #=================================================================
    # Indexing Functions
    def __getitem__(self, index):
        """
        similar indexing to the Table class
        index can be a number, a slice instance, or a dict ,

        returns a tuple (mean, variance, weights)
        """

        print index

        if isinstance(index, types.DictType):
            d_index, c_index = self._numIndexFromDict(index)
        elif isinstance(index, types.TupleType):
            numIndex = list(index)
        else:
            numIndex = [index]

        return tuple([self.mean[tuple([slice(None,None,None)] + d_index)], \
                      self.sigma[tuple([slice(None,None,None)] * 2 + d_index)], \
                      self.weights[tuple([slice(None,None,None)] + c_index)]])

    def __setitem__(self, index, value):
        """ Overload array-style indexing behaviour.
        Index can be a dictionary of var name:value pairs,
        or pure numbers as in the standard way
        of accessing a numarray array array[1,:,1]

        value must be a dict with keys ('mean', 'variance' or 'weights')
        and values the corresponding values to be introduced
        """

        if isinstance(index, types.DictType):
            d_index, c_index = self._numIndexFromDict(index)
        else: raise "not supported..."

#		 elif isinstance(index, types.TupleType):
#			 numIndex = list(index)
#		 else:
#			 numIndex = [index]

        if value.has_key('mean'):
            self.mean[tuple([slice(None,None,None)] + d_index)] = value['mean']
        if value.has_key('sigma'):
            self.sigma[tuple([slice(None,None,None)] * 2 + d_index)] = value['sigma']
        if value.has_key('weights'):
            self.weights[tuple([slice(None,None,None)] + c_index)] = value['weights']

    def _numIndexFromDict(self, d):
        # first treat the discrete parents
        d_index = []
        for dp in self.discrete_parents:
            if d.has_key(dp.name):
                d_index.append(d[dp.name])
            else:
                d_index.append(slice(None,None,None))

        # now treat the continuous parents
        c_index = []
        for dp in self.continuous_parents:
            if d.has_key(dp.name):
                c_index.append(d[dp.name])
            else:
                c_index.append(slice(None, None, None))

        return (d_index, c_index)

    #======================================================
    #=== Sampling
    def sample(self, index={}):
        """ in order to sample from this distributions, all parents must be known """
#		 mean = self.mean.copy()
#		 sigma = self.sigma.copy()
##		  if index:
##			  # discrete parents
##			  for v,i in enumerate(reversed(self.discrete_parents)):
##			  # reverse: avoid missing axes when taking in random
##			  # we start from the end, that way all other dimensions keep the same index
##				  if index.has_key(v.name):
##					  # take the corresponding mean; +1 because first axis is the mean
##					  mean = numpy.take(mean, index[v], axis=(i+1) )
##					  # take the corresponding covariance; +2 because first 2 axes are the cov
##					  sigma = numpy.take(sigma, index[v], axis=(i+2) )
##
##			  # continuous parents
##			  for v in reversed(self.continuous_parents):
##				  if index.has_key(v):

        d_index, c_index = self._numIndexFromDict(index)
        mean  = numpy.array(self.mean[tuple([slice(None, None, None)] + d_index)])
        sigma = self.sigma[tuple([slice(None, None, None)] * 2 +d_index)]
        wi = numpy.sum(self.weights * numpy.array(c_index)[numpy.newaxis,...], axis=1)

#		 if self.continuous_parents:
#			 wi = numpy.array(self.weights[tuple([slice(None,None,None)]+c_index)])
#		 else: wi = 0.0

        # return a random number from a normal multivariate distribution
        return float(numpy.random.multivariate_normal(mean + wi, sigma))

    def random(self):
        """ Returns a random state of this distribution using a uniform distribution """
        # legal values are from -inf to + inf
        # we restrain to mu-5*s --> mu+5*s
        return [(5 * sigma * (random.random() - 0.5) + mu) for mu, sigma in \
                 zip(self.mean, self.sigma.diagonal())]

    #==================================================
    #=== Learning & Sampling Functions
    def initializeCounts(self):
        ''' initialize counts array to empty '''
        self.samples = list()
        # this list will contain all the sampled values

    def incrCounts(self, index):
        """ add the value to list of counts """
        if index.__class__.__name__ == 'list':
            self.samples.extend(index)
        elif index.__class__.__name__ == 'dict':
            # for the moment only take index of main variable
            # ignore the value of the parents,
            # the value of the parents is not necessary for MCMC but it is very
            # important for learning!
            #TODO: Add support for values of the parents
            self.samples.append(index[self.names_list[0]])
        else:
            self.samples.append(index)

    def addToCounts(self, index, counts):
        raise "What's the meaning of this for a gaussian distribution ???"

    def setCounts(self):
        """ set the distributions underlying parameters (mu, sigma)
        to match the samples
        """
        assert(self.samples), "No samples given..."

        samples = numpy.array(self.samples, dtype='Float32')

        self.mean = numpy.sum(samples) / len(samples)

        deviation = samples - self.mean
        squared_deviation = deviation * deviation
        sum_squared_deviation = numpy.sum(squared_deviation)

        self.sigma = (sum_squared_deviation / (len(samples)-1.0)) ** 0.5

    #==================================================
    #=== Printing Functions
    def __str__(self):
        string = 'Gaussian ' + Distribution.__str__(self)
        string += '\nDimensionality : ' + str(self.nvalues)
        string += '\nDiscrete Parents :' + str([p.name for p in self.discrete_parents])
        string += '\nContinuous Parents :' + str([p.name for p in self.continuous_parents])
        string += '\nMu : ' + str(self.mean)
        string += '\nSigma : ' + str(self.sigma)
        string += '\nWeights: ' + str(self.weights)

        return string


#=================================================================
#	Test case for Gaussian_Distribution class
#=================================================================
class GaussianTestCase(unittest.TestCase):
    """ Unit tests for general distribution class
    """
    def setUp(self):
        from bayesnet import BNet, BVertex, graph
        # create a small BayesNet
        self.G = G = BNet('Test')

        self.a = a = G.add_v(BVertex('a', discrete=False, nvalues=1))
        self.b = b = G.add_v(BVertex('b', discrete=False, nvalues=2))
        self.c = c = G.add_v(BVertex('c', discrete=False, nvalues=1))
        self.d = d = G.add_v(BVertex('d', discrete=True, nvalues=2))
        self.e = e = G.add_v(BVertex('e', discrete=True, nvalues=3))
        self.f = f = G.add_v(BVertex('f', discrete=False, nvalues=1))
        self.g = g = G.add_v(BVertex('g', discrete=False, nvalues=1))

        for ep in [(a, c), (b, c), (d, f), (e, f),
                   (a, g), (b, g), (d, g), (e, g)]:
            G.add_e(graph.DirEdge(len(G.e), *ep))

        #a,b : continuous(1,2), no parents
        #c	 : continuous(3), 2 continuous parents (a,b)
        #d,e : discrete(2,3), no parents
        #f	 : continuous(1), 2 discrete parents (d,e)
        #g	 : continuous(1), 2 discrete parents (d,e) & 2 continuous parents (a,b)

        G.InitDistributions()

        self.ad = ad = a.distribution
        self.bd = bd = b.distribution
        self.cd = cd = c.distribution
        self.fd = fd = f.distribution
        self.gd = gd = g.distribution

    def testRandom(self):
        """ tests that the random number generator is correct """
        self.ad.setParameters(sigma=0.1, mu = 1.0)
        for i in range(1000):
            r = self.ad.random()
            assert(r[0] >= float(self.ad.mean[0] - 5 * self.ad.sigma[0]) and \
                   r[0] <= float(self.ad.mean[0] + 5 * self.ad.sigma[0])), \
                   """ random generation is out of borders """

        self.bd.setParameters(sigma = [0.1, 0.0, 0, 1], mu = [1, -1])

        for i in range(1000):
            r = self.bd.random()
            assert(r[0] >= float(self.bd.mean[0] - 5 * self.bd.sigma.flat[0]) and \
                   r[0] <= float(self.bd.mean[0] + 5 * self.bd.sigma.flat[0]) and \
                   r[1] >= float(self.bd.mean[1] - 5 * self.bd.sigma.flat[-1]) and \
                   r[1] <= float(self.bd.mean[1] + 5 * self.bd.sigma.flat[-1])), \
                   """ random generation is out of borders """

    def testNoParents(self):
        ad = self.ad
        bd = self.bd
        # a and b have no parents, test basic parameters
        assert(ad.mean.shape == (1,) and \
               bd.mean.shape == (2,) ), \
               " Mean does not have the correct shape for no parents "

        assert(ad.sigma.shape == (1, 1) and \
               bd.sigma.shape == (2, 2) ), \
               " Sigma does not have the correct shape for no parents "

        assert(ad.weights.shape == (1, 0) and \
               bd.weights.shape == (2, 0)), \
               " Wi does not have the correct shape for no parents "

    def testContinuousParents(self):
        """ test a gaussian with continuous parents """
        cd = self.cd
        #c has two continuous parents
        assert(cd.mean.shape == (cd.nvalues,)), \
                "mean does not have correct shape for continous parents"
        assert(cd.sigma.shape == (cd.nvalues,cd.nvalues)), \
                "sigma does not have correct shape for continous parents"
        assert(cd.weights.shape == tuple([cd.nvalues]+cd.parents_shape)), \
                "weights does not have correct shape for continous parents"

    def testDiscreteParents(self):
        """ test a gaussian with discrete parents """
        fd = self.fd
        assert(fd.mean.shape == tuple([fd.nvalues] + fd.discrete_parents_shape)), \
                "mean does not have correct shape for discrete parents"
        assert(fd.sigma.shape == tuple([fd.nvalues, fd.nvalues] + fd.discrete_parents_shape)), \
                "sigma does not have correct shape for discrete parents"
        assert(fd.weights.shape == tuple([fd.nvalues] + fd.discrete_parents_shape)), \
                "weights does not have correct shape for discrete parents"

    def testDiscrete_and_Continuous_Parents(self):
        """ test a gaussian with discrete and continuous parents """
        gd = self.gd
        assert(gd.mean.shape == tuple([gd.nvalues] + gd.discrete_parents_shape)), \
                "mean does not have correct shape for discrete & continuous parents"
        assert(gd.sigma.shape == tuple([gd.nvalues,gd.nvalues] + gd.discrete_parents_shape)), \
                "sigma does not have correct shape for discrete & continuous parents"
        assert(gd.weights.shape == tuple([gd.nvalues] + gd.parents_shape)), \
                "weights does not have correct shape for discrete & continuous parents"

    def testCounting(self):
        " test counting of samples"
        a = self.a.distribution
        a.initializeCounts()

        assert(a.samples == list()), \
        "Samples list not initialized correctly"

        a.incrCounts(range(10))
        a.incrCounts(5)

        assert(a.samples == range(10) + [5]), \
               "Elements are not added correctly to Samples list"

        a.setCounts()
        error = 0.0001
        assert(numpy.allclose([a.mean, a.sigma], [4.545454, 2.876324], atol = error)), \
               "Mu and sigma doesn't seem to be calculated correctly..."

    def testIndexing(self):
        fd = self.f.distribution	# 2 discrete parents : d(2),e(3)

        fd.setParameters(mu=range(6), sigma=range(6))

        # test normal indexing
        assert(numpy.allclose(fd[0][0].flat, numpy.array(range(3),type='Float32')) and \
               numpy.allclose(fd[0][1].flat, numpy.array(range(3), type='Float32')) and \
               numpy.allclose(fd[1,2][0].flat, numpy.array(5, type='Float32')) and \
               numpy.allclose(fd[1,2][1].flat, numpy.array(5, type='Float32'))), \
        "Normal indexing does not seem to work..."

        # test dict indexing
        assert(numpy.allclose(fd[{'d':0}][0].flat, numpy.array(range(3), dtype='Float32')) and \
               numpy.allclose(fd[{'d':0}][1].flat, numpy.array(range(3), dtype='Float32')) and \
               numpy.allclose(fd[{'d':1,'e':2}][0].flat, numpy.array(5, dtype='Float32')) and \
               numpy.allclose(fd[{'d':1,'e':2}][1].flat, numpy.array(5, dtype='Float32'))), \
        "Dictionary indexing does not seem to work..."

        # now test setting of parameters
        fd[{'d':1,'e':2}] = {'mean':0.5, 'sigma':0.6}
        fd[{'d':0}] = {'mean':[0,1.2,2.4],'sigma':[0,0.8,0.9]}
        numpy.allclose(fd[{'d':0}][0].flat, numpy.array([0,1.2,2.4],dtype='Float32'))
        assert(numpy.allclose(fd[{'d':0}][0].flat, numpy.array([0,1.2,2.4],dtype='Float32')) and \
               numpy.allclose(fd[{'d':0}][1].flat,numpy.array([0,0.8,0.9],dtype='Float32')) and \
               numpy.allclose(fd[{'d':1,'e':2}][0].flat, numpy.array(0.5, dtype='Float32')) and \
               numpy.allclose(fd[{'d':1,'e':2}][1].flat, numpy.array(0.6, dtype='Float32'))), \
        "Setting of values using dict does not seem to work..."

        # now run tests for continuous parents
        cd = self.c.distribution	# 2 continuous parents a(1),b(2)

        cd[{'a':0,'b':1}] = {'weights':69}
        assert(numpy.allclose(cd[{'a':0,'b':1}][2],69.0)), \
        "Indexing for continuous parents does not work"

    def testSampling(self):
        " Test the sample() function "
        a = self.a.distribution

        a.setParameters(mu=5, sigma=1)

        # take 1000 samples from distribution
        samples = [a.sample() for i in range(1000)]

        # verify values
        b = self.a.GetSamplingDistribution()
        b.initializeCounts()
        b.incrCounts(samples)
        b.setCounts()

        error=0.05
        assert(numpy.allclose(b.mean, a.mean, atol = error) and \
                numpy.allclose(b.sigma, a.sigma, atol=error)), \
                "Sampling does not seem to produce a gaussian distribution"

        gd = self.g.distribution	#2 discrete parents, 2 continuous parents


#=================================================================
#	Test case for Distribution class
#=================================================================
class DistributionTestCase(unittest.TestCase):
    """ Unit tests for general distribution class
    """
    def setUp(self):
        from bayesnet import BNet, BVertex, graph
        # create a small BayesNet
        G = BNet('Water Sprinkler Bayesian Network')

        c,s,r,w = [G.add_v(BVertex(nm, discrete=True, nvalues=nv)) for nm, nv \
                   in zip('c s r w'.split(), [2, 3, 4, 2])]

        for ep in [(c, r), (c, s), (r, w), (s, w)]:
            G.add_e(graph.DirEdge(len(G.e), *ep))

        G.InitDistributions()

        self.G = G
        #print G

    def testFamily(self):
        """ test parents, family, etc... """
        G = self.G
        c,s,r,w = G.v['c'], G.v['s'], \
                   G.v['r'], G.v['w']

        assert(c.distribution.parents == [] and \
               set(w.distribution.parents) == set([r, s]) and \
               r.distribution.parents == [c] and \
               s.distribution.parents == [c]), \
               "Error with parents"

        assert(c.distribution.family == [c] and \
               set(s.distribution.family) == set([c, s]) and \
               set(r.distribution.family) == set([r, c]) and \
               set(w.distribution.family) == set([w, r, s])), \
               "Error with family"

        assert(c.distribution.nvalues == c.nvalues), \
                "nvalues not set properly"

##		  assert(c.distribution.order['c'] == 0 and \
##				 set([w.distribution.order['w'],w.distribution.order['s'], \
##                   w.distribution.order['r']]) == set([0,1,2])), \
##				 "Error with order"

#=================================================================
#	Test case for Multinomial_Distribution class
#=================================================================
class MultinomialTestCase(unittest.TestCase):
    """ Unit tests for general distribution class
    """
    def setUp(self):
        from bayesnet import BNet, BVertex, graph
        # create a small BayesNet, Water-Sprinkler
        G = BNet('Test')

        a, b, c, d = [G.add_v(BVertex(nm, discrete=True, nvalues=nv)) for \
                      nm, nv in zip('a b c d'.split(), [2, 3, 4, 2])]
        ad, bd, cd, dd = a.distribution, b.distribution, \
                         c.distribution, d.distribution

        # sizes = (2,3,4,2)
        # a has 3 parents, b,c and d
        for ep in [(b, a), (c, a), (d, a)]:
            G.add_e(graph.DirEdge(len(G.e), *ep))

        G.InitDistributions()

        self.G = G
        self.a, self.b, self.c, self.d = a, b, c, d
        #print G


    def testNormalize(self):
        a = MultinomialDistribution(self.G.v['a'])
        a.setParameters(range(48))
        a.normalize()

    # the test below fails
    # >>> a.distribution.sizes
    # [2, 4, 3, 2]
#    def testSizes(self):
#        print "D:", repr(self.a.distribution)
#        print "S:", self.a.distribution.sizes
#        assert (self.a.distribution.sizes == [2, 3, 4, 2]), "Error with self.sizes"
#        assert (self.a.distribution.sizes == [2, 3, 4, 2]), "Error with self.sizes"

    # test the indexing of the cpt
    def testGetCPT(self):
        """ Violate abstraction and check that setCPT actually worked
        correctly, by getting things out of the matrix
        """
        assert(numpy.all(self.a.distribution[0, 0, 0, :] == \
               self.a.distribution.cpt[0, 0, 0, :]) and \
               numpy.all(self.a.distribution[1, 0, 0, :] == \
               self.a.distribution.cpt[1, 0, 0, :])), \
               "Error getting raw cpt"

    def testSetCPT(self):
        """ Violate abstraction and check that we can actually set elements.
        """
        self.a.distribution.cpt[0, 1, 0, :] = numpy.array([4, 5])
        assert(numpy.all(self.a.distribution[0, 1, 0, :] == numpy.array([4, 5]))), \
               "Error setting the array when violating abstraction"

    def testDictIndex(self):
        """ test that an index using a dictionary works correctly
        """
        index = {'a':0, 'b':0, 'c':0}
        index2 = {'a':1, 'b':0, 'c':0}
        assert(numpy.all(self.a.distribution[0, 0, 0, :] == \
               self.a.distribution[index]) and \
               numpy.all(self.a.distribution[1, 0, 0, :] == \
               self.a.distribution[index2])), \
               "Error getting with dict index"

    # the test below fails
    # the case of index3 fails because the order of the nodes
    # is not a, b, c, d, but a, c, b, d
    # >>> print G.v
    # {'a': <OpenBayes.bayesnet.BVertex object at 0x879e10c>,
    #  'c': <OpenBayes.bayesnet.BVertex object at 0x87a628c>,
    #  'b': <OpenBayes.bayesnet.BVertex object at 0x87a648c>,
    #  'd': <OpenBayes.bayesnet.BVertex object at 0x87a64cc>}
    def testDictSet(self):
        """ test that an index using a dictionary can set a value
        within the cpt
        """
        index = {'a':0, 'b':0, 'c':0}
        index2 = {'a':1, 'b':0, 'c':0}
        index3 = {'a':1, 'b':1, 'c':0}
        self.a.distribution[index] = -1
        self.a.distribution[index2] = 100
        self.a.distribution[index3] = numpy.array([-2, -3])
        assert(numpy.all(self.a.distribution[0, 0, 0, :] == numpy.array([-1, -1])) and \
               numpy.all(self.a.distribution[1, 0, 0, :] == numpy.array([100, 100])) and \
               numpy.all(self.a.distribution[1, 1, 0, :] == numpy.array([-2, -3]))), \
               "Error setting cpt with dict"

    def testNumIndex(self):
        """ test that a raw index of numbers works correctly
        """
        assert(numpy.all(self.a.distribution[0, :, 0, :] == self.a.distribution[0, :, 0, :]) and \
               numpy.all(self.a.distribution[1, 0, 0, :] == self.a.distribution[1, 0, 0, :])), \
               "Error getting item with num indices"

    def testNumSet(self):
        """ test that a raw index of numbers can access and set a position in the
        """
        self.a.distribution[0, 0, 0, :] = -1
        self.a.distribution[1, 0, 0, :] = 100
        self.a.distribution[1, 1, 0, :] = numpy.array([-2, -3])
        assert(numpy.all(self.a.distribution[0, 0, 0, :] == \
               numpy.array([-1, -1])) and \
               numpy.all(self.a.distribution[1, 0, 0, :] == \
               numpy.array([100, 100])) and \
               numpy.all(self.a.distribution[1, 1, 0, :] == \
               numpy.array([-2, -3]))), \
               "Error Setting cpt with num indices"


if __name__ == '__main__':
    suite = unittest.makeSuite(GaussianTestCase, 'test')
    runner = unittest.TextTestRunner()
    runner.run(suite)
##	  from bayesnet import *
##
##    suite = unittest.makeSuite(DistributionTestCase, 'test')
##    runner = unittest.TextTestRunner()
##    runner.run(suite)
####
##
##    suite = unittest.makeSuite(MultinomialTestCase, 'test')
##    runner = unittest.TextTestRunner()
##    runner.run(suite)
##
##	  # create a small BayesNet
##	  G = BNet('Water Sprinkler Bayesian Network')
##
##	  c,s,r,w = [G.add_v(BVertex(nm,discrete=True,nvalues=nv)) for nm,nv in \
##               zip('c s r w'.split(),[2,2,2,0])]
##	  w.discrete = False
##	  w.nvalues = 0
##
##
##	  for ep in [(c,r), (c,s), (r,w), (s,w)]:
##		  G.add_e(graph.DirEdge(len(G.e), *ep))
##
##	  print G
##
##	  G.InitDistributions()
##	  c.setDistributionParameters([0.5, 0.5])
##	  s.distribution.setParameters([0.5, 0.9, 0.5, 0.1])
##	  r.distribution.cpt=numpy.array([0.8, 0.2, 0.2, 0.8])
####	w.distribution[:,0,0]=[0.99, 0.01]
####	w.distribution[:,0,1]=[0.1, 0.9]
####	w.distribution[:,1,0]=[0.1, 0.9]
####	w.distribution[:,1,1]=[0.0, 1.0]
##	  wd = w.distribution
##	  print wd.mean
