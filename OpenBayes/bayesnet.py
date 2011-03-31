"""Bayesian network implementation.  Influenced by Cecil Huang's and
Adnan Darwiche's "Inference in Belief Networks: A Procedural Guide,"
International Journal of Approximate Reasoning, 1994.

Copyright 2005, Kosta Gaitanis (gaitanis@tele.ucl.ac.be).  Please see
the license file for legal information.
"""

__all__ = ['BVertex', 'BNet']
__version__ = '0.1'
__author__ = 'Kosta Gaitanis & Elliot Cohen'
__author_email__ = 'gaitanis@tele.ucl.ac.be; elliot.cohen@gmail.com'
#Python Standard Distribution Packages
import sys
import unittest
import types
import copy
#from timeit import Timer, time
#import profile
import bisect       # for appending elements to a sorted list
import logging

#Major Packages
import numpy

#Library Specific Modules
import graph
import delegate
import distributions
import potentials
import inference

numpy.random.seed()
#logging.basicConfig(level= logging.INFO)

class BVertex(graph.Vertex):
    def __init__(self, name, discrete=True, nvalues=2, observed=True):
        '''
        Name needn't be a string but must be hashable and immutable.
        if discrete = True:
            nvalues = number of possible values for variable contained \
                      in Vertex
        if discrete = False:
            nvalues is not relevant = 0
        observed = True means that this node CAN be observed
        '''
        graph.Vertex.__init__(self, name)
        self.distribution = None
        self.nvalues = int(nvalues)

        self.discrete = discrete
            # a continuous node can be scalar (self.nvalues=1)
            # or vectorial (self.nvalues=n)
            # n=2 is equivalent to 2D gaussian for example

        # True if variable can be observed
        self.observed = observed
        self.family = [self] + list(self.in_v)

    def InitDistribution(self, *args, **kwargs):
        """ Initialise the distribution, all edges must be added"""
        # first decide which type of Distribution
        # if all nodes are discrete, then Multinomial)
        if numpy.alltrue([v.discrete for v in self.family]):
            #print self.name,'Multinomial'
            # FIX: should be able to pass through 'isAdjustable=True' and it work
            self.distribution = distributions.MultinomialDistribution(self, *args, **kwargs)
            return

        # gaussian distribution
        if not self.discrete:
            #print self.name,'Gaussian'
            self.distribution = distributions.Gaussian_Distribution(self, *args, **kwargs)
            return

        # other cases go here

    def setDistributionParameters(self, *args, **kwargs):
        # sets any parameters for the distribution of this node
        self.distribution.setParameters(*args, **kwargs)

    def __str__(self):
        if self.discrete:
            return graph.Vertex.__str__(self) + '    (discrete, %d)' %self.nvalues
        else:
            return graph.Vertex.__str__(self) + '    (continuous)'

    #============================================================
    # This is used for the MCMC engine
    # returns a new distributions of the correct type, containing only
    # the current without its family
    def GetSamplingDistribution(self):
        if self.discrete:
            d = distributions.MultinomialDistribution(self, ignoreFamily = True)
        else:
            d = distributions.Gaussian_Distribution(self, ignoreFamily = True)

        return d

    # This function is necessary for correct Message Pass
    # we fix the order of variables, by using a cmp function
    def __cmp__(a,b):
        ''' sort by name, any other criterion can be used '''
        return cmp(a.name, b.name)


class BNet(graph.Graph):
    log = logging.getLogger('BNet')
    log.setLevel(logging.ERROR)
    def __init__(self, name=None):
        graph.Graph.__init__(self, name)

    def copy(self):
        ''' returns a deep copy of this BNet '''
        G_new = copy.deepcopy(self)
        G_new.InitDistributions()
        for v in self.all_v:
                G_new.v[v.name].distribution.setParameters(v.distribution.Convert_to_CPT())

        return G_new

    def add_e(self, e):
        if e.__class__.__name__ == 'DirEdge':
            graph.Graph.add_e(self, e)
            #e._v[1] = [e._v[1]] + [parent for parent in e._v[1].in_v]
            for v in e._v:
                v.family = [v] + list(v.in_v)
        else:
            raise "All edges should be directed"

    def del_e(self, edge):
        # remove the parent from the child node
        edge._v[1].family.pop(edge._v[1].family.index(edge._v[0]))
        graph.Graph.del_e(self, edge.name)

    def inv_e(self, e):
        self.e[e].invert()
        # change the families of the corresponding nodes
        e._v[0].family.append(e._v[1])
        e._v[1].family.pop(e._v[1].family.index(e._v[0]))

    def Moralize(self):
        logging.info('Moralising Tree')
        G = inference.MoralGraph(name='Moralized '+str(self.name))

        # for each vertice, create a corresponding vertice
        for v in self.v.values():
            G.add_v(BVertex(v.name, v.discrete, v.nvalues))

        # create an UndirEdge for each DirEdge in current graph
        for e in self.e.values():
            # get corresponding vertices in G (and not in self!)
            v1 = G.v[e._v[0].name]
            v2 = G.v[e._v[1].name]
            G.add_e(graph.UndirEdge(len(G.e), v1, v2))

        # add moral edges
        # connect all pairs of parents for each node
        for v in self.v.values():
            # get parents for each vertex
            self.log.debug('Node : ' + str(v))
            parents = [G.v[p.name] for p in list(v.in_v)]
            self.log.debug('parents: ' + str([p.name for p in list(v.in_v)]))

            for p1, i in zip(parents, range(len(parents))):
                for p2 in parents[i+1:]:
                    if not p1.connecting_e(p2):
                        self.log.debug('adding edge '+ str(p1) + ' -- ' + str(p2))
                        G.add_e(graph.UndirEdge(len(G.e), p1, p2))

        return G

    @graph._roprop('List of observed vertices.')
    def observed(self):
        return [v for v in self.v.values() if v.observed]

    def split_into_components(self):
        """ returns a list of BNets with the connected components of this BNet
        """
        components = self.connex_components()

        BNets = []
        i = 0
        # create a BNet for each element in components
        for comp in components:
            new = BNet(self.name + ' (' + str(i+1) + '/' + \
                       str(len(components)) + ')')
            BNets.append(new)

            #add vertices to this new BNet
            for v in comp:
                new.add_v(v)
            #add all edges into this BNet
            for v in comp:
                for e in v.out_e:
                    new.add_e(e)
            i += 1
        return BNets

    def InitDistributions(self):
        """ Finalizes the network, all edges must be added. A
        distribution (unknown) is added to each node of the network
        """
        #---TODO: test if DAG (fdebrouc)
        # this replaces the InitCPTs() function
        for v in self.v.values(): v.InitDistribution()

##    def InitCPTs(self):
##        for v in self.v.values(): v.InitCPT()

    def RandomizeCPTs(self):
        for v in self.v.values():
            v.rand()
            v.makecpt()

    def Sample(self, n=1):
        """ Generate a sample of the network, n is the number of
        samples to generate
        """
        assert(len(self.v) > 0)
        samples = []

        # find a node without parents and start from there.
        # There is always at least one node without parents
        # because a BNet is a Directed Acyclic Graph
        # this is critical in small networks:
        # e.g. A--> B
        #      starting at B will produce an empty output...
        for v in self.v.values():
            if len(v.in_v) == 0:
                start_node = v
                break

        topological = self.topological_sort()

        for i in range(n):
            sample = {}
            for v in topological:
                assert(not v.distribution == None), \
                "vertex's distribution is not initialized"
                sample[v.name] = v.distribution.sample(sample)
            samples.append(sample)

        return samples

    def Dimension(self, node):
        ''' Computes the dimension of node
        = (nbr of state - 1)*nbr of state of the parents
        '''
        q = 1
        for Pa in self.v[node.name].distribution.parents:
            q = q * self.v[Pa.name].nvalues
        dim = (self.v[node.name].nvalues-1) * q
        return dim


class BNetTestCase(unittest.TestCase):
    """ Basic Test Case suite for BNet
    """
    def setUp(self):
        G = BNet('Water Sprinkler Bayesian Network')
        c, s, r, w = [G.add_v(BVertex(name, True, 2)) for name in 'c s r w'.split()]
        for ep in [(c, r), (c, s), (r, w), (s, w)]:
            G.add_e(graph.DirEdge(len(G.e), *ep))
##        G.InitCPTs()
##        c.setCPT([0.5, 0.5])
##        s.setCPT([0.5, 0.9, 0.5, 0.1])
##        r.setCPT([0.8, 0.2, 0.2, 0.8])
##        w.setCPT([1, 0.1, 0.1, 0.01, 0.0, 0.9, 0.9, 0.99])

        G.InitDistributions()

        c.setDistributionParameters([0.5, 0.5])
        s.setDistributionParameters([0.5, 0.9, 0.5, 0.1])
        r.setDistributionParameters([0.8, 0.2, 0.2, 0.8])
        w.setDistributionParameters([1, 0.1, 0.1, 0.01, 0.0, 0.9, 0.9, 0.99])

        self.c = c
        self.s = s
        self.r = r
        self.w = w
        self.BNet = G

    def testTopoSort(self):
        sorted = self.BNet.topological_sort() #xue deleted self
        assert(sorted[0] == self.c and \
               sorted[1] == self.s and \
               sorted[2] == self.r and \
               sorted[3] == self.w), \
               "Sorted was not in proper topological order"

    def testSample(self):
##        cCPT = distributions.MultinomialDistribution(self.c)
##        sCPT = distributions.MultinomialDistribution(self.s)
##        rCPT = distributions.MultinomialDistribution(self.r)
##        wCPT = distributions.MultinomialDistribution(self.w)

        cCPT = self.c.distribution
        sCPT = self.s.distribution
        rCPT = self.r.distribution
        wCPT = self.w.distribution

        cCPT.initializeCounts()
        sCPT.initializeCounts()
        rCPT.initializeCounts()
        wCPT.initializeCounts()

        for i in range(1000):
            sample = self.BNet.Sample()[0]
            # Can use sample in all of these, it will ignore extra variables
            cCPT.incrCounts(sample)
            sCPT.incrCounts(sample)
            rCPT.incrCounts(sample)
            wCPT.incrCounts(sample)
##            cCPT[sample] += 1
##            sCPT[sample] += 1
##            rCPT[sample] += 1
##            wCPT[sample] += 1
        assert(numpy.allclose(cCPT,self.c.distribution.cpt,atol=.1) and \
               numpy.allclose(sCPT,self.s.distribution.cpt,atol=.1) and \
               numpy.allclose(rCPT,self.r.distribution.cpt,atol=.1) and \
               numpy.allclose(wCPT,self.w.distribution.cpt,atol=.1)), \
               "Samples did not generate appropriate CPTs"

    def testFamily(self):
        cFamily = self.BNet.v['c'].family
        sFamily = self.BNet.v['s'].family
        rFamily = self.BNet.v['r'].family
        wFamily = self.BNet.v['w'].family

        assert(set(cFamily) == set([self.c]) and \
               set(sFamily) == set([self.s, self.c]) and \
               set(rFamily) == set([self.r, self.c]) and \
               set(wFamily) == set([self.w, self.r, self.s])), \
              "Families are not set correctly"

if __name__ == '__main__':
    suite = unittest.makeSuite(BNetTestCase, 'test')
    runner = unittest.TextTestRunner()
    runner.run(suite)

    from graph import DirEdge

    # create the network
    G = BNet( 'Water Sprinkler Bayesian Network' )
    c, s, r, w = [G.add_v( BVertex( nm, True, 2 ) ) for nm in 'c s r w'.split()]
    for ep in [( r, w ), ( s, w )]:
        G.add_e( DirEdge( len( G.e ), *ep ) )

    print G


    # finalize the bayesian network once all edges have been added
    G.InitDistributions()

    # c | Pr(c)
    #---+------
    # 0 |  0.5
    # 1 |  0.5
    c.setDistributionParameters([0.5, 0.5])
    r.setDistributionParameters([0.5, 0.5])
    s.setDistributionParameters([0.5, 0.5])

    w.distribution[:,0,0]=[0.99, 0.01]
    w.distribution[:,0,1]=[0.1, 0.9]
    w.distribution[:,1,0]=[0.1, 0.9]
    w.distribution[:,1,1]=[0.0, 1.0]

#    print 'SPLITTING'
#    gg = G.split_into_components()
#    for g in gg:print g

#    for v in G.topological_sort():
#        print v