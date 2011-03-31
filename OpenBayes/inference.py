__all__ = ['JoinTree','MCMCEngine']

import logging
import copy
import unittest
import types
import random

import numpy

#from numarray.ieeespecial import getnan

#Library Specific Modules
import graph
import bayesnet
import distributions
from potentials import DiscretePotential
from table import Table

# show INFO messages
#logging.basicConfig(level= logging.INFO)
#uncomment the following to remove all messages
logging.basicConfig(level = logging.NOTSET)


class InferenceEngine:
    """ General Inference Engine class
    Does not really implement something but creates a standard set of
    attributes that any inference engine should implement
    """
    BNet = None         # The underlying bayesian network
    evidence = dict()   # the evidence for the BNet

    def __init__(self, BNet):
        self.BNet = BNet
        self.evidence = {}

    def SetObs(self, ev = dict()):
        """ Incorporate new evidence """
        logging.info('Incorporating Observations')
        # evidence = {var.name:observed value}
        self.evidence = dict(ev)

    def MarginaliseAll(self):
        assert 0, 'In InferenceEngine, method must not be implemented at \
                   Child level'

    def Marginalise(self, v):
        assert 0, 'In InferenceEngine, method must not be implemented at \
                   Child level'

    def MarinaliseFamily(self, v):
        assert 0, 'In InferenceEngine, method must not be implemented at \
                   appropriate level'

    def LearnMLParams(self, cases):
        """ Learn and set the parameters of the network to the ML estimate
        contained in cases.

        Warning: this is destructive, it does not take any prior parameters
                 into account. Assumes that all evidence is specified.
        """
        for v in self.BNet.v.values():
            if v.distribution.isAdjustable:
                v.distribution.initializeCounts()
        for case in cases:
            assert(set(case.keys()) == set(self.BNet.v.keys())), \
                   "Not all values of 'case' are set"
            for v in self.BNet.v.values():
                if v.distribution.isAdjustable:
                    v.distribution.incrCounts(case)
        for v in self.BNet.v.values():
            if v.distribution.isAdjustable:
                v.distribution.setCounts()
                v.distribution.normalize(dim=v.name)


class Cluster(graph.Vertex):
    """
    A Cluster/Clique node for the Join Tree structure
    """
    def __init__(self, Bvertices):

        self.vertices = [v for v in Bvertices]    # list of vertices contained in this cluster
        #self.vertices.sort()    # sort list, much better for math operations

        name = ''
        for v in self.vertices: name += v.name
        graph.Vertex.__init__(self, name)
        names = [v.name for v in self.vertices]
        shape = [v.nvalues for v in self.vertices]
        #---TODO: Continuous....
        self.potential = DiscretePotential(names, shape)

        # weight
        self.weight = reduce(lambda x, y:x * y, [v.nvalues for v in self.vertices])

    def NotSetSepOf(self, clusters):
        """
        returns True if this cluster is not a sepset of any of the clusters
        """
        for c in clusters:
            count = 0
            for v in self.vertices:
                if v.name in [cv.name for cv in c.vertices]: count += 1
            if count == len(self.vertices): return False

        return True

    def ContainsVar(self, v):
        """
        v = list of variable name
        returns True if cluster contains them all
        """
        success = True
        for vv in v:
            if not vv.name in self.potential.names:
                success = False
                break
        return success

    def not_in_s(self, sepset):
        """ set of variables in cluster but not not in sepset, X\S"""
        return set(self.potential.names) - set(sepset.potential.names)
        #return set(v.name for v in self.vertices) - set(v.name for v in sepset.vertices)

    def other(self, v):
        """ set of all variables contained in cluster except v, only one at a time... """
        allVertices = set(vv.name for vv in self.vertices)
        if isinstance(v, (list, set, tuple)):
            setV = set(v)
        else:
            setV = set((v,))
        return allVertices - setV

    def MessagePass(self, c):
        """ Message pass from self to cluster c """
        ####################################################
        ### This part must be revisioned !!!!!!!!!
        ####################################################
        logging.debug('Message Pass from '+ str(self)+' to '+str(c))
        # c must be connected to self by a sepset
        e = self.connecting_e(c)    # sepset that connects the two clusters
        if not e: raise 'Clusters ' + str(self) + ' and ' + str(c) + ' are not connected'
        e = e[0]    # only one edge should connect 2 clusters

        # Projection
        oldphiR = copy.copy(e.potential)                # oldphiR = phiR
        newphiR = self.potential + e.potential            # phiR = sum(X/R)phiX

        #e.potential = newphiR
        e.potential.Update(newphiR)

        # Absorption
        newphiR /= oldphiR

        #print 'ABSORPTION'
        #print newphiR

        c.potential *= newphiR

    def CollectEvidence(self, X=None):
        """
        Recursive Collect Evidence,
        X is the cluster that invoked CollectEvidence
        """
        self.marked = True
        for v in self.in_v:
            if not v.marked: v.CollectEvidence(self)

        if not X == None: self.MessagePass(X)

    def DistributeEvidence(self):
        """
        Recursive Distribute Evidence,
        """
        self.marked = True
        for v in self.in_v:
            if not v.marked: self.MessagePass(v)

        for v in self.in_v:
            if not v.marked: v.DistributeEvidence()


class SepSet(graph.UndirEdge):
    """
    A Separation Set
    """
    def __init__(self, name, c1, c2):
        """
        SepSet between c1, c2

        c1, c2 are Cluster instances
        """
        # find intersection between c1 and c2
        self.vertices = list(set(c1.vertices) & set(c2.vertices))
        self.vertices.sort()

        self.label = ''
        for v in self.vertices: self.label += v.name
        names = [v.name for v in self.vertices]
        shape = [v.nvalues for v in self.vertices]
        #---TODO: Continuous....
        self.potential = DiscretePotential(names, shape)        # self.psi = ones
        graph.UndirEdge.__init__(self, name, c1, c2)

        # mass and cost
        self.mass = len(self.vertices)
        self.cost = self._v[0].weight + self._v[1].weight


    def __str__(self):
        # this also prints mass and cost
        # return '%s: %s -- %s -- %s, mass: %s, cost: %s' %
        # (str(self.name), str(self._v[0]), str(self.label),
        # str(self._v[1]), str(self.mass), str(self.cost))
        return '%s: %s -- %s -- %s' % (str(self.name), str(self._v[0]),
                                       str(self.label), str(self._v[1]))

    def __cmp__(self, other):
        """ first = sepset with largest mass and smallest cost """
        comp = cmp(other.mass, self.mass ) # largest mass
        if comp == 0:
            return cmp(self.cost, other.cost ) # smallest cost
        else: return comp


#=======================================================================

class MoralGraph(graph.Graph):
    def ChooseVertex(self):
        """
        Chooses a vertex from the list according to criterion :

        Selection Criterion :
        Choose the node that causes the least number of edges to be added in
        step 2b, breaking ties by choosing the nodes that induces the
        cluster with the smallest weight
        Implementation in Graph.ChooseVertex()

        The WEIGHT of a node V is the nmber of values V can take
        (BVertex.nvalues)
        The WEIGHT of a CLUSTER is the product of the weights of its
        constituent nodes

        Only works with graphs composed of BVertex instances
        """
        vertices = self.all_v
        # for each vertex, check how many edges will be added
        edgestoadd = [0 for v in vertices]
        clusterweight = [1 for v in vertices]

        for v,i in zip(vertices, range(len(vertices))):
            cluster = [a.name for a in v.adjacent_v]
            cluster.append(v.name)
            clusterleft = copy.copy(cluster)

            # calculate clusterweight
            for c in cluster:
                clusterweight[i] *= self.v[c].nvalues

            for v1 in cluster:
                clusterleft.pop(0)
                for v2 in clusterleft:
                    if not v1 in [a.name for a in self.v[v2].adjacent_v]:
                        edgestoadd[i] += 1

        # keep only the smallest ones, the index
        minedges = min(edgestoadd)
        mini = [vertices[i] for e, i in zip(edgestoadd, \
                range(len(edgestoadd))) if e == minedges]

        # from this list, pick the one that has the smallest clusterweight = nvalues
        # this only works with BVertex instances
        v = mini[numpy.argmin([clusterweight[vertices.index(v)] for v in mini])]

        return v

    def Triangulate(self):
        """
        Returns a Triangulated graph and its clusters.

        POST :  Graph, list of clusters

        An undirected graph is TRIANGULATED iff every cycle of length
        four or greater contains an edge that connects two
        nonadjacent nodes in the cycle.

        Procedure for triangulating a graph :

        1. Make a copy of G, call it Gt
        2. while there are still nodes left in Gt:
        a) Select a node V from Gt according to the criterion
        described below
        b) The node V and its neighbours in Gt form a cluster.
        Connect of the nodes in the cluster. For each edge added
        to Gt, add the same corresponding edge t G
        c) Remove V from Gt
        3. G, modified by the additional arcs introduces in previous
        steps is now triangulated.

        The WEIGHT of a node V is the nmber of values V can take (BVertex.nvalues)
        The WEIGHT of a CLUSTER is the product of the weights of its
        constituent nodes

        Selection Criterion :
        Choose the node that causes the least number of edges to be added in
        step 2b, breaking ties by choosing the nodes that induces the cluster with
        the smallest weight
        Implementation in Graph.ChooseVertex()
        """
        logging.info('Triangulating Tree and extracting Clusters')
        # don't touch this graph, create a copy of it
        Gt = copy.deepcopy(self)
        Gt.name = 'Triangulised ' + str(Gt.name)

        # make a copy of Gt
        G2 = copy.deepcopy(Gt)
        G2.name = 'Copy of '+ Gt.name

        clusters = []

        while len(G2.v):
            v = G2.ChooseVertex()
            #logging.debug('Triangulating: chosen '+str(v))
            cluster = list(v.adjacent_v)
            cluster.append(v)

            #logging.debug('Cluster: '+str([str(c) for c in cluster]))

            c = Cluster(cluster)
            if c.NotSetSepOf(clusters):
                #logging.debug('Appending cluster')
                clusters.append(c)

            clusterleft = copy.copy(cluster)

            for v1 in cluster:
                clusterleft.pop(0)
            for v2 in clusterleft:
                if not (v1 in v2.adjacent_v):
                    v1g = Gt.v[v1.name]
                    v2g = Gt.v[v2.name]
                    Gt.add_e(graph.UndirEdge(max(Gt.e.keys())+1, v1g, v2g))
                    G2.add_e(graph.UndirEdge(max(G2.e.keys())+1, v1, v2))

            # remove from G2
            del G2.v[v.name]
        return Gt, clusters

#=======================================================================
#=======================================================================
class Likelihood(distributions.MultinomialDistribution):
    """ Discrete Likelihood class """
    def __init__(self, BVertex):
        distributions.MultinomialDistribution.__init__(self, BVertex)
        self.v = BVertex
        self.AllOnes()      # -1 = unobserved

    def AllOnes(self):
        self.val = -1
        self.cpt = numpy.ones(self.cpt.shape, dtype='Float32')

    def SetObs(self, i):
        if i == -1: self.AllOnes()
        else:
            self.cpt = numpy.zeros(self.cpt.shape, dtype='Float32')
            self.cpt[i] = 1
            self.val = i

    def IsRetracted(self, val):
        """
        returns True if likelihood is retracted.

        V=v1 in e1. In e2 V is either unobserved, or V=v2
        """
        return (self.val != -1 and self.val != val)

    def IsUnchanged(self, val):
        return self.val == val

    def IsUpdated(self, val):
        return (self.val == -1 and val != -1)

#========================================================================

class JoinTree(InferenceEngine, graph.Graph):
    """ Join Tree inference engine"""
    def __init__(self, BNet):
        """Creates an 'Optimal' JoinTree from a BNet """
        logging.info('Creating JunctionTree engine for ' + str(BNet.name))
        InferenceEngine.__init__(self, BNet)
        graph.Graph.__init__(self, 'JT: ' + str(BNet.name))

        # key = variable name, value = cluster instance containing variable
        # {var.name:cluster}
        self.clusterdict = dict()

        self.likelihoods = [Likelihood(v) for v in self.BNet.observed]
        # likelihood dictionary, key = var name, value = likelihood instance
        self.likedict = dict((v.name, l) for v, l in zip(self.BNet.observed,
                                                         self.likelihoods))

        logging.info('Constructing Optimal Tree')
        self.ConstructOptimalJTree()

        JoinTree.Initialization(self)

        self.GlobalPropagation()

    def ConstructOptimalJTree(self):
        # Moralize Graph
        Gm = self.BNet.Moralize()

        # triangulate graph and extract clusters
        Gt, clusters = Gm.Triangulate()

        # Create Clusters for this JoinTree
        for c in clusters: self.add_v(c)

        logging.info('Connecting Clusters Optimally')
        # Create candidate SepSets
        # one candidate sepset for each pair of clusters
        candsepsets = []
        clustersleft = copy.copy(clusters)
        for c1 in clusters:
            clustersleft.pop(0)
            for c2 in clustersleft:
                candsepsets.append(SepSet(len(candsepsets), c1, c2))

        # remove all edges added to clusters by creating candidate sepsets
        for c in clusters:  c._e=[]

        # sort sepsets, first = largest mass, smallest cost
        candsepsets = sorted(candsepsets)

        # Create trees
        # initialise = one tree for each cluster
        # key = cluster name, value = tree index
        trees = dict([(c.name, i) for c, i in zip(clusters,
                     range(len(clusters)))])

        # add SepSets according to criterion, iff the two clusters connected
        # are on different trees
        for s in candsepsets:
            # if on different trees
            if trees[s._v[0].name] != trees[s._v[1].name]:
                # add SepSet
                self.add_e(SepSet(len(self.e), s._v[0], s._v[1]))

                # merge trees
                oldtree = trees[s._v[1].name]
                for t in trees.items():
                    if t[1] == oldtree: trees[t[0]] = trees[s._v[0].name]

            del s
            # end if n-1 sepsets have been added
            if len(self.e) == len(clusters) - 1: break

    def Initialization(self):
        logging.info('Initialising Potentials for clusters and SepSets')
        # for each cluster and sepset X, set phiX = 1
        for c in self.v.values():   c.potential.AllOnes()         # PhiX = 1
        for s in self.e.values():   s.potential.AllOnes()

        # assign a cluster to each variable
        # multiply cluster potential by v.cpt,
        for v in self.BNet.all_v:
            for c in self.all_v:
                if c.ContainsVar(v.family):
                    # assign a cluster for each variable
                    self.clusterdict[v.name] = c
                    v.parentcluster = c

                    # in place multiplication!
                    #logging.debug('JT:initialisation '+c.name+' *= '+v.name)
                    c.potential *= v.distribution   # phiX = phiX*Pr(V|Pa(V)) (special in-place op)

                    # stop here for this node otherwise we count it
                    # more than once, bug reported by Michael Munie
                    break

        # set all likelihoods to ones
        for l in self.likelihoods: l.AllOnes()


    def UnmarkAllClusters(self):
        for v in self.v.values(): v.marked = False

    def GlobalPropagation(self, start = None):
        if start == None: start = self.v.values()[0]    # first cluster found

        logging.info('Global Propagation, starting at :'+ str(start))
        logging.info('      Collect Evidence')

        self.UnmarkAllClusters()
        start.CollectEvidence()

        logging.info('      Distribute Evidence')
        self.UnmarkAllClusters()
        start.DistributeEvidence()

    def Marginalise(self, v):
        """ returns Pr(v), v is a variable name"""

        # find a cluster containing v
        # v.parentcluster is a convenient choice, can make better...
        c = self.clusterdict[v]
        res = c.potential.Marginalise(c.other(v))
        res.Normalise()

        vDist = self.BNet.v[v].GetSamplingDistribution()
        vDist.setParameters(res)
        return vDist

    def MarginaliseFamily(self, v):
        """ returns Pr(fam(v)), v is a variable name
        """
        c = self.clusterdict[v]
        res = c.Marginalise(c.other(self.BNet.v[v].family))
        return res.Normalise()

    def SetObs(self, ev = dict()):
        """ Incorporate new evidence """
        InferenceEngine.SetObs(self, ev)


        # add any missing variables, -1 means not observed:
        for vv in self.BNet.v.values():
            if not self.evidence.has_key(vv.name):
                self.evidence[vv.name] = -1

        # evidence contains all variables and their observed value (-1 if unobserved)
        # this is necessary to find out which variables have been retracted,
        # unchanged or updated
        self.PropagateEvidence()

    def PropagateEvidence(self):
        """ propagate the evidence in the bayesian structure """
        # Check for Global Retraction, or Global Update
        ev = self.evidence
        retraction = False
        for vv in self.BNet.all_v:
            # check for retracted variables, was observed and now it's observed
            # value has changed
            if self.likedict[vv.name].IsRetracted(ev[vv.name]):
                retraction = True
            # remove any observed variables that have not changed their observed
            # value since last iteration
            elif self.likedict[vv.name].IsUnchanged(ev[vv.name]):
                del ev[vv.name]
            # remove any unobserved variables
            elif ev[vv.name] == -1:
                del ev[vv.name]

        # propagate evidence
        if retraction: self.GlobalRetraction(ev)
        else: self.GlobalUpdate(ev)

    def SetFinding(self, v):
        ''' v becomes True (v=1), all other observed variables are false '''
        logging.info('Set finding, '+ str(v))
        temp = dict((vi.name,0) for vi in self.BNet.observed)
        if temp.has_key(v): temp[v] = 1
        else: raise str(v) + ''' is not observable or doesn't exist'''


        self.Initialization()
        self.ObservationEntry(temp.keys(),temp.values())
        self.GlobalPropagation()

    def GlobalUpdate(self, evidence):
        """ perform message passing to update netwrok according to evidence """
        # evidence = {var.name:value} ; -1=unobserved
        #print evidence
        logging.info('Global Update')
        self.ObservationEntry(evidence.keys(),evidence.values())

        # check if only one Cluster is updated.
        # If true, only DistributeEvidence
        startcluster = set()
        for v in evidence.keys():
            startcluster.add(self.BNet.v[v].parentcluster)

        if len(startcluster) == 1:
            # all variables that have changed are in the same cluster
            # perform DistributeEvidence only
            logging.info('distribute only')
            self.UnmarkAllClusters()
            startcluster.pop().DistributeEvidence()
        else:
            # perform global propagation
            self.GlobalPropagation()

    def GlobalRetraction(self, evidence ):
        logging.info('Global Retraction')
        self.Initialization()
        self.ObservationEntry(evidence.keys(), evidence.values())
        self.GlobalPropagation()

    def ObservationEntry(self, v, val):
        logging.info('Observation Entry')
        for vv, vval in zip(v, val):
            c = self.clusterdict[vv]     # cluster containing likelihood, same as v
            l = self.likedict[vv]
            l.SetObs(vval)
            c.potential *= l

    def MarginaliseAll(self):
        """ returns a dict with all the marginals """
        res = dict()
        for v in self.BNet.v.values():
            if not v.observed: res[v.name] = self.Marginalise(v.name)
        for v in self.BNet.observed:
            res[v.name] = self.Marginalise(v.name)

        return res

    def LearnMLParams(self, cases):
        InferenceEngine.LearnMLParams(self, cases)

        # reinitialize the JunctionTree to take effect of new parameters learned
        self.Initialization()

        self.GlobalPropagation()

    def Print(self):
        for c in self.v.values():
            print c
            print c.cpt
            print c.cpt.shape
            print numpy.sum(c.cpt.flat)

        for c in self.e.values():
            print c
            print c.cpt
            print c.cpt.shape
            print numpy.sum(c.cpt.flat)

    def ExtractCPT (self, v):
        return self.Marginalise(v).cpt

class ConnexeInferenceJTree(JoinTree):
    """ Accepts a non connexe BNet as entry.
        Creates an JoinTree Inference engine for each component of the BNet
        and acts transparently to the user
    """
    def __init__(self, BNet):
        #JoinTree.__init__(self, BNet)
        self.BNets = BNet.split_into_components()
        self.engines={}
        for G in self.BNets:
            JoinTree.__init__(self, G)
            self.engines[G] = JoinTree(G)

    def Marginalise(self, vname):
        """ trouver dans quel reseau appartient le noeud et faire l'inference
        sur celui-ci"""
        for G in self.BNets:
            for v in G.all_v:
                if v.name == vname:
                    #engine = JoinTree(G)
                    return self.engines[G].Marginalise(vname)

##    def SetObs(self, ev = dict()):
##        """ trouver dans quel reseau appartient le noeud et faire l'inference
##        sur celui-ci"""
##        for vert in ev:
##            for G in self.BNets:
##                for v in G.all_v:
##                    if v.name == vert:
##                        evidence = {vert:ev[vert]}
##                        #engine = JoinTree(G)
##                        self.engines[G].SetObs(evidence)

    def SetObs(self, ev=dict()):
        """ trouver dans quel reseau appartient le noeud et faire l'inference
        sur celui-ci"""
        for G in self.BNets:
            evidence = {}
            for vert in ev:
                for v in G.all_v:
                    if v.name == vert:
                        evidence[vert] = ev[vert]
            self.engines[G].SetObs(evidence)



class MCMCEngine(InferenceEngine):
        """ MCMC in the way described in the presentation by Rina Rechter """
        def __init__(self, BNet, Nsamples = 1000):
            InferenceEngine.__init__(self, BNet)
            self.N = Nsamples

        def MarginaliseAll(self):
            samples = self.BNet.Sample(self.N)
            res = dict()
            for v in self.BNet.all_v:
                res[v.name] = self.Marginalise(v.name, samples = samples)

            return res

        def Marginalise(self, vname, samples = None):
            # 1.Sample the network N times
            if not samples:
                # if no samples are given, get them
                samples = self.BNet.Sample(self.N)

            # 2. Create the distribution that will be returned
            v = self.BNet.v[vname]        # the variable
            vDist = v.GetSamplingDistribution()
            vDist.initializeCounts()                 # set all 0s

            # 3.Count number of occurences of vname = i
            #    for each possible value of i, that respects the evidence
            for s in samples:
                if numpy.alltrue([s[e] == i for e, i in self.evidence.items()]):
                    # this samples respects the evidence
                    # add one to the corresponding instance of the variable
                    vDist.incrCounts(s)

            vDist.setCounts()    #apply the counts as the distribution
            vDist.normalize()    #normalize to obtain a probability

            return vDist

class InferenceEngineTestCase(unittest.TestCase):
    """ An abstract set of inference test cases.  Basically anything
    that is similar between the different inference engines can be
    implemented here and automatically applied to lower engines.
    For example, we can define the learning tests here and they
    shouldn't have to be redefined for different engines.
    """
    def setUp(self):
        # create a discrete network
        G = bayesnet.BNet('Water Sprinkler Bayesian Network')
        c, s, r, w = [G.add_v(bayesnet.BVertex(nm, True, 2)) \
                      for nm in 'c s r w'.split()]
        for ep in [(c, r), (c, s), (r, w), (s, w)]:
            G.add_e(graph.DirEdge(len(G.e), *ep))
        G.InitDistributions()
        c.setDistributionParameters([0.5, 0.5])
        s.setDistributionParameters([0.5, 0.9, 0.5, 0.1])
        r.setDistributionParameters([0.8, 0.2, 0.2, 0.8])
        w.distribution[:,0,0]=[0.99, 0.01]
        w.distribution[:,0,1]=[0.1, 0.9]
        w.distribution[:,1,0]=[0.1, 0.9]
        w.distribution[:,1,1]=[0.0, 1.0]

        self.c = c
        self.s = s
        self.r = r
        self.w = w
        self.BNet = G

        # create a simple continuous network
        G2 = bayesnet.BNet('Gaussian Bayesian Network')
        a, b = [G2.add_v(bayesnet.BVertex(nm, False, 1)) \
                for nm in 'a b'.split()]
        for ep in [(a, b)]:
            G2.add_e(graph.DirEdge(len(G2.e), *ep))

        G2.InitDistributions()
        a.setDistributionParameters(mu = 1.0, sigma = 1.0)
        b.setDistributionParameters(mu = 1.0, sigma = 1.0, wi = 2.0)

        self.a = a
        self.b = b
        self.G2 = G2

####class LearningTestCase(InferenceEngineTestCase):
####    """ Learning Test case """
####    def setUp(self):
####        InferenceEngineTestCase.setUp(self)
####
####    def testML(self):
####        # sample the network 2000 times
####        cases = self.BNet.Sample(2000)
####
####        # create a new BNet with same nodes as self.BNet but all parameters
####        # set to 1s
####        G = copy.deepcopy(self.BNet)
####
####        G.InitDistributions()
####
####        # create an infeence engine
####        engine = JoinTree(G)
####
####        # learn according to the test cases
####        engine.LearnMLParams(cases)
####
####        tol = 0.05
####        assert(numpy.alltrue([numpy.allclose(v.distribution.cpt, self.BNet.v[v.name].distribution.cpt, atol=tol) \
####               for v in G.all_v])), \
####                " Learning does not converge to true values "

class MCMCTestCase(InferenceEngineTestCase):
    """ MCMC unit tests.
    """
    def setUp(self):
        InferenceEngineTestCase.setUp(self)
        self.engine = MCMCEngine(self.BNet, 2000)
        self.engine2 = MCMCEngine(self.G2, 5000)

    def testUnobservedDiscrete(self):
        """ DISCRETE: Compute and check the probability of
        water-sprinkler given no evidence
        """
        res = self.engine.MarginaliseAll()

        cprob, sprob, rprob, wprob = res['c'], res['s'], res['r'], res['w']

        error = 0.05
        #print cprob[True] <= (0.5 + error)and cprob[True] >= (0.5-error)
        #print wprob[True] <= (0.65090001 + 2*error) and wprob[True] >= (0.65090001 - 2*error)
        #print sprob[True] <= (0.3 + error) and sprob[True] >= (0.3 - error)

        assert(numpy.allclose(cprob[True], 0.5, atol = error) and \
               numpy.allclose(sprob[True], 0.3, atol = error) and \
               numpy.allclose(rprob[True], 0.5, atol = error) and \
               numpy.allclose(wprob[True], 0.6509, atol = error)), \
        "Incorrect probability with unobserved water-sprinkler network"

    def testUnobservedGaussian(self):
        """ GAUSSIAN: Compute and check the marginals of a simple
        gaussian network
        """
        G = self.G2
        a, b = self.a, self.b
        engine = self.engine2

        res = engine.MarginaliseAll()

        #---TODO: find the true results and compare them...

    def testObservedDiscrete(self):
        """ DISCRETE: Compute and check the probability of
        water-sprinkler given some evidence
        """
        self.engine.SetObs({'c':1,'s':0})
        res = self.engine.MarginaliseAll()

        cprob, sprob, rprob, wprob = res['c'], res['s'], res['r'], res['w']

        error = 0.05
        assert(numpy.allclose(cprob.cpt, [0.0,1.0], atol=error) and \
               numpy.allclose(rprob.cpt, [0.2,0.8], atol=error) and \
               numpy.allclose(sprob.cpt, [1.0,0.0], atol=error) and \
               numpy.allclose(wprob.cpt, [ 0.278, 0.722], atol=error) ), \
               " Somethings wrong with MCMC inference with evidence "



class JTreeTestCase(InferenceEngineTestCase):
    def setUp(self):
        InferenceEngineTestCase.setUp(self)
        self.engine = JoinTree(self.BNet)

    def testGeneral(self):
        """ Check that the overall algorithm works """
        c = self.engine.Marginalise('c')
        r = self.engine.Marginalise('r')
        s = self.engine.Marginalise('s')
        w = self.engine.Marginalise('w')

        assert(numpy.allclose(c.cpt, [0.5, 0.5]) and \
                numpy.allclose(r.cpt, [0.5, 0.5]) and \
                numpy.allclose(s.cpt, [0.7, 0.3]) and \
                numpy.allclose(w.cpt, [0.34909999, 0.65090001])), \
                " Somethings wrong with JoinTree inference engine"

    def testEvidence(self):
        """ check that evidence works """
        print 'evidence c=1,s=0'
        self.engine.SetObs({'c':1,'s':0})

        c = self.engine.Marginalise('c')
        r = self.engine.Marginalise('r')
        s = self.engine.Marginalise('s')
        w = self.engine.Marginalise('w')

        assert(numpy.allclose(c.cpt,[0.0,1.0]) and \
                numpy.allclose(r.cpt,[0.2,0.8]) and \
                numpy.allclose(s.cpt,[1.0,0.0]) and \
                numpy.allclose(w.cpt,[ 0.278, 0.722]) ), \
                " Somethings wrong with JoinTree evidence"

    def testMarginaliseAll(self):
        res = self.engine.MarginaliseAll()

        assert(res.__class__.__name__ == 'dict' and \
               set(res.keys()) == set(self.BNet.v)), \
               " MarginaliseAll is not a correct dictionary "


    ###########################################################
    ### SHOULD ADD A MORE GENERAL TEST:
    ###     - not only binary nodes
    ###     - more complex structure
    ###     - check message passing
    ###########################################################


if __name__=='__main__':
##    suite = unittest.makeSuite(MCMCTestCase, 'test')
##    runner = unittest.TextTestRunner()
##    runner.run(suite)

    suite = unittest.makeSuite(JTreeTestCase, 'test')
    runner = unittest.TextTestRunner()
    runner.run(suite)

####    suite = unittest.makeSuite(LearningTestCase, 'test')
####    runner = unittest.TextTestRunner()
####    runner.run(suite)
