########################################################################
## OpenBayes
## OpenBayes for Python is a free and open source Bayesian Network
## library
## Copyright (C) 2006  Gaitanis Kosta, Francois de Brouchoven
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
########################################################################

import random
import unittest
import copy
import time
import logging
import math

import numpy
#import numarray as na

import graph
import bayesnet
import distributions
import readexcel
from inference import ConnexeInferenceJTree, JoinTree


# show INFO messages
#logging.basicConfig(level= logging.INFO)
#uncomment the following to remove all messages
logging.basicConfig(level = logging.NOTSET)


class LearnEngine:

    BNet = None

    def __init__(self, BNet):
        self.BNet = BNet #.copy()#Xue
        self.engine = ConnexeInferenceJTree(self.BNet)

    def ReadFile(self, file):
        """ file is an excell file. This method reads the file and return
        a list of dictionaries (ex: [{'c':0,'s':1,'r':'?','w':1},{...},...]).
        Each dictionary represents a row of the excell table (= a case for the
        learning methods)
        """
        xl = readexcel.readexcel(file)
        sheetnames = xl.worksheets()
        cases = []
        for sheet in sheetnames:
            for row in xl.getiter(sheet):
                cases.append(row)
        return cases

    def SaveInFile(self, file, G_initial = None, G_learned = None, engine = None):
        return
        f = open(file, 'w+')
        if G_initial:
            f.write('Initial structure:' + '\n' + '\n')
            s = str((G_initial))
            f.write(s + '\n' + '\n')
        if G_learned:
            f.write('Learned structure:' + '\n' + '\n')
            s = str((G_learned))
            f.write(s + '\n' + '\n')
        if engine:
            for node in engine.BNet.all_v:
                pa = []
                for i in node.distribution.parents:
                    pa.append(i.name)
                if len(pa) > 0:
                    combi = self.Combinations(pa)
                    for cas in combi:
                        s = str(('node: ', node.name,', parents: ', cas,', \
                                distribution: ', node.distribution[cas]))
                        f.write(s + '\n')
                else:
                    s = str(('node: ', node.name,', distribution: ', \
                            node.distribution))
                    f.write(s + '\n')
        f.close()

class MLLearningEngine(LearnEngine):
    def __init__(self, BNet):
        LearnEngine.__init__(self, BNet)

##    def Combinations(self, vertices):
##        ''' vertices is a list of BVertex instances
##
##        in the case of the water-sprinkler BN :
##        >>> Combinations(['c','r','w']) ##Xue
##        [{'c':0,'r':0,'w':0},{'c':0,'r':0,'w':1},...,{'c'=1,'r':1,'w':1}]
##        '''
##        if len(vertices) > 1:
##            list_comb = self.Combinations(vertices[:-1])
##            new_list_comb = []
##            for el in list_comb:
##                for i in range(self.BNet.v[vertices[-1]].nvalues):
##                    temp = copy.copy(el)
##                    temp[self.BNet.v[vertices[-1]].name]=i
##                    new_list_comb.append(temp)
##            return new_list_comb
##        else:
##            return [{self.BNet.v[vertices[0]].name:el} for el in range(self.BNet.v[vertices[0]].nvalues)]

    def LearnMLParams(self, cases, augmented=0):
        """ Learn and set the parameters of the network to the ML estimate
        contained in cases.

        Warning: this is destructive, it does not take any prior parameters
                 into account. Assumes that all evidence is specified.
        """
        for v in self.BNet.v.values():
            if v.distribution.isAdjustable:
                if augmented:
                    v.distribution.initializeAugmentedEq() # sets all a_ij and b_ij to equivalent sample size #momo
                v.distribution.initializeCounts()
####                if augmented:
####                    v.distribution.normalize(dim=v.name) # set the initial Pr's to a_ij/(a_ij+b_ij)

        for case in cases:
            assert(set(case.keys()) == set(self.BNet.v.keys())), "Not all values of 'case' are set"
            for v in self.BNet.v.values():
                if v.distribution.isAdjustable:
                    v.distribution.incrCounts(case)
####                    if augmented:
####                        v.distribution.setAugmentedAndCounts() #added

        for v in self.BNet.v.values():
            if v.distribution.isAdjustable:
                if augmented:
                    v.distribution.setAugmentedAndCounts() #added
                else:
                    v.distribution.setCounts()
                v.distribution.normalize(dim=v.name)

class EMLearningEngine(LearnEngine):
    """ EM learning algorithm
    Learns the parameters of a known bayesian structure from incomplete data.
    """

    def __init__(self, BNet):
        LearnEngine.__init__(self, BNet)
        #self.engine = ConnexeInferenceJTree(BNet)

    def EMLearning(self, cases, max_iter, initializeFirst=False):
        """ cases = [{'c':0,'s':1,'r':'?','w':1},{...},...]
        Put '?' when the data is unknown.
        Will estimate  the '?' by inference.
        """
        # if wanted the network is initialized (previously learned CPT's are
        # resetted, the previous learned data is lost)
        if initializeFirst:
            for v in self.BNet.v.values():
                if v.distribution.isAdjustable:
                    v.distribution.initializeAugmentedEq() # sets all a_ij and b_ij to equivalent sample size
                    v.distribution.initializeCounts() # sets all counts to zero
                    v.distribution.setAugmentedCounts()
                    v.distribution.normalize(dim=v.name) # set the initial Pr's to a_ij/(a_ij+b_ij)

        iter = 0
        old = None
        new = self.BNet
        precision = 0.05
        while self.hasntConverged(old, new, precision) and iter < max_iter:
            iter += 1
            print 'EM iteration: ', iter
            old = copy.deepcopy(new)
            self.LearnEMParams(cases) ##Xue
            # reinitialize the JunctionTree to take effect of new
            # parameters learned
            self.engine.Initialization()
            # self.engine.GlobalPropagation()
            new = copy.deepcopy(self.BNet)

    def LearnEMParamsWM(self, cases):
        # Initialise the counts of each vertex
        for v in self.BNet.v.values():
            if v.distribution.isAdjustable:
                v.distribution.initializeCounts() # sets all counts to zero

        # First part of the algorithm : Estimation of the unknown
        # data. (E-part)
        for case in cases:
            # assert(set(case.keys()) == set(self.BNet.v.keys())),
            # "Not all values of 'case' are set"

            known = dict() # will contain al the known data of the case
            for key in case.iterkeys():
                if case[key] != '?':
                    known[key] = case[key]

            for v in self.BNet.v.values():
                if v.distribution.isAdjustable:
                    names = [parent.name for parent in v.family[1:]]
                    nvals = [parent.nvalues for parent in v.family[1:]]
                    names.append(v.name)
                    nvals.append(v.nvalues)
                    self.calcExpectation(v,known,names,nvals)

        # Second part of the algorithm : Estimation of the parameters.
        # (M-part)
        for v in self.BNet.v.values():
            if v.distribution.isAdjustable:
                v.distribution.setAugmentedAndCounts()
                v.distribution.normalize(dim=v.name)

    def calcExpectation(self, v, known, names, nvals, cumPr=1):
        """	calculate and set expectations of node v
            based on chain rule: Pr(names | known) = Pr(names[0]|names[1] ..
            names[end], known).Pr(names[1]|names[2]..names[end], known)...

            in:
            v           : node to calc and set
            known       : the known nodes of the net
            names       : the names of the nodes necessary to calcute P(X_v =
                          val_v, pa_v = val_ij| known, prior_CPT)
            nvals       : the nvalues ...
            cumPr       : the cumulative chance of the chain rule

            out:
            Counts of node v are adjusted based on the known nodes
        """
        if cumPr == 0:
            # chance will be remain zero so no use in calculating any further
            return

        newnames = list(names)
        name = newnames.pop()
        newnvals = list(nvals)
        nval = newnvals.pop()

        if known.has_key(name):
            # Value of P(name=X | pa_v(i+1) ... , known) is 1 because of the known information
            if len(newnames) == 0:
                v.distribution.addToCounts(known,cumPr)
            else:
                self.calcExpectation(v, known, newnames, newnvals, cumPr)
        else:
            self.engine.Initialization() #Only Initializiatino is enough to reset because SetObs calls GlobalPropagation
            self.engine.SetObs(known)
            marg = self.engine.Marginalise(name)

            for value in range(nval):
                newknown = dict(known)
                newknown[name] = value
                thisPr = marg[value]
                if not(str(thisPr) == 'nan' or thisPr == 0):
                    if len(newnames) == 0:
                        v.distribution.addToCounts(newknown, cumPr*thisPr)
                    else:
                        self.calcExpectation(v, newknown, newnames, \
                                             newnvals, cumPr*thisPr)


    def LearnEMParams(self, cases):
        """
        First part of the algorithm : Estimation of the unknown
        data. (E-part)
        """
        # Initialise the counts of each vertex
        for v in self.BNet.v.values():
            v.distribution.initializeCounts()##Xue
        for case in cases:
            known = {} # will contain all the known data of case
            unknown = [] # will contain all the unknown keys of case
            for key in case.iterkeys():
                if case[key] != '?': # It's the only part of code you have to change if you want to have another 'unknown sign' instead of '?'
                    known[key] = case[key]
                else:
                    unknown.append(key)
            if len(case) == len(known): # Then all the data is known -> proceed as LearnMLParams (inference.py)
                for v in self.BNet.v.values():
                    if v.distribution.isAdjustable:
                        v.distribution.incrCounts(case)
            else:
                states_list = self.Combinations(unknown) # Give a dictionary list of all the possible states of the unknown parameters
                likelihood_list = self.DetermineLikelihood(known, states_list) # Give a list with the likelihood to have the states in states_list
                for j, index_unknown in enumerate(states_list):
                    index = copy.copy(known)
                    index.update(index_unknown)
                    for v in self.BNet.v.values():
                        if v.distribution.isAdjustable:
                            v.distribution.addToCounts(index, likelihood_list[j])
        """
        Second part of the algorithm : Estimation of the parameters.
        (M-part)
        """
        for v in self.BNet.v.values():
            if v.distribution.isAdjustable:
                v.distribution.setCounts()
                v.distribution.normalize(dim=v.name)

    def DetermineLikelihood(self, known, states_list):
        """
        Give a list with the likelihood to have the states in states_list
        I think this function could be optimized
        """
        likelihood = []
        for states in states_list:
            # states = {'c':0,'r':1} for example (c and r were unknown in the beginning)
            like = 1
            temp_dic = {}
            copy_states = copy.copy(states)
            for key in states.iterkeys():
                """
                It has to be done for all keys because we have to set every
                observation but key to compute the likelihood to have key in
                his state. The multiplication of all the likelihoods gives the
                likelihood to have states.
                """
                temp_dic[key] = (copy_states[key])
                del copy_states[key]
                if len(copy_states) != 0:
                    copy_states.update(known)
                    self.engine.SetObs(copy_states)
                else:
                    self.engine.SetObs(known) # Has to be done at each iteration because of the self.engine.Initialization() below
                #print 'finished	to try setobs'
                like = self.engine.Marginalise(key)[temp_dic[key]]#like = like*self.BNet.v[key].distribution.Convert_to_CPT()[temp_dic[key]]
                #like = like*self.engine.ExtractCPT(key)[temp_dic[key]]
                if str(like) == 'nan':
                    like = 0

                copy_states.update(temp_dic)
                del temp_dic[key]
                self.engine.Initialization()
            likelihood.append(like)
        return likelihood

    def DetermineLikelihoodApprox(self, known, states_list):
        """
        Give a list with the likelihood to have the states in states_list.
        6 to 10 time faster than the DetermineLikelihood function above, but
        has to be tested more... comme algo loopy belief propagation: pearl
        wikipedia
        """
        likelihood = []
        for states in states_list:
            # states = {'c':0,'r':1} for example (c and r were unknown in the beginning)
            like = 1
            parents_state = copy.copy(known)
            parents_state.update(copy.copy(states))
            for key in states.iterkeys():
                cpt = self.BNet.v[key].distribution[parents_state]
                like = like * cpt
            likelihood.append(like)
        return likelihood

    def Combinations(self, vertices):
        ''' vertices is a list of BVertex instances

        in the case of the water-sprinkler BN :
        >>> Combinations(['c','r','w']) ##Xue
        [{'c':0,'r':0,'w':0},{'c':0,'r':0,'w':1},...,{'c'=1,'r':1,'w':1}]
        '''
        if len(vertices) > 1:
            list_comb = self.Combinations(vertices[:-1])
            new_list_comb = []
            for el in list_comb:
                for i in range(self.BNet.v[vertices[-1]].nvalues):
                    temp = copy.copy(el)
                    temp[self.BNet.v[vertices[-1]].name] = i
                    new_list_comb.append(temp)
            return new_list_comb
        else:
            return [{self.BNet.v[vertices[0]].name:el} for el \
                    in range(self.BNet.v[vertices[0]].nvalues)]

    def hasntConverged(self, old, new, precision):
        '''
        Return true if the difference of distribution of at least one vertex
        of the old and new BNet is bigger than precision
        '''
        if not old :
            return True
        else:
            return not	numpy.alltrue([numpy.allclose(v.distribution, \
                                   new.v[v.name].distribution, \
                                   atol=precision) for v in old.v.values()])



class SEMLearningEngine(LearnEngine, EMLearningEngine):
    """ Greedy Structural learning algorithm
    Learns the structure of a bayesian network from known parameters and an
    initial structure.
    """
##    BNet = None # The underlying bayesian network
##    engine = None

    def __init__(self, BNet):
        LearnEngine.__init__(self, BNet)
        #self.engine = ConnexeInferenceJTree(self.BNet)
        self.converged = False
        self.inverted = []#[{'NCEP':'BMI'}, {'age':'comp'}, {'TABAC':'sexe'}, {'BMI':'HOMA'}, {'comp':'albu'}, {'sexe':'HDL'}, {'NCEP':'HDL'}, {'HbA1c':'NHDL'}]
        self.changed = []

    def SEMLearning(self, cases,alpha=0.5, max_iter = 30):
        """Structural EM for optimal structure and parameters if some of the
        data is unknown (put '?' for unknown data).
        """
        t1 = time.time()
        N = len(cases)
        iter = 0
        self.BNet.InitDistributions()
        self.EMLearning(cases, 15)
        while (not self.converged) and iter < max_iter:
            #First we estimate the distributions of the initial structure
            #self.BNet.InitDistributions()
            #self.EMLearning(cases, 15)
            #Then we find a better structure in the neighborhood of self.BNet
            self.LearnStruct(cases, N, alpha, False)
            iter += 1
            print 'Structure Expectation-Maximisation iteration:', iter
        t2 = time.time()
        t = t2-t1
        print t

    def SEMLearningApprox(self, cases, alpha=0.5, max_iter=30):
        """Structural EM for optimal structure and parameters if some of the
        data is unknown (put '?' for unknown data).
        """
        t1 = time.time()
        N = len(cases)
        iter = 0
        #self.BNet.InitDistributions()
        #self.EMLearning(cases, 15)
        while (not self.converged) and iter < max_iter:
            #First we estimate the distributions of the initial structure
            self.BNet.InitDistributions()
            self.EMLearning(cases, 15)
            #Then we find a better structure in the neighborhood of self.BNet
            self.LearnStruct(cases,N,alpha,True)
            iter += 1
            print 'Structure Expectation-Maximisation iteration:', iter
        self.EMLearning(cases, 15)
        t2 = time.time()
        t = t2 - t1
        print t

    def GlobalBICScore(self, N, cases, alpha = 0.5, approx = 0): #Xue
        '''Computes the BIC score of self.BNet'''
        score = 0
        for v in self.BNet.all_v:
            cpt_matrix = v.distribution.Convert_to_CPT()
            dim = self.BNet.Dimension(v)
            score = score + self.ScoreBIC(N, dim, self.BNet, v, cases, alpha, approx)
        return score

    def LearnStruct(self, cases, N, alpha, approx):
        """Greedy search for optimal structure (all the data in cases are known).
        It will go through every node of the BNet. At each node, it will delete
        every outgoing edge, or add every possible edge, or reverse every
        possible edge. It will compute the BIC score each time and keep the BNet
        with the highest score.
        """
        G_initial = self.BNet.copy()
        engine_init = SEMLearningEngine(G_initial)
        G_best = self.BNet.copy()
        prec_var_score = 0
        invert = {}
        change = {}

        for v in self.BNet.all_v:
            G = copy.deepcopy(engine_init.BNet)
            edges = copy.deepcopy(G.v[v.name].out_e)
            temp = {}
            # delete the outgoing edges
            while edges:
                edge = edges.pop(0)
                node = edge._v[1] #node is the child node, the only node for which the cpt table changes
                dim_init = G_initial.Dimension(node)
                score_init = engine_init.ScoreBIC(N, dim_init, G_initial, \
                             G_initial.v[node.name], cases, alpha, approx)
                self.ChangeStruct('del', edge) #delete the current edge
                self.SetNewDistribution(engine_init.BNet, node, cases, approx)
                dim = self.BNet.Dimension(node)
                score = self.ScoreBIC(N, dim, self.BNet, self.BNet.v[node.name], \
                                      cases, alpha, approx)
                var_score = score - score_init
                if var_score > prec_var_score:
                    change = {}
                    invert = {}
                    change[v.name] = node.name
                    print 'deleted:', v.name, node.name, var_score
                    prec_var_score = var_score
                    G_best = self.BNet.copy()
                self.BNet = G_initial.copy()

            # Add all possible edges
            G = copy.deepcopy(engine_init.BNet)
            nodes = []
            for node in G.all_v:
                if (not (node.name in [vv.name for vv in self.BNet.v[v.name].out_v])) and \
                    (not (node.name == v.name)):
                    nodes.append(node)
            while nodes:
                node = nodes.pop(0)
                if G.e.keys():
                    edge = graph.DirEdge(max(G.e.keys()) + 1, \
                           self.BNet.v[v.name], self.BNet.v[node.name])
                else:
                    edge = graph.DirEdge(0, self.BNet.v[v.name], \
                           self.BNet.v[node.name])
                self.ChangeStruct('add', edge)
                if self.BNet.HasNoCycles(self.BNet.v[node.name]):
                    dim_init = engine_init.BNet.Dimension(node)
                    score_init = engine_init.ScoreBIC(N, dim_init, G_initial, \
                                 G_initial.v[node.name], cases, alpha, approx)
                    self.SetNewDistribution(engine_init.BNet, node, cases, approx)
                    dim = self.BNet.Dimension(node)
                    score = self.ScoreBIC(N, dim, self.BNet, \
                                          self.BNet.v[node.name], cases, \
                                          alpha, approx)
                    var_score = score - score_init
                    if var_score > prec_var_score:
                        change = {}
                        invert = {}
                        change[v.name] = node.name
                        print 'added: ', v.name, node.name, var_score
                        prec_var_score = var_score
                        G_best = self.BNet.copy()
                self.BNet = G_initial.copy()

            # Invert all possible edges
            G = copy.deepcopy(G_initial)
            edges = copy.deepcopy(G.v[v.name].out_e)
            while edges:
                edge = edges.pop(0)
                node = self.BNet.v[edge._v[1].name] #node is the child node
                temp[v.name] = node.name
                if temp not in self.inverted:
                    dim_init1 = G_initial.Dimension(node)
                    score_init1 = engine_init.ScoreBIC(N, dim_init1, G_initial, \
                                  G_initial.v[node.name], cases, alpha, approx)
                    self.ChangeStruct('del', edge)
                    self.SetNewDistribution(engine_init.BNet, node, \
                                            cases, approx)
                    dim1 = self.BNet.Dimension(node)
                    score1 = self.ScoreBIC(N, dim1, self.BNet, \
                             self.BNet.v[node.name], cases, alpha, approx)
                    G_invert = self.BNet.copy()
                    engine_invert = SEMLearningEngine(G_invert)
                    inverted_edge = graph.DirEdge(max(G.e.keys()) + 1, \
                                    self.BNet.v[node.name], self.BNet.v[v.name])
                    self.ChangeStruct('add', inverted_edge)
                    if self.BNet.HasNoCycles(self.BNet.v[node.name]):
                        dim_init = G_initial.Dimension(v)
                        score_init = engine_init.ScoreBIC(N, dim_init, \
                                     G_initial, G_initial.v[v.name], cases, \
                                     alpha, approx)
                        self.SetNewDistribution(engine_invert.BNet, v, \
                                                cases, approx)
                        dim = self.BNet.Dimension(v)
                        score = self.ScoreBIC(N, dim, self.BNet, \
                                self.BNet.v[v.name], cases, alpha, approx)
                        var_score = score1 - score_init1 + score - score_init
                        if var_score > prec_var_score + 5: #+ 5 is to avoid recalculation due to round errors
                            invert = {}
                            change = {}
                            invert[node.name] = v.name
                            print 'inverted:', v.name, node.name, var_score
                            prec_var_score = var_score
                            G_best = self.BNet.copy()
                    self.BNet = G_initial.copy()

        #self.BNet is the optimal graph structure
        if prec_var_score == 0:
            self.converged = True
        self.BNet = G_best.copy()
        self.inverted.append(invert)
        self.changed = []
        self.changed.append(change)
        #self.engine = ConnexeInferenceJTree(self.BNet)

    def SetNewDistribution(self, G_initial, node, cases, approx):
        '''Set the new distribution of the node node. The other distributions
        are the same as G_initial (only node has a new parent, so the other
        distributions don't change). Works also with incomplete data'''
        self.BNet.InitDistributions()
        for v in G_initial.all_v:
            if v.name != node.name:
                cpt = G_initial.v[v.name].distribution.Convert_to_CPT()
                self.BNet.v[v.name].distribution.setParameters(cpt)
            else:
                if self.BNet.v[node.name].distribution.isAdjustable:
                    self.BNet.v[node.name].distribution.initializeCounts()
                    for case in cases :
                        known = {} # will contain all the known data of case
                        unknown = [] # will contain all the unknown keys of case
                        for key in case.iterkeys():
                            if case[key] != '?': # It's the only part of code you have to change if you want to have another 'unknown sign' instead of '?'
                                known[key] = case[key]
                            else:
                                unknown.append(key)
                        if len(case) == len(known): # Then all the data is known -> proceed as LearnMLParams (inference.py)
                            self.BNet.v[node.name].distribution.incrCounts(case)
                        else:
                            states_list = self.Combinations(unknown) # Give a dictionary list of all the possible states of the unknown parameters
                            if approx:
                                likelihood_list = self.DetermineLikelihoodApprox(known, states_list) # Give a list with the likelihood to have the states in states_list
                            else:
                                likelihood_list = self.DetermineLikelihood(known, states_list)
                            for j, index_unknown in enumerate(states_list):
                                index = copy.copy(known)
                                index.update(index_unknown)
                                self.BNet.v[node.name].distribution.addToCounts(index, likelihood_list[j])
                    self.BNet.v[node.name].distribution.setCounts()
                    self.BNet.v[node.name].distribution.normalize(dim=node.name)

    def ScoreBIC (self, N, dim, G, node, data, alpha, approx):
        ''' This function computes the BIC score of one node.
        N is the size of the data from which we learn the structure
        dim is the dimension of the node, = (nbr of state - 1)*nbr of state of the parents
        data is the list of cases
        return the BIC score
        Works also with incomplete data!
        '''
        score = self.ForBIC(G, data, node, approx)
        score = score - alpha * dim * math.log(N)
        return score

    def ForBIC(self, G, cases, node, approx):
        ''' Computes for each case the probability to have node and his parents
        in the case state, take the log of that probability and add them.'''
        score = 0
        for case in cases :
            cpt = 0
            known = {} # will contain all the known data of case
            unknown = [] # will contain all the unknown data of case
            for key in case.iterkeys():
                if case[key] != '?': # It's the only part of code you have to change if you want to have another 'unknown sign' instead of '?'
                    known[key] = case[key]
                else:
                    unknown.append(key)
            if len(case) == len(known): # Then all the data is known
                cpt = node.distribution[case]
            else:
                states_list = self.Combinations(unknown) # Give a dictionary list of all the possible states of the unknown parameters
                if approx:
                    likelihood_list = self.DetermineLikelihoodApprox(known, states_list) # Give a list with the likelihood to have the states in states_list
                else:
                    likelihood_list = self.DetermineLikelihood(known, states_list)
                for j, index_unknown in enumerate(states_list):
                    index = copy.copy(known)
                    index.update(index_unknown)
                    cpt = cpt + likelihood_list[j] * node.distribution[index]
            if cpt == 0: # To avoid log(0)
                cpt = math.exp(-700)
            score = score + math.log(cpt)
        return score

    def ChangeStruct(self, change, edge):
        """Changes the edge (add, remove or reverse)"""
        if change == 'del':
            self.BNet.del_e(edge)
        elif change == 'add':
            self.BNet.add_e(edge)
        elif change == 'inv':
            self.BNet.inv_e(edge)
        else:
            assert(False), "The asked change of structure is not possible. Only 'del' for delete, 'add' for add, and 'inv' for invert"

class MLLearningTestCase(unittest.TestCase):
    '''ML Learning Test Case'''
    def setUp(self):
        # create a discrete network
        G = bayesnet.BNet('Water Sprinkler Bayesian Network')
        c, s, r, w = [G.add_v(bayesnet.BVertex(nm, True, 2)) for nm in 'c s r w'.split()]
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
        a, b = [G2.add_v(bayesnet.BVertex(nm, False, 1)) for nm in 'a b'.split()]
        for ep in [(a,b)]:
            G2.add_e(graph.DirEdge(len(G2.e), *ep))

        G2.InitDistributions()
        a.setDistributionParameters(mu=1.0, sigma=1.0)
        b.setDistributionParameters(mu=1.0, sigma=1.0, wi=2.0)

        self.a = a
        self.b = b
        self.G2 = G2

    def testML(self):
        # sample the network 2000 times
        cases = self.BNet.Sample(2000)

        # create a new BNet with same nodes as self.BNet but all parameters
        # set to 1s
        G = copy.deepcopy(self.BNet)

        G.InitDistributions()

        # create an infeence engine
        engine = MLLearningEngine(G)

        # learn according to the test cases
        engine.LearnMLParams(cases)

        tol = 0.05
        assert(numpy.alltrue([numpy.allclose(v.distribution.cpt, \
               self.BNet.v[v.name].distribution.cpt, atol=tol) \
               for v in G.all_v])), \
                " Learning does not converge to true values "

class SEMLearningTestCase(unittest.TestCase):
    def setUp(self):
        # create a discrete network
        G = bayesnet.BNet('Water Sprinkler Bayesian Network')
        c, s, r, w = [G.add_v(bayesnet.BVertex(nm, True, 2)) for \
                      nm in 'c s r w'.split()]
        for ep in [(c, r), (c, s), (r, w), (s, w)]:
            G.add_e(graph.DirEdge(len(G.e), *ep))
        G.InitDistributions()
        c.setDistributionParameters([0.5, 0.5])
        s.setDistributionParameters([0.5, 0.9, 0.5, 0.1])
        r.setDistributionParameters([0.8, 0.2, 0.2, 0.8])
        w.distribution[:, 0, 0] = [0.99, 0.01]
        w.distribution[:, 0, 1] = [0.1, 0.9]
        w.distribution[:, 1, 0] = [0.1, 0.9]
        w.distribution[:, 1, 1] = [0.0, 1.0]
        self.c = c
        self.s = s
        self.r = r
        self.w = w
        self.BNet = G

    def testSEM(self):
        return 
        N = 700
        # sample the network N times, delete some data
        cases = self.BNet.Sample(N)    # cases = [{'c':0,'s':1,'r':0,'w':1},{...},...]
        for i in range(25):
            case = cases[3 * i]
            rand = random.sample(['c', 's', 'r', 'w'], 1)[0]
            case[rand] = '?'
        for i in range(3):
            case = cases[3 * i]
            rand = random.sample(['c', 's', 'r', 'w'], 1)[0]
            case[rand] = '?'
        G = bayesnet.BNet('Water Sprinkler Bayesian Network2')
        c, s, r, w = [G.add_v(bayesnet.BVertex(nm, True, 2)) for nm \
                      in 'c s r w'.split()]
        G.InitDistributions()

        # Test SEMLearning
        struct_engine = SEMLearningEngine(G)
        struct_engine.SEMLearningApprox(cases)
        struct_engine.SaveInFile('./output/testSEM05.txt', G, \
                                 struct_engine.BNet, struct_engine)
        struct_engine.EMLearning(cases, 10)
        struct_engine.SaveInFile('./output/testSEM205.txt', G, \
                                 struct_engine.BNet, struct_engine)
        print '1 OK!'

        G1 = bayesnet.BNet('Water Sprinkler Bayesian Network2')
        c, s, r, w = [G1.add_v(bayesnet.BVertex(nm, True, 2)) for nm \
                      in 'c s r w'.split()]
        G1.InitDistributions()
        struct_engine1 = SEMLearningEngine(G1)
        struct_engine1.SEMLearning(cases, 0)
        struct_engine1.SaveInFile('./output/testSEM0.txt', G1, \
                                  struct_engine1.BNet, struct_engine1)
        struct_engine1.EMLearning(cases, 10)
        struct_engine1.SaveInFile('./output/testSEM20.txt', G1, \
                                  struct_engine1.BNet, struct_engine1)
        print '2 OK!'

        G2 = bayesnet.BNet('Water Sprinkler Bayesian Network2')
        c, s, r, w = [G2.add_v(bayesnet.BVertex(nm, True, 2)) for nm \
                      in 'c s r w'.split()]
        G2.InitDistributions()
        struct_engine2 = SEMLearningEngine(G2)
        struct_engine2.SEMLearning(cases, 10)
        struct_engine2.SaveInFile('./output/testSEM10.txt', G2, \
                                  struct_engine2.BNet, struct_engine2)
        struct_engine2.EMLearning(cases, 10)
        struct_engine2.SaveInFile('./output/testSEM210.txt', G2, \
                                  struct_engine2.BNet, struct_engine2)
        print '3 OK!'

class GreedyStructLearningTestCase(unittest.TestCase):
    # TEST ASIA
    def setUp(self):
        # create the network
        G = bayesnet.BNet( 'Asia Bayesian Network' )

        visit, smoking, tuberculosis, bronchitis, lung, ou, Xray, dyspnoea = \
        [G.add_v(bayesnet.BVertex( nm, True, 2)) for nm in \
        'visit smoking tuberculosis bronchitis lung ou Xray dyspnoea'.split()]

        for ep in [(visit,tuberculosis), (tuberculosis, ou), (smoking,lung), \
                   (lung, ou), (ou, Xray), (smoking, bronchitis), \
                   (bronchitis, dyspnoea), (ou, dyspnoea)]:
            G.add_e(graph.DirEdge(len(G.e), *ep))
        G.InitDistributions()
        visit.setDistributionParameters([0.99, 0.01])
        tuberculosis.distribution[:, 0] = [0.99, 0.01]
        tuberculosis.distribution[:, 1] = [0.95, 0.05]
        smoking.setDistributionParameters([0.5, 0.5])
        lung.distribution[:, 0] = [0.99, 0.01]
        lung.distribution[:, 1] = [0.9, 0.1]
        ou.distribution[:, 0, 0] = [1, 0]
        ou.distribution[:, 0, 1] = [0, 1]
        ou.distribution[:, 1, 0] = [0, 1]
        ou.distribution[:, 1, 1] = [0, 1]
        Xray.distribution[:, 0] = [0.95, 0.05]
        Xray.distribution[:, 1] = [0.02, 0.98]
        bronchitis.distribution[:, 0] = [0.7, 0.3]
        bronchitis.distribution[:, 1] = [0.4, 0.6]
        dyspnoea.distribution[{'bronchitis':0, 'ou':0}] = [0.9, 0.1]
        dyspnoea.distribution[{'bronchitis':1, 'ou':0}] = [0.2, 0.8]
        dyspnoea.distribution[{'bronchitis':0, 'ou':1}] = [0.3, 0.7]
        dyspnoea.distribution[{'bronchitis':1, 'ou':1}] = [0.1, 0.9]
        self.visit = visit
        self.tuberculosis = tuberculosis
        self.smoking = smoking
        self.lung = lung
        self.ou = ou
        self.Xray = Xray
        self.bronchitis = bronchitis
        self.dyspnoea = dyspnoea
        self.BNet = G

    def testStruct(self):
        N = 5000
        # sample the network N times
        cases = self.BNet.Sample(N)    # cases = [{'c':0,'s':1,'r':0,'w':1},{...},...]
        for i in range(25):
            case = cases[3*i]
            rand = random.sample(['visit', 'smoking', 'tuberculosis', \
                   'bronchitis', 'lung', 'ou', 'Xray', 'dyspnoea'], 1)[0]
            case[rand] = '?'
        for i in range(3):
            case = cases[3 * i]
            rand = random.sample(['visit', 'smoking', 'tuberculosis', \
                   'bronchitis', 'lung', 'ou', 'Xray', 'dyspnoea'], 1)[0]
            case[rand] = '?'
        # create two new bayesian network with the same parameters as self.BNet
        G1 = bayesnet.BNet( 'Asia Bayesian Network2' )

        visit, smoking, tuberculosis, bronchitis, lung, ou, Xray, dyspnoea = \
        [G1.add_v( bayesnet.BVertex( nm, True, 2 ) ) for nm in \
        'visit smoking tuberculosis bronchitis lung ou Xray dyspnoea'.split()]

        for ep in [(visit, tuberculosis), (tuberculosis, ou), (smoking, lung), \
                   (lung, ou), (ou, Xray), (smoking, bronchitis), \
                   (bronchitis, dyspnoea), (ou, dyspnoea)]:
            G1.add_e( graph.DirEdge(len(G1.e), *ep))
        G1.InitDistributions()
##        tuberculosis.distribution[:,0]=[0.99, 0.01]
##        tuberculosis.distribution[:,1]=[0.95, 0.05]
##        smoking.setDistributionParameters([0.5, 0.5])
##        lung.distribution[:,0]=[0.99, 0.01]
##        lung.distribution[:,1]=[0.9, 0.1]
##        ou.distribution[:,0,0]=[1, 0]
##        ou.distribution[:,0,1]=[0, 1]
##        ou.distribution[:,1,0]=[0, 1]
##        ou.distribution[:,1,1]=[0, 1]
##        Xray.distribution[:,0]=[0.946, 0.054]
##        Xray.distribution[:,1]=[0.0235, 0.9765]
##        bronchitis.distribution[:,0]=[0.7, 0.3]
##        bronchitis.distribution[:,1]=[0.4, 0.6]
##        dyspnoea.distribution[{'bronchitis':0,'ou':0}]=[0.907, 0.093]
##        dyspnoea.distribution[{'bronchitis':1,'ou':0}]=[0.201, 0.799]
##        dyspnoea.distribution[{'bronchitis':0,'ou':1}]=[0.322, 0.678]
##        dyspnoea.distribution[{'bronchitis':1,'ou':1}]=[0.132, 0.868]
        # Test StructLearning
        struct_engine = SEMLearningEngine(G1)
##        struct_engine.SEMLearningApprox(cases)
        struct_engine.EMLearning(cases,10)
        struct_engine.SaveInFile('./output/asiaEM.txt', G1, \
                                 struct_engine.BNet, struct_engine)
        casestemp = self.BNet.Sample(1000)
##        for i in range(25):
##            case = casestemp[3*i]
##            rand = random.sample(['visit', 'smoking', 'tuberculosis', 'bronchitis', 'lung', 'ou', 'Xray', 'dyspnoea'],1)[0]
##            case[rand] = '?'
##        for i in range(3):
##            case = casestemp[3*i]
##            rand = random.sample(['visit', 'smoking', 'tuberculosis', 'bronchitis', 'lung', 'ou', 'Xray', 'dyspnoea'],1)[0]
##            case[rand] = '?'
        cases21 = []
        cases20 = []
        i = 1
        j = 1
        for cas in casestemp:
            if cas['ou'] == 1:
                del cas['ou']
                cases21.append(cas)
                i = i + 1
            elif cas ['ou'] == 0:
                del cas['ou']
                cases20.append(cas)
                j = j + 1
        print 'length0: ', len(cases20), i
        print 'length1: ', len(cases21), j
        ie = ConnexeInferenceJTree(struct_engine.BNet)
        #print struct_engine.engine.Marginalise('tuberculosis')
        print ie.Marginalise('tuberculosis')
        print ie.Marginalise('lung')
        print ie.Marginalise('dyspnoea')
#        ie.SetObs(cases2[0])
#        print cases2[0]
#        print ie.Marginalise('tuberculosis')
#        print ie.Marginalise('lung')
#        print ie.Marginalise('dyspnoea')

        Gcopy = G1.copy()
        import sys
        f = sys.stdout
        #f = open('./output/testvalidationasia1.txt', 'w+')
        nbr1 = 0
        print ie.Marginalise('ou')
        for truc in cases21:
            cases3 = {}
            ie.Initialization()
##            for v in ie.BNet.all_v:
##                cpt = Gcopy.v[v.name].distribution.Convert_to_CPT()
##                ie.BNet.v[v.name].distribution.setParameters(cpt)
            for iter in truc:
                if truc[iter] != '?':
                    cases3[iter] = truc[iter]
            ie.SetObs(cases3)
            if ie.Marginalise('ou')[1] < 0.055:
                f.write(str((0, ie.Marginalise('ou')[1])))
            else:
                f.write(str((1, ie.Marginalise('ou')[1])))
                nbr1 = nbr1 + 1
            f.write(str(('nombre de 1: ', nbr1)))
            f.write('\n')
        pourcentage = nbr1 * 100 / len(cases21)
        f.write(str((pourcentage)))
#        f.close()

#        f = open('./output/testvalidationasia0.txt', 'w+')
        nbr0 = 0
        for truc in cases20:
            cases3 = {}
            ie.Initialization()
##            for v in ie.BNet.all_v:
##                cpt = Gcopy.v[v.name].distribution.Convert_to_CPT()
##                ie.BNet.v[v.name].distribution.setParameters(cpt)
            for iter in truc:
                if truc[iter] != '?':
                    cases3[iter] = truc[iter]
            ie.SetObs(cases3)
            if ie.Marginalise('ou')[1] < 0.055:
                f.write(str((0, ie.Marginalise('ou')[1])))
                nbr0 = nbr0 + 1
            else:
                f.write(str((1, ie.Marginalise('ou')[1])))
            f.write(str(('nombre de 0: ', nbr0)))
            f.write('\n')
        pourcentage = nbr0 * 100 / len(cases20)
        f.write(str((pourcentage)))
#        f.close()

class EMLearningTestCase(unittest.TestCase):
    def setUp(self):
        # create a discrete network
        G = bayesnet.BNet('Water Sprinkler Bayesian Network')
        c,s,r,w = [G.add_v(bayesnet.BVertex(nm, True, 2)) for nm \
                   in 'c s r w'.split()]
        for ep in [(c, r), (c, s)]:
            G.add_e(graph.DirEdge(len(G.e), *ep))
        G.InitDistributions()
        c.setDistributionParameters([0.5, 0.5])
        s.setDistributionParameters([0.5, 0.9, 0.5, 0.1])
        r.setDistributionParameters([0.8, 0.2, 0.2, 0.8])
        w.setDistributionParameters([0.5, 0.5])
##        w.distribution[:,0,0]=[0.99, 0.01]
##        w.distribution[:,0,1]=[0.1, 0.9]
##        w.distribution[:,1,0]=[0.1, 0.9]
##        w.distribution[:,1,1]=[0.0, 1.0]

        self.c = c
        self.s = s
        self.r = r
        self.w = w
        self.BNet = G

    def testEM(self):
        # sample the network 2000 times
        cases = self.BNet.Sample(200)
        # delete some observations
        for i in range(50):
            case = cases[3 * i]
            rand = random.sample(['c', 's', 'r', 'w'], 1)[0]
            case[rand] = '?'
        for i in range(5):
            case = cases[3 * i]
            rand = random.sample(['c', 's', 'r', 'w'], 1)[0]
            case[rand] = '?'
        # create a new BNet with same nodes as self.BNet but all parameters
        # set to 1s
        G = copy.deepcopy(self.BNet)

        G.InitDistributions()

        engine = EMLearningEngine(G)
        engine.EMLearning(cases, 10)
#        engine.SaveInFile('./output/testerdddddddd.txt', G,None, engine)
        tol = 0.08
        assert(numpy.alltrue([numpy.allclose(v.distribution.cpt, \
               self.BNet.v[v.name].distribution.cpt, atol=tol) \
               for v in engine.BNet.all_v])), \
                " Learning does not converge to true values "
        print 'ok!!!!!!!!!!!!'

if __name__ == '__main__':
    suite = unittest.makeSuite(MLLearningTestCase, 'test')
    runner = unittest.TextTestRunner()
    runner.run(suite)

##    suite = unittest.makeSuite(SEMLearningTestCase, 'test')
##    runner = unittest.TextTestRunner()
##    runner.run(suite)

##    suite = unittest.makeSuite(GreedyStructLearningTestCase, 'test')
##    runner = unittest.TextTestRunner()
##    runner.run(suite)

##    suite = unittest.makeSuite(EMLearningTestCase, 'test')
##    runner = unittest.TextTestRunner()
##    runner.run(suite)
