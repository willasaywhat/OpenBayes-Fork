########################################################################
## OpenBayes
## OpenBayes for Python is a free and open source Bayesian Network library
## Copyright (C) 2006  Gaitanis Kosta, Sebastien Arnaud
##
## This library is free software; you can redistribute it and/or
## modify it under the terms of the GNU Lesser General Public
## License as published by the Free Software Foundation; either
## version 2.1 of the License, or (at your option) any later version.
##
## This library is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
## Lesser General Public License for more details.
##
## You should have received a copy of the GNU Lesser General Public
## License along with this library (LICENSE.TXT); if not, write to the
## Free Software Foundation,
## Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
########################################################################
#####!/usr/bin/env python2.4
##### encoding: utf-8

"""
bayesiannetworks.py

Created by Sebastien Arnaud on 2006-11-20.
Copyright (c) 2006 All rights reserved.
"""
__all__ = ['BNController']

import cPickle, gzip
import types

from OpenBayes import BNet, BVertex, DirEdge, graph
from OpenBayes import learning, MCMCEngine  #, LoadXBN, SaveXBN


class BNController(object):
    _BN = None
    _BNDef = None

    def __init__(self, BNName=None, BNnodesdef=None):
        if BNnodesdef != None:
            self._init_BN(BNName, BNnodesdef)

    def _init_BN(self, BNName, BNnodesdef):
        BNnodes = {}
        BNconnections = []
        BN = BNet(BNName)

        # Save the nodesdef for future use (saving)
        self._BNDef = BNnodesdef

        for node in BNnodesdef:
            (nodename, isdiscrete, numstates, leafnode) = node
            BNnodes[nodename] = BN.add_v(BVertex(nodename, isdiscrete, numstates))

        # TODO: Find a way to improve this and avoid to have to loop a
        # second time
        # reason is because BNnodes[leafnode] is not sure to be there when
        # going through the first loop

        for node in BNnodesdef:
            (nodename, isdiscrete, numstates, leafnode) = node
            if type(leafnode) == type([]):
                for r in leafnode:
                    if r != None:
                        BNconnections.append((BNnodes[nodename], BNnodes[r]))
            elif leafnode != None:
                BNconnections.append((BNnodes[nodename], BNnodes[leafnode]))
            else:
                #do nothing
                pass

        for ep in BNconnections:
            BN.add_e(DirEdge(len(BN.e), *ep))

        # Ok our Bnet has been created, let's save it in the controller
        self._BN = BN

        # Let's not forget to initialize the distribution
        self._BN.InitDistributions()

    def show_graph(self):
        print self._BN

    def show_distribution(self):
        for v in self._BN.all_v:
            print v.distribution, '\n'

    def load(self, filename):
        xbnparser = LoadXBN(filename)
        self._BN = xbnparser.Load()

    def save(self, filename):
        SaveXBN(filename, self._BN)

    def train(self, cases):
        engine = learning.MLLearningEngine(self._BN)
        engine.LearnMLParams(cases)

    def eval(self, evidences, node):
        #TODO: node could be a list if more than 1 node needs to be guessed
        ie = MCMCEngine(self._BN)

        for e in evidences:
            ie.SetObs(e)
        return ie.Marginalise(node).Convert_to_CPT()


if __name__ == "__main__":
    """
    Simple example on how to use the class above
    """
    output_trace = False

    # =================================
    # PHASE 1: BN Definition & Training
    # =================================
    if output_trace: print "Phase 1"

    # 1] Create a test BN
    gname = 'a simple test'
    gdef = [('a', True, 3, ['b']), ('b', True, 2, None)]
    myBN = BNController(gname, gdef)
    if output_trace: myBN.show_graph()

    # 2] Train the test BN with data
    training_data = [{'a':0, 'b':0}, {'a':1, 'b':1}, {'a':2, 'b':1}]
    myBN.train(training_data)
    if output_trace: myBN.show_distribution()

    # 3] Save & Destory the test BN to conclude
    myBN.save('./output/my_test_bn.xbn')
    myBN = None

    # =====================================
    # PHASE 2: BN Evaluation with test data
    # =====================================
    if output_trace: print "Phase 2"

    # 1] Recreate quickly the test BN & CPTs learned during training
    myBN = BNController()
    myBN.load('./output/my_test_bn.xbn')
    if output_trace: myBN.show_distribution()

    # 2] Present evidences and see what the BN returns
    test_data = [{'a': 2}]
    result_cpt = myBN.eval(test_data, 'b')

    # Voila ;)
    print "Presented the following test_data: %s \nBN returned the following:\n\
P(b=0)= %.2f \nP(b=1)= %.2f" % (test_data, result_cpt[0], result_cpt[1])