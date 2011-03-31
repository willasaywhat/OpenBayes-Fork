__all__ = ['Table']

#!/usr/bin/env python
""" This is a set of code for subclassing numarray.
It creates a new table class which is similar to numarray's basic array
except that each dimension of the array is associated with a name.
This allows indexing via a dictionary and transposing dimensions
according to an ordered list of dimension names.

Copyright 2005 Elliot Cohen and Kosta Gaitanis, please see the license
file for further legal information.
"""

__version__ = '0.1'
__author__ = 'Kosta Gaitanis & Elliot Cohen'
__author_email__ = 'gaitanis@tele.ucl.ac.be; elliot.cohen@gmail.com'
#import random
import unittest
import types
from copy import copy

import numpy

# avoid divide by zero warnings...
# TODO - this really sucks.
numpy.seterr(invalid='ignore', divide='ignore')

class Table:
    def __init__(self, names, shape=None, elements=None, type='Float32'):
      ''' names = ['a','b',...]
          shape = (2, 3, ...) (default: binary)
          elements = [0, 1, 2,....] (a list or a numarray, default: all ones)
          type = 'Float32' or 'Float64' or 'UInt8', etc... (default: Float32)
      '''
      # set default parameters
      if shape == None:
          shape = [2] * len(names)
      if elements == None:
          elements = numpy.ones(shape=shape)

      self.cpt = numpy.array(elements, dtype=type).reshape(shape)

      self.names = set(names)
      self.names_list = list(names) # just to keep the order in an easy to use way

      # dict of name:dim number pairs
      self.assocdim = dict(zip(self.names_list, range(len(self.names_list))))

      # dict of dim:name pairs
      self.assocname = dict(enumerate(self.names_list))

#===============================================================================
#    def normalize(self, dim=-1):
#        """ If dim=-1 all elements sum to 1.  Otherwise sum to specific dimension, such that
#        sum(Pr(x=i|Pa(x))) = 1 for all values of i and a specific set of values for Pa(x)
#        """
#        if dim == -1 or len(self.cpt.shape) == 1:
#            self.cpt /= self.cpt.sum()
#        else:
#            ndim = self.assocdim[dim]
#            order = range(len(self.names_list))
#            order[0] = ndim
#            order[ndim] = 0
#            tcpt = numpy.transpose(self.cpt, order)
#            t1cpt = numpy.sum(tcpt, axis=0)
#            t1cpt = numpy.resize(t1cpt,tcpt.shape)
#            tcpt = tcpt/t1cpt
#            self.cpt = numpy.transpose(tcpt, order)
#    #======================================================
#    #=== Sampling
#    def sample(self, index={}):
#        """ returns the index of the sampled value
#        eg. a=Pr(A)=[0.5 0.3 0.0 0.2]
#            a.sample() -->  5/10 times will return 0
#                            3/10 times will return 1
#                            2/10 times will return 3
#                            2 will never be returned
#
#            - returns an integer
#            - only works for one variable tables
#              eg. a=Pr(A,B); a.sample() --> ERROR
#        """
#        assert(len(self.names) == 1 or len(self.names - set(index.keys())) == 1),\
#              "Sample only works for one variable tables"
#        if not index == {}:
#            tcpt = self.__getitem__(index)
#        else:
#            tcpt = self.cpt
#        # csum is the cumulative sum of the distribution
#        # csum[i] = numpy.sum(self.cpt[0:i])
#        # csum[-1] = numpy.sum(self.cpt)
#        csum = [numpy.sum(tcpt.flat[0:end+1]) for end in range(tcpt.shape[0])]
#
#        # sample in this distribution
#        r = random.random()
#        for i,cs in enumerate(csum):
#            if r < cs: return i
#        return i
#===============================================================================
    #==================================
    #Administration stuff
    def __getattr__(self, name):
        """ delegate to self.cpt """
        return getattr(self.cpt, name)

    def __coerce__(self, other):
        assert(isinstance(other, Table))
        return (self, other)

    def __copy__(self):
        """ copy method """
        return Table(self.names_list, self.shape, self.cpt, self.cpt.dtype)

    def Update(self, other):
        """ updates this Table with the values contained in the other"""
        # check that all variables in self are contained in other
        if self.names != other.names:
            return "error in update, all variables in other should be contained in self"

        # find the correspondance vector
        correspond = []
        for vara in self.names_list:
            correspond.append(other.assocdim[vara])

        self.cpt = copy(numpy.transpose(other.cpt, axes=correspond))
    #===================================
    # Put values into the cpt
    def rand(self):
        ''' put random values to self.cpt '''
        self.cpt = numpy.mlab.rand(*self.shape)

    def AllOnes(self):
        self.cpt = numpy.ones(self.shape, dtype='Float32')

    def setValues(self, values):###X ???self.sizes is not a atribute, change to self.shape
        self.cpt = numpy.array(values, dtype='Float32').reshape(self.sizes)
    #==================================
    # Indexing
    def __getitem__(self, index):
      """ Overload array-style indexing behaviour.
      Index can be a dictionary of var name:value pairs,
      or pure numbers as in the standard way
      of accessing a numarray array array[1,:,1]

      returns the indexed cpt
      """
      if isinstance(index, types.DictType):
         numIndex = self._numIndexFromDict(index)
      else:
         numIndex = index
      return self.cpt[numIndex]

    def __setitem__(self, index, value):
      """ Overload array-style indexing behaviour.
      Index can be a dictionary of var name:value pairs,
      or pure numbers as in the standard way
      of accessing a numarray array array[1,:,1]
      """
      if isinstance(index, types.DictType):
         numIndex = self._numIndexFromDict(index)
      else:
         numIndex = index
      self.cpt[numIndex] = value

    def _numIndexFromDict(self, d):
      index = []
      for dim in range(len(self.shape)):
         if d.has_key(self.assocname[dim]):###X might be bug
            index.append(d[self.assocname[dim]])
         else:
            index.append(slice(None, None, None))
      return tuple(index) # must convert to tuple in order to work, bug fix

    #=====================================
    # Printing
    def __repr__(self):
      " Return printable representation of instance."
      className = self.__class__.__name__
      className = className.zfill(5).replace('0', ' ')
      rep= className + repr(self.cpt)[5:]
      rep += '\nVariables :' + str(self.names_list)
      return rep

    #=====================================
    # Operations
    def addDim(self, newDimName):###X bug??? e.g. abc->abcd the no of elements of self.cpt still 8
        """adds a new unary dimension to the table """
        # add a new dimension to the cpt
        self.cpt = self.cpt[..., numpy.newaxis]

        self.names.add(newDimName)
        self.names_list.append(newDimName) # just to keep the order in an easy to use way

        # dict of name:dim number pairs
        self.assocdim[newDimName] = len(self.names) - 1
        # dict of dim:name pairs
        self.assocname[len(self.names) - 1] = newDimName

    def __eq__(a, b):
        """ True if a and b have same elements, size and names """
        if b.__class__ == numpy.ndarray:
        # in case b is a just a numarray and not a Table instance
        # in this case, variable should absoltely be at the same order
        # otherwise the Table and numArray are considered as different
            return (numpy.alltrue(a.cpt.flat == b.flat) \
                    and a.shape == b.shape)

        elif b == None:
        # in case b is None type
            return False

        elif isinstance(b, (int, float, long)):
        # b is just a number, int, float, long
            return a.cpt == b

        else:
        # the b class should better be a Table or something like that
        # order of variables is not important
            # put the variables in the same order
            # first must get the correspondance vector :
            bcpt = a.prepareOther(b)
            return (a.names == b.names and \
                    bcpt.shape == a.shape and \
                    numpy.allclose(bcpt, a.cpt))

## This code checks that order is the same
##            return (a.shape == b.shape \
##                    and a.names_list == b.names_list \
##                    and numpy.alltrue(a.cpt.flat == b.cpt.flat)  \
##                    )

    def __imul__(a, b):
        """
        in place multiplication
        PRE:
            - B must be a subset of A!!!
            eg.
                a=Pr(A); A = {'a','b','c'}
                b=Pr(B); B = {'c','a'}

        usage:
        a*=b

        POST:
            a=Pr(A)*Pr(B) = Pr(a,b,c)


        Notes :
        -   a keeps the order of its existing variables
        -   b is not touched during this operation
        -   operation is done in-place for a, a is not the same after the operation
        """
        # prepare dimensions in b for multiplication
        cptb = a.prepareOther(b)

        # multiply in place, a's values are changed
        a.cpt *= cptb  # this does not work correctly for some reason...
        #numpy.multiply(a.cpt,cptb,a.cpt) # does not work either
        #a.cpt = a.cpt * cptb    #this one works fine
                                #is this a numarray BUG????

        return a

    def __idiv__(a,b):
        """
        in place division
        PRE:
            - B must be a subset of A!!!
            eg.
                a=Pr(A); A = {'a','b','c'}
                b=Pr(B); B = {'c','a'}

        usage:
        a/=b

        POST:
            a=Pr(A)/Pr(B) = Pr(a,b,c)


        Notes :
        -   a keeps the order of its existing variables
        -   b is not touched during this operation
        -   operation is done in-place for a, a is not the same after the operation
        """
        # prepare dimensions in b for multiplication
        cptb = a.prepareOther(b)

        # multiply in place, a's values are changed
        #a.cpt /= cptb  # this does not work correctly for some reason...
        #numpy.divide(a.cpt,cptb,a.cpt) # does not work either
        a.cpt = a.cpt / cptb    #this one works fine
                                #is this a numarray BUG????

        ## WARNING, division by zero, avoided using numpy.Error.setMode(invalid='ignore')
        # replace INFs by 0s
        a.cpt[numpy.isnan(a.cpt)] = 0
        #---TODO: replace this very SLOW function with a ufunc

        return a

    def __mul__(a, b):
        """
        multiplication
        PRE:
            a=Pr(A); A = {'a','b','c'}
            b=Pr(B); B = {'c','a','d','e'}

        usage:
        c = a*b
        c is a NEW Table instance

        POST:
            c=Pr(A U B) = Pr(a,b,c,d,e)

        Notes :
        -   c keeps the order of the variables in a
        -   any new variables in b (d and e) are added at the end of c in the
            order they appear in b
        -   a and b are not touched during this operation
        -   return a NEW Table instance
        """
        # prepare dimensions in a and b for multiplication
        new, cptb = a.union(b)

        # multiply
        #new.cpt *= cptb  # this does not work correctly for some reason...
        #numpy.multiply(new.cpt,cptb,new.cpt) # does not work either
        new.cpt = new.cpt * cptb    #this one works fine
                                #is this a numarray BUG????

        return new

    def __div__(a, b):
        """
        multiplication
        PRE:
            a=Pr(A); A = {'a','b','c'}
            b=Pr(B); B = {'c','a','d','e'}

        usage:
        c = a/b
        c is a NEW Table instance

        POST:
            c=Pr(A U B) = Pr(a,b,c,d,e)

        Notes :
        -   c keeps the order of the variables in a
        -   any new variables in b (d and e) are added at the end of c in the
            order they appear in b
        -   a and b are not touched during this operation
        -   return a NEW Table instance
        """
        #########################################
        #---TODO: add division with a number
        #########################################

        # prepare dimensions in a and b for multiplication
        new, cptb = a.union(b)

        # multiply
        #new.cpt /= cptb  # this does not work correctly for some reason...
        #numpy.divide(new.cpt,cptb,new.cpt) # does not work either
        new.cpt = new.cpt / cptb    #this one works fine
                                #is this a numarray BUG????

        ## WARNING, division by zero, avoided using numpy.Error.setMode(invalid='ignore')
        # replace INFs by 0s
        new.cpt[numpy.isnan(new.cpt)] = 0
        #---TODO: replace this very SLOW function with a ufunc

        return new

    def prepareOther(self, other):
        """
        Prepares other for inplace multiplication/division with self. Returns
        a *view* of other.cpt ready for an operation. other must contain a
        subset of the variables of self. NON-DESTRUCTIVE!

        eg. a= Pr(A,B,C,D)
            b= Pr(D,B)
            a.prepareOther(b) --> returns a numarray Pr(1,B,1,D)

            a= Pr(A,B,C,D)
            b= Pr(C,B,E)
            a.prepareOther(b) --> ERROR (E not in {A,B,C,D})

        Notes:
        -   a and b are not altered in any way. NON-DESTRUCTIVE
        -   b must contain a subset of a's variables
            a=Pr(X),b=Pr(Y); Y entirely included in X
        """
        #self contains all variables found in other
        if len(other.names - self.names) > 0:
            raise "ERROR :" + str((other.names-self.names)) + "not in" + str(self.names)

        # add new dimensions to b
        bcpt = other.cpt.view()
        b_assocdim = copy(other.assocdim)
        for var in (self.names - other.names):
            #for all variables found in self and not in other
            #add a new dimension to other
            bcpt = bcpt[..., numpy.newaxis]
            b_assocdim[var] = bcpt.ndim - 1

        #create the transposition vector
        trans = list()
        for var in self.names_list:
                trans.append(b_assocdim[var])

        bcpt_trans = numpy.transpose(bcpt, axes=trans)

        # transpose and return bcpt
        return bcpt_trans

    def union(a,b):
        """ Returns a new instance of same class as a that contains all
        data contained in a but also has any new variables found in b with unary
        dimensions. Also returns a view of b.cpt ready for an operation with
        the returned instance.

        eg. a= Pr(A,B,C,D,E)
            b= Pr(C,G,A,F)
            a.union(b) --> returns (Pr(A,B,C,D,E,1,1),numarray([A,1,C,1,1,G,F]))



        Notes:
        -    a and b remain unchanged
        -    a and b must be Table instances (or something equivalent)
        -    a always keeps the same order of its existing variables
        -    any new variables found in b are added at the end of a in the order
             they appear in b.
        -    new dimensions are added with numarray.newaxis
        -    the two numarrays objects returns have exactly the same dimensions
             and are ready for any kind of operation, *,/,...
        """
        # make a copy of a
        new = copy(a)

        for varb in b.names_list:
            # varb is the name of a variable in b
            if not new.assocdim.has_key(varb):
                new.addDim(varb) # add new variable to new

        # new now contains all the variables contained in a and b
        # new = A U B

        correspond = []
        b_assocdim = copy(b.assocdim)
        bcpt = b.cpt.view()
        for var in new.names_list:
            # var is the name of a variable in new
            if not b.assocdim.has_key(var):
                bcpt = bcpt[..., numpy.newaxis]
                b_assocdim[var] = bcpt.ndim - 1
            correspond.append(b_assocdim[var])

        # transpose dimensions in b to match those in a
        btr = numpy.transpose(bcpt, axes=correspond)

        # btr is now ready for any operation with new
        return new, btr

    def ones(self):
        """ All CPT elements are set to 1 """
        self.cpt = numpy.ones(self.cpt.shape, dtype=self.cpt.dtype)

    def zeros(self):
        """ All CPT elements are set to 0 """
        self.cpt = numpy.zeros(self.cpt.shape, dtype=self.cpt.dtype)

#=====================================================================
#=====================================================================

class TableTestCase(unittest.TestCase):
   def setUp(self):
      names = ('a', 'b', 'c')
      shape = (2, 3, 4)
      self.a = Table(names, shape, type='Float32')###X ones -> Table
      self.b = Table(names[1:], shape[1:], type='Float32')
      self.names = names
      self.shape = shape

   def testEq(self):
       a = Table(['a', 'b'], [2, 3], range(6), 'Float32')
       b = Table(['a', 'b'], [2, 3], range(6), 'Float32')
       c = Table(['a'], [6], range(6), 'Float32')
       d = numpy.arange(6).reshape((2, 3))
       e = Table(['b', 'a'], [3, 2], numpy.transpose(a.cpt))
       assert(a == b and \
              not a == c and \
              a == d and \
              a == e and e == a), \
                "__eq__ does not work"

   def testIMul(self):
       """ test inplace multiplication """
       b = Table(['c', 'b', 'e'], [4, 3, 6], range(12*6))
       c = Table(['a', 'b', 'c', 'd', 'e'], [2, 3, 4, 5, 6], \
                 range(2*3*4*5*6))

       bcpt = b.cpt[...,numpy.newaxis,numpy.newaxis]
       bcpt.transpose([3,1,0,4,2])
       res = bcpt*c.cpt

       c *= b

       assert (numpy.all(c.cpt == res)), \
              " InPlace Multiplication does not work"

   def testIDiv(self):
       """ test inplace division """
       b = Table(['c', 'b', 'e'], [4, 3, 6], range(12*6))
       c = Table(['a', 'b', 'c', 'd', 'e'], [2, 3, 4, 5, 6], range(2*3*4*5*6))

       bcpt = b.cpt[..., numpy.newaxis, numpy.newaxis]
       bcpt.transpose([3, 1, 0, 4, 2])
       res = c.cpt/bcpt
       res[numpy.isnan(res)] = 0.0

       c /= b

       assert (numpy.all(c.cpt == res)), \
              " InPlace Division does not work"

   def testMul(self):
       """ test multiplication """
       a = Table(['a', 'b', 'c', 'd'], [2, 3, 4, 5], range(2*3*4*5))
       b = Table(['c', 'b', 'e'], [4, 3, 6], range(12*6))
       c = Table(['a', 'b', 'c', 'd', 'e'], [2, 3, 4, 5, 6], range(2*3*4*5*6))

       acpt = a.cpt[..., numpy.newaxis]
       bcpt = b.cpt[..., numpy.newaxis, numpy.newaxis]
       bcpt = numpy.transpose(bcpt, [3, 1, 0, 4, 2])
       resab = acpt * bcpt

       ab = a * b
       cc = c * c
       bb = b * b

       assert (ab == Table(['a','b','c','d','e'], [2,3,4,5,6], resab) and \
               cc == Table(['a','b','c','d','e'], [2,3,4,5,6], \
                           numpy.arange(2*3*4*5*6)**2) and \
               bb == Table(['c','b','e'], [4,3,6], numpy.arange(12*6)**2)), \
              " Multiplication does not work"

   def testDiv(self):
       """ test division """
       a  = Table(['a','b','c','d'], [2,3,4,5], range(2*3*4*5))
       b = Table(['c','b','e'], [4,3,6], range(12*6))
       c = Table(['a','b','c','d','e'], [2,3,4,5,6], range(2*3*4*5*6))

       acpt = copy(a.cpt)[..., numpy.newaxis]
       bcpt = copy(b.cpt)[..., numpy.newaxis, numpy.newaxis]
       bcpt.transpose([3,1,0,4,2])

       ab = a/b
       cc = c/c
       bb = b/b

       cres = numpy.ones(2*3*4*5*6)
       cres[0] = 0
       bres = numpy.ones(12*6)
       bres[0] = 0
       ares = acpt/bcpt
       ares[numpy.isnan(ares)] = 0.0

       assert (ab == Table(['a','b','c','d','e'], [2,3,4,5,6], ares) and \
               cc == Table(['a','b','c','d','e'], [2,3,4,5,6], cres) and \
               bb == Table(['c','b','e'],[4,3,6], bres) ), \
              " Division does not work"

   def testDelegate(self):
       assert (numpy.alltrue(self.a.flat == self.a.cpt.flat)), \
              " Delegation does not work check __getattr__"

   def testBasicIndex(self):
      assert(self.a[1,1,1] == 1.0), \
            "Could not execute basic index 1,1,1 properly"

   def testDictIndex(self):
      index = dict(zip(self.names, (1,1,1)))
      assert(self.a[index] == self.a[1,1,1]), \
            "Dictionary Index is not equivalent to standard index"

   def testBasicSet(self):
      self.a[1,1,1] = 2.0
      assert(self.a[1,1,1] == 2), \
            "Could not set execute basic set 1,1,1 = 2"

   def testDictSet(self):
      index = dict(zip(self.names,(1,1,1)))
      self.a[index] = 3.0
      assert(self.a[index] == self.a[1,1,1] and \
             self.a[index] == 3.0), \
            "Dictionary Index not equivalent to normal index or could not set properly"

   def testAddDim(self):
        a = Table('a b c'.split())
        a.addDim('d')

        assert(a.names == set('a b c d'.split()) and \
               a.names_list == 'a b c d'.split() and \
               a.assocdim.has_key('d') and \
               a.assocname.has_key(3)), \
               "add Dim does not work correctly..."

   def testUnion(self):
        """ test Union between two Tables """
        a = Table(['a','b','c','d'], [2,3,4,5], range(2*3*4*5))
        b = Table(['c','b','e'], [4,3,6], range(12*6))

        ab, bb = a.union(b)

        assert(ab.names_list == ['a','b','c','d','e'] and \
               ab.shape == tuple([2,3,4,5,1]) and \
               numpy.all(bb == numpy.transpose(b.cpt[..., numpy.newaxis,numpy.newaxis], axes=[3,1,0,4,2]))), \
               """ union does not work ..."""

   def testPrepareOther(self):
        c = Table(['e','b'], [2,3], range(6))
        d = Table(['a','b','c','d','e'], [2,3,2,2,2], range(3*2**4))
        e = Table(['e','b','f'], [2,3,4], range(6*4))
        src = Table(['s','r','c'], [2,3,4], range(24))
        cr = Table(['c','r'], [4,3], range(12))

        dc = d.prepareOther(c)
        try:
            d.prepareOther(e)
            assert(0), """ this should produce an error..."""
        except:
            pass

        cr_ = src.prepareOther(cr)

        assert(dc.shape == tuple([1,3,1,1,2]) and \
               numpy.all(dc[0,:,0,0,:] == numpy.transpose(c.cpt, axes=[1,0])) and \
               cr_.shape == (1,3,4)), \
               """ problem with prepareOther"""


if __name__ == '__main__':
    suite = unittest.makeSuite(TableTestCase, 'test')
    runner = unittest.TextTestRunner()
    runner.run(suite)

    a = Table(['a','b'],[2,3],range(6))
    b = Table(['b'],[3],range(3))
    c = Table(['e','b'],[2,3],range(6))
    d = Table(['a','b','c','d','e'],[2,3,2,2,2],range(3*2**4))

    ac,cc = a.union(c)


##    a*c
##    print a
##    print c

    #a*b
    #print 'mul'
    #print a




