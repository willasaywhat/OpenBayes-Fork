#!/usr/bin/env python

'''Directed and undirected graph data structures and algorithms.

Copyright 2004, Robert Dick (dickrp@ece.northwestern.edu).
Please see the license at the end of the source code for legal
information.'''

__version__ = '0.2'
__author__ = 'Robert Dick (dickrp@ece.northwestern.edu)'


import copy, new, operator, struct, delegate

# The following four lines borrowed from Gregory R. Warnes's fpconst
# (until it's standard in Python distributions).
if struct.pack('i', 1)[0] != '\x01':
    PosInf = struct.unpack('d', '\x7F\xF0\x00\x00\x00\x00\x00\x00')[0]
else:
    PosInf = struct.unpack('d', '\x00\x00\x00\x00\x00\x00\xf0\x7f')[0]


class GraphError(StandardError):
    '''Exception for graph errors.'''
    pass


class SortStruct(float, object):
    '''Throw-away class for temporarily holding values.

    Initialize with a float distance and as many keyword args as desired.
    The keyword args become attributes.'''

    def __new__(cls, val, **kargs):
        obj = float.__new__(cls, val)
        obj.__dict__ = kargs
        return obj

    @property
    def number(self): return float(self)


def _condop(cond, v1, v2):
    if cond:
        return v1
    else:
        return v2

def _roprop(description = None):
    def prop_func(ro_method):
        return property(ro_method, None, None, description)
    return prop_func

#=======================================================================

class Vertex(delegate.Delegate):
    '''General graph vertex.

    All methods colaborate with directed and undirected
    edges.  If the edges are undirected, ine == oute.

    Variables :
    --------------------
        self._e = list of edges of this vertex
        self.in_e = list of incoming edges (decorator)
        self.out_e = list of outgoing edges (decorator)
        self.all_e = list of all edges = self._e (decorator)

    '''

    def attach_e(self, e):
        '''Attach an edge.'''
        self._e.append(e)

    def connecting_e(self, v):
        '''List of edges connecting self and other vertex.'''
        return [e for e in self._e if e.leaves(self) and e.enters(v)]

    def __init__(self, name):
        '''Name needn't be a string but must be hashable
        and immutable.
        '''
        self.__Delegate.__init__(self)
        self._e = []
        self.name = name

    def __str__(self): return str(self.name)

    def __getstate__(self):
        '''Need to break cycles to prevent recursion blowup in pickle.

        Dump everything except for edges.'''
        dcp = copy.copy(self.__dict__)
        dcp['_e'] = []
        return dcp

    @_roprop('All edges.')
    def all_e(self): return self._e

    @_roprop('Incoming edges.')
    def in_e(self): return [e for e in self._e if e.enters(self)]

    @_roprop('Outgoing edges.')
    def out_e(self): return [e for e in self._e if e.leaves(self)]

    @_roprop('Set of adjacent vertices.  Edge direction ignored.')
    def adjacent_v(self):
        adj = set(v for e in self._e for v in e.all_v)
        try: adj.remove(self)
        except: pass
        return adj

    @_roprop('Set of vertices connected by incoming edges.')
    def in_v(self):
        return set(v for e in self._e for v in e.src_v if e.enters(self))

    @_roprop('Set of vertices connected by outgoing edges.')
    def out_v(self):
        return set(v for e in self._e for v in e.dest_v if e.leaves(self))

    @_roprop('True if vertex has no outgoing edges.')
    def is_sink(self): return not self.out_e

    @_roprop('True if vertex has no incoming edges.')
    def is_src(self): return not self.in_e

    @_roprop('True if vertex has incoming and outgoing edges.')
    def is_intermed(self): return not self.is_sink and not self.is_src

    def __eq__(a,b):
        return hash(a) == hash(b)

#=======================================================================

class RawEdge(delegate.Delegate):
    '''Base class for undirected and directed edges.
    Not directly useful.
    '''

    def __init__(self, name, v1, v2):
        '''Name needn't be a string but must be immutable
        and hashable.
        '''
        self.__Delegate.__init__(self)
        self.name = name
        self._v = [v1, v2]
        v1.attach_e(self)
        v2.attach_e(self)

    def __setstate__(self, state):
        '''Restore own state and add self to connected vertex edge
        lists.
        '''
        self.__dict__ = state
        self._v[0].attach_e(self)
        self._v[1].attach_e(self)

    @_roprop('All connected vertices.')
    def all_v(self): return self._v


class UndirEdge(RawEdge):
    '''Undirected edge.'''

    def enters(self, v):
        '''True if this edge is connected to the vertex.'''
        return v in self._v

    def leaves(self, v):
        '''True if this edge is connected to the vertex.'''
        return v in self._v

    def __str__(self):
        return '%s: %s -- %s' % (str(self.name), str(self._v[0]),
            str(self._v[1]))

    def weight(self, v1, v2):
        '''1 if this edge connects the vertices.'''
        if v1 not in self._v or v2 not in self._v:
                    raise GraphError('vertices not connected')
        return 1

    @_roprop('Source vertices.')
    def src_v(self): return self._v

    @_roprop('Destination vertices.')
    def dest_v(self): return self._v


class DirEdge(RawEdge):
    '''Directed edge.

        Variables :
        --------------
        self._v[0] = source vertex
        self._v[1] = destination vertex
    '''

    def enters(self, v):
        '''True only if the vertex is this edge's destination.'''
        return v is self._v[1]

    def leaves(self, v):
        '''True only if the vertex is this edge's source.'''
        return v is self._v[0]

    def invert(self):
        '''Inverts the edge direction.'''
        self._v.reverse()

    def weight(self, v1, v2):
        '''1 if this edge has v1 as its source and v2 as
        its destination.'''
        if v1 is not self._v[0] or v2 is not self._v[1]:
            raise GraphError('vertices not connected')
        return 1

    def __str__(self):
        return '%s: %s -> %s' % (str(self.name), str(self._v[0]),
                                 str(self._v[1]))

    @_roprop(
            '''Single element list containing the edge's source vertex.

            Must be a list to conform with UndirEdge's interface.''')
    def src_v(self): return [self._v[0]]

    @_roprop(
            '''Single element list containing the edge's destination
            vertex.

            Must be a list to conform with UndirEdge's interface.
            ''')
    def dest_v(self): return [self._v[1]]

#=======================================================================

def DataWrapped(cls):
    '''Returns a wrapped class with a new 'data' member.'''

    class __Wr(cls):
        '''Wrapper to add 'data' to another class.'''

        def __init__(self, *pargs, **kargs):
            '''Use last parg for data and sends all other args to
            base.
            '''
            if kargs:
                cls.__init__(self, *pargs[:-1], **kargs)
            else:
                cls.__init__(self, *pargs[:-1])
            self.data = pargs[-1]

        def __str__(self):
            '''Append data to base's __str__.'''
            return ' '.join(filter(None, (cls.__str__(self), str(self.data))))

    __Wr.__name__ = 'Data%s' % cls.__name__.split('.')[-1]
    return __Wr

DataVertex = DataWrapped(Vertex)
DataUndirEdge = DataWrapped(UndirEdge)
DataDirEdge = DataWrapped(DirEdge)

#=======================================================================

class VertexDict(dict, delegate.Delegate):
    '''Dictionary of vertices.'''

    def __init__(self, graph):
        self.__dict.__init__(self)
        self.__Delegate.__init__(self)
        self.graph = graph

    def __setitem__(self, key, val):
        if self.has_key(key):
            raise KeyError('VertexDict already has (%s, %s).' % (key, val))
        self.__dict.__setitem__(self, key, val)

    def __delitem__(self, key):
        '''Delete all edges connected to the vertex, along with the
        vertex.
        '''
        v = self[key]
        for e in reversed(v._e):
                    del self.graph.e[e.name]
        self.__dict.__delitem__(self, key)


class EdgeDict(dict, delegate.Delegate):
    '''Dictionary of edges.'''

    def __setitem__(self, key, val):
        if self.has_key(key):
            raise KeyError('EdgeDict already has (%s, %s).' % (key, val))
        self.__dict.__setitem__(self, key, val)

    def __delitem__(self, key):
        '''Remove edge from vertices to which it is attached.'''
        e = self[key]
        for v in e._v:
            v._e.remove(e)
        self.__dict.__delitem__(self, key)

#=======================================================================

class Graph(delegate.Delegate):
    '''General-purpose graph data structure.'''

    def __init__(self, name = None):
        self.name = name
        self.v = VertexDict(self)
        self.e = EdgeDict()

    def __str__(self):
        return self.__class__.__name__ + \
                    _condop(self.name, ' ' + str(self.name), '') + \
                    '\nVertices:\n' + \
                    '\n'.join([str(v) for v in self.v.values()]) + \
                    '\n\nEdges:\n' + \
                    '\n'.join([str(e) for e in self.e.values()]) + '\n'

    @_roprop('List of all vertices.')
    def all_v(self): return [v for n, v in self.v.items()]

    @_roprop('List of all vertices without incoming edges.')
    def src_v(self): return [v for n, v in self.v.items() if v.is_src]

    @_roprop('List of all vertices without outgoing edges.')
    def sink_v(self): return [v for n, v in self.v.items() if v.is_sink]

    @_roprop(
            'List of all vertices with both incoming and outgoing edges.')
    def intermed_v(self):   return [v for n, v in self.v.items() if v.is_intermed]

    def add_v(self, v):
        '''Add and return a vertex.'''
        self.v[v.name] = v
        return v

    def add_e(self, e):
        '''Add and return an edge.'''
        self.e[e.name] = e
        return e

    def del_e(self, e):
        '''delete an edge (e is the name of the edge)'''
        assert(self.e.has_key(e)), "The edge is not in the BNet"
        del self.e[e]

    def connex_components(self):
        """ returns a list of list of nodes that are connected between
         them"""
        unchecked = set(self.v.values())
        groups = []
        while len(unchecked):
            vcon = self.member_family(unchecked.pop())
            unchecked -= set(vcon)
            groups.append(set(vcon))
        return groups

    def member_family(self, node):
        unprocessed = [node]
        visited = []
        while unprocessed:
            v = unprocessed.pop()
            if v not in visited:
                visited.append(v)
                unprocessed.extend(v.out_v)
                unprocessed.extend(v.in_v)
        return visited

    def connected_components(self):
        '''Return a list of lists.  Each holds transitively-connected vertices.'''
        unchecked = set(self.v.values())
        groups = []
        while len(unchecked):
            vcon = self.depth_first_search(unchecked.pop())
            unchecked -= set(vcon)
            groups.append(vcon)
        return groups

    @staticmethod
    def depth_first_search(start_v):
        '''Return a depth-first search list of vertices.'''
        unprocessed = [start_v]
        visited = []
        while unprocessed:
            v = unprocessed.pop()
            if v not in visited:
                visited.append(v)
                unprocessed.extend(v.out_v)
        return visited

    @staticmethod
    def breadth_first_search(start_v):
        '''Return a breadth-first search list of vertices.'''
        unprocessed = [start_v]
        visited = []
        while unprocessed:
            v = unprocessed.pop(0)
            if v not in visited:
                visited.append(v)
                unprocessed.extend(v.out_v)
        return visited

    #@staticmethod
    def topological_sort(self):
        '''Return a topological sort list of vertices.'''
        # unprocessed is a list of all nodes that have no parents
        unprocessed = [v for v in self.v.values() if not v.in_v]

        return Graph.topological_sort_by_node(unprocessed)

    @staticmethod
    def topological_sort_by_node(start_v):
        '''Return a topological sort list of vertices.'''
        unprocessed = start_v
        visited = []
        while unprocessed:
            v = unprocessed.pop(0)
            incoming_v = v.adjacent_v - v.out_v
            if v not in visited and not (incoming_v - set(visited)):
                visited.append(v)
                unprocessed.extend(v.out_v)
        return visited

    def HasNoCycles(self, start_v):
        ''' Return True if the node start_v is not in a cycle
        '''
        unprocessed = [start_v]
        visited = []
        result = True
        i = 0
        while unprocessed:
            i += 1
            v = unprocessed.pop(0)
            if i == 1:
                unprocessed.extend(v.out_v)
            elif start_v in visited:
                result = False
                break
            elif v not in visited:
                visited.append(v)
                unprocessed.extend(v.out_v)
        return result

##            def HasNoCycles(self, start_v):
##                ''' Return True if the node start_v is not in a cycle
##                '''
##                unprocessed = [start_v]
##                visited = []
##                result = True
##                i = 0
##                while unprocessed:
##                            i += 1
##                            v = unprocessed.pop(0)
##                            if v not in visited:
##                                visited.append(v)
##                                unprocessed.extend(v.out_v)
##                            else:
##                                result = False
##                                break
##                return result

#================================================================================

    def minimal_span_tree(self, **kargs):
        '''Return minimal spanning 'Tree'.

        Keyword arg 'weight_func' is a function taking (edge, v1, v2) and
        returning a weight.  Defaults to e.weight().
        'targets' list of target vertices.  Defaults to all vertices.'''

        def def_weight_func(e, v1, v2): return e.weight(v1, v2)
        weight_func = kargs.get('weight_func', def_weight_func)
        targets = kargs.get('targets', self.v.values())
        visited = set([targets.pop()])
        unvisited = set(self.v.values()) - visited
        mst = Tree()

        while targets:
# Haven't found it yet.  Search more.
            connected = []
            for v in visited:
                for u in unvisited:
                    conn = v.connecting_e(u)
                    if conn:
                        dist = weight_func(conn.pop(), v, u)
                        connected.append(SortStruct(dist, src=v, dest=u))

            if not connected:
                raise GraphError('unreachable vertices in minimal_spanning_tree')

# Connect it
            near = min(connected)
            if near.src.name not in mst.v:
                mst.add_v(DataVertex(near.src.name, near.src))
            if near.dest.name not in mst.v:
                mst.add_v(DataVertex(near.dest.name, near.src))
            e = mst.auto_add_e(DirEdge(len(mst.e), mst.v[near.src.name], \
                               mst.v[near.dest.name]))

            visited.add(near.dest)
            unvisited.remove(near.dest)
            targets.remove(near.dest)
        return mst

#================================================================================

    def shortest_tree(self, start, **kargs):
        '''Return a 'Tree' of shortest paths to all nodes.

        Keyword arg 'weight_func' is a function taking (edge, v1, v2) and
        returning a weight.  Defaults to e.weight().
        'targets' list of target vertices.  Defaults to all vertices.'''

        def def_weight_func(e, v1, v2): return e.weight(v1, v2)
        weight_func = kargs.get('weight_func', def_weight_func)
        targets = set(kargs.get('targets', self.v.values()))
        path_tr = Tree()
        dist = {start:0.0}
        unvisited = set(self.v.values())
        while targets:
# Determine the closest vertex
            closest = min([(dist.get(v, PosInf), v) for v in unvisited])[1]
# Add it and push the distances
            unvisited.remove(closest)
            targets.discard(closest)
            adj_v = closest.out_v
            if not adj_v - unvisited:
                raise GraphError('unreachable vertices in shortest_tree')
            for v in adj_v:
                for e in closest.connecting_e(v):
                    push_dist = weight_func(e, closest, v) + dist[closest]
                    if push_dist < dist.get(v, PosInf):
                        dist[v] = push_dist
# Add the vertices.  Remove the old edges, if any.  Add the new edge.
                        if closest.name not in path_tr.v:
                            path_tr.add_v(DataVertex(closest.name, closest))
                        if v.name not in path_tr.v:
                            path_tr.add_v(DataVertex(v.name, v))
                        e_nm = len(path_tr.e)
                        for e2 in path_tr.v[v.name].in_e:
                            del path_tr.e[e2.name]
                            e_nm = e2.name
                        path_tr.auto_add_e(DirEdge(e_nm, path_tr.v[closest.name],
                                                   path_tr.v[v.name]))
        return path_tr

#================================================================================

    def greedy_paths(self, start, goal, weight_func=None):
        '''Return a dict of greedy paths with (start vertex, end vertex) keys.

        Always makes the highest-gain decision.  Will find a path if one exists.
        Not necessarily optimal.  'weight_func' is a function of (edge, v1, v2)
        returning a weight.  Defaults to e.weight()'''

        def def_weight_func(e, v1, v2): return e.weight(v1, v2)
        weight_func = weight_func or def_weight_func
        path = [start]
        visited = set([start])
        while path[-1] is not goal:
            adj_v = [SortStruct(weight_func(e, path[-1], v), dest = v)
                     for e in path[-1].out_e for v in e.dest_v if v not in visited]
            if adj_v:
                closest_v = min(adj_v).dest
                visited.add(closest_v)
                path.append(closest_v)
            else:
                path.pop()
# Prepare the dict
        d = {}
        for i1 in range(len(path)):
            for i2 in range(i1, len(path)):
                d[path[i1], path[i2]] = path[i1:i2 + 1]
        return d

    def all_pairs_sp(self, weight_func=None):
            '''Return a dictionary of shortest path lists for all vertex pairs.

            Keys are (source, destination) tuples.
            'weight_func' is a function taking (edge, v1, v2) that returns a weight.
            Defaults to e.weight()'''

            return dict(self.shortest_tree(v,
                        weight_func = weight_func).path_dict().items() for v in self.v.values())

#================================================================================

    @staticmethod
    def path_weight(path, weight_func=None):
        '''Return the weight of the path, which is a list of vertices.

        'weight_func' is a function taking (edge, v1, v2) and returning a weight.'''

        def def_weight_func(e, v1, v2): return e.weight(v1, v2)
        weight_func = weight_func or def_weight_func
        wt = 0.0
        for v1, v2 in zip(path[:-1], path[1:]):
            connect_e = [e for e in v1.out_e if e.enters(v2)]
            if not connect_e:
                raise GraphError('vertices in path are not connected')
            wt += min(weight_func(e, v1, v2) for e in connect_e)
        return wt

#=======================================================================
#=======================================================================
#                   MY CODE !!!!!!!!!!!!!!! (Kostas Gaitanis)
#=======================================================================
#=======================================================================


class TreeError(GraphError): pass

class Tree(Graph):
    '''Tree data structure.

    Edges must be directed and reconvergent paths do not occur.'''

    def add_e(self, e):
        '''Add edge if tree invariant holds.  Otherwise, raise exception.'''
        if len(e.dest_v) != 1:
            raise TreeError('undirected edge')
        if len(e.dest_v[0].in_e) != 1:
            raise TreeError('edge introduces reconvergent paths into tree')
        return self.__Graph.add_e(self, e)

    def auto_add_e(self, e):
        '''Automatically choose correct direction for new edge.'''
        if len(e.dest_v) != 1:
            raise TreeError('undirected edge')
        if len(e.dest_v[0].in_e) != 1:
            e.invert()
        self.add_e(e)

    def is_safe_e(self, target_v):
        '''True if edge maintains tree invariant.'''
        return not target_v.in_e

    def path_dict(self):
        '''Return a dictionary of path lists from the root to each vertex.

        Keys are (source vertex, destination vertex) tuples.'''

        root = self.src_v
        assert len(root) == 1
        root = root.pop()

        d = {(root.data, root.data):[root.data]}
        for v in self.depth_first_search(root):
            parent = v.in_v
            assert len(parent) <= 1
            if parent:
                parent = parent.pop()
                d[root.data, v.data] = d[root.data, parent.data] + [v.data]
        return d

########################################################################
if __name__ == '__main__':
    from OpenBayes import BVertex
    G = Graph()
    a, b, c, d, e, f, g = [G.add_v(BVertex(nm)) for nm in 'a b c d e f g'.split()]
    for ep in [(a,b), (a,c), (b,d), (b,f), (b,e), (c,e), (d,f), (e,f),
    (f, g)]:
        G.add_e(UndirEdge(len(G.e), *ep))

    print G
    #print 'DFS:', map(str, G.depth_first_search(a))
    #print 'BFS:', map(str, G.breadth_first_search(a))
    #print 'top sort:', map(str, G.topological_sort(a))

    #T = G.minimal_span_tree()
    #print T
    #print [(map(str, k), map(str, v)) for k, v in T.path_dict().items()]

    #S = G.shortest_tree(a)
    #print S

    print a

    ###################################################################
    # just in case, this is the copy method I created, but don't seem to need it
    ######################################
    def __copy__(self):
        g = Graph('Copy of ' + str(self.name))

        # copy vertices
        for v in self.v.values():
            if v.__class__.__name__ == 'BVertex':
                g.add_v(BVertex(v.name, v.nvalues))
            elif v.__class__.__name__ == 'Vertex':
                g.add_v(Vertex(v.name))


        # for each edge, create a corresponding edge in g
        for e in self.e.values():
            print e
            print e._v[0].name
            v1 = g.v[e._v[0].name]
            v2 = g.v[e._v[1].name]
            if e.__class__.__name__ == 'UndirEdge':
                g.add_e(UndirEdge(len(g.e),v1,v2))
            elif e.__class__.__name__ == 'DirEdge':
                g.add_e(DirEdge(len(g.e),v1,v2))

        return g


#!/usr/bin/env python

'''Class to automate delegation decisions based on inheritance graph.

Copyright 2004, Robert Dick (dickrp@ece.northwestern.edu).

Whenever you need to delegate to something, inherit from delegate and use
self.__<base>.<method()> to access the base.  If the delegation was
inappropriate due to reconverging paths in the inheritance graph, the return
value will be None.  In the case of reconverging paths, the left-most call in
the method resolution order will be honored.  The rest will be nulified.  You
can also check to see if the base is the no_delegation object.  Delegate to all
your bases if you need everything in the inheritance graph to be visited.  As
long as one of a class's (transitive) bases inherits from Delegate, that's
enough.

For examples of use, please see the delegate.py file.

Python doesn't yet automate meta-class instantiation.  If you need to inherit
from Delegate and another class that does not have a 'type' metaclass, you'll
need to generate a shared derived metaclass and explicitly use that as your
class's metaclass.  For example:

    import Delegate, qt

    class sip_meta_join(type(Delegate), type(qt.QObject)):
    def __init__(*args):
        type(Delegate).__init__(*args)
        type(qt.QObject).__init__(*args)

    class MyClass(Delegate, qt.QObject):
    __metaclass__ = sip_meta_join
    ...

Please see the license at the end of the source code for legal information.'''


__version__ = '0.1'
__author__ = 'Robert Dick (dickrp@ece.northwestern.edu)'


def should_call(obj, pos, supr):
    '''Returns bool.  Should 'self' delegate to 'super' at 'pos'?
    Determines whether pos is left-most derived of super in MRO.'''

    for c in type(obj).__mro__:
        if supr in c.__bases__:
            return pos is c
    return False

class _no_delegation(object):
    '''All class's attributes are null callable's.'''

    _to_base = set(['__bases__', '__name__', '__mro__', '__module__'])

    def __getattribute__(self, attr):
        if attr in _no_delegation._to_base:
            return getattr(object, attr)
        def no_action(*pargs, **kargs): pass
        return no_action

'''Whatever'''
no_delegation = _no_delegation()
'''Whatever'''


class _delegate_meta(type):
    '''Sets up delegation private variables.

    Traverses inheritance graph on class construction.  Creates a private
    __base variable for each base class.  If delegating to the base class is
    inappropriate, uses _no_delegation class.'''

    def __init__(cls, name, bases, dict):
        type.__init__(cls, name, bases, dict)
        visited_supr = set()
        for sub in cls.__mro__[:-1]:
            subnm = sub.__name__.split('.')[-1]
            for supr in sub.__bases__:
                suprnm = supr.__name__.split('.')[-1]
                if supr not in visited_supr:
                    visited_supr.add(supr)
                    deleg = supr
                else:
                    deleg = no_delegation
                    setattr(cls, '_%s__%s' % (subnm, suprnm), deleg)


class Delegate(object):
    '''Inherit from Delegate to get delegation variables on class construction.'''
    __metaclass__ = _delegate_meta


if __name__ == '__main__':
    class Base(Delegate):
        def __init__(self, basearg):
            self.__Delegate.__init__(self)
            self.basearg = basearg
            print 'base'

        def __str__(self): return 'BASE'


    class Left(Base):
        def __init__(self, basearg, leftarg):
            self.__Base.__init__(self, basearg)
            self.leftarg = leftarg
            print 'left'

        def __str__(self):
            return ' '.join(filter(None, (self.__Base.__str__(self), 'LEFT')))


    class Right(Base):
        def __init__(self, basearg):
            self.__Base.__init__(self, basearg)
            print 'right'

        def __str__(self):
            return ' '.join(filter(None, (self.__Base.__str__(self), 'RIGHT')))


    class Der(Left, Right):
        def __init__(self, basearg, leftarg):
            self.__Left.__init__(self, basearg, leftarg)
            self.__Right.__init__(self, basearg)
            print 'der'

        def __str__(self):
            return ' '.join(filter(None, (self.__Left.__str__(self), \
                                          self.__Right.__str__(self), 'DER')))


    print 'should print base, left, right, der'
    der = Der('basearg', 'leftarg')

    print '\nshould print base, left'
    left = Left('basearg', 'leftarg')

    print '\nshould print base right'
    right = Right('basearg')

    print '\nshould print BASE LEFT RIGHT DER'
    print der


