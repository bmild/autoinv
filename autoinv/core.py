import warnings
from contextlib import contextmanager
import inspect
from inspect import isfunction
import types
import itertools

import autoinv.numpy.numpy_wrapper as anp
import numpy as _np

def trace(fun, x):
    with trace_stack.new_trace() as t:
        print 'New trace with value {}'.format(t)
        x.node = anp.Node(None, None, None, None, [], [])
    	x._trace = t
        end_box = fun(x)
        if isinstance(end_box, anp.Box) and end_box._trace == t:
            print 'Trace successful'
            return end_box
        else:
            warnings.warn("Output seems independent of input.")
            return end_box, None

def make_inverse(fun, x):
    end_box = trace(fun, x)
    def inv(g): return backward_pass(g, end_box.node)
    return inv


primitive_invs = {}
def definv(fun, *invs):
    def inv(argnum):
        assert argnum < len(invs)
        return invs[argnum]
    primitive_invs[fun] = inv

def definv_argnum(fun, argnum_inv):
    primitive_invs[fun] = argnum_inv



#############################


definv(anp.add,         lambda ans, x, y : lambda g : g - y,
                        lambda ans, x, y : lambda g : g - x)
definv(anp.multiply,    lambda ans, x, y : lambda g : g / y,
                        lambda ans, x, y : lambda g : g / x)
definv(anp.subtract,    lambda ans, x, y : lambda g : g + y,
                        lambda ans, x, y : lambda g : x - g)
definv(anp.divide,      lambda ans, x, y : lambda g : g * y,
                        lambda ans, x, y : lambda g : x / g)
definv(anp.power,
    lambda ans, x, y : lambda g : anp.power(anp.maximum(0, g), 1./y), 
    lambda ans, x, y : lambda g : anp.log(g) / anp.log(x))

######


definv(anp.negative, lambda ans, x: lambda g: -g)
definv(anp.exp,    lambda ans, x : lambda g: anp.log(g))
definv(anp.log,    lambda ans, x : lambda g : anp.exp(g))
definv(anp.square, lambda ans, x : lambda g : anp.sqrt(g))
definv(anp.sqrt,   lambda ans, x : lambda g : anp.square(g))


#####

definv(anp.swapaxes, lambda ans, x, axis1, axis2: lambda g: anp.swapaxes(g, axis2, axis1))
definv(anp.roll,     lambda ans, x, shift, axis=None  : lambda g: anp.roll(g, -shift, axis=axis))
definv(anp.reshape,  lambda ans, x, shape, order=None : lambda g: anp.reshape(g, anp.shape(x), order=order))
definv(anp.expand_dims, lambda ans, x, axis     : lambda g: anp.reshape(g, anp.shape(x)))
definv(anp.squeeze, lambda ans, x, axis=None    : lambda g: anp.reshape(g, anp.shape(x)))
definv(anp.flipud,  lambda ans, x,              : lambda g: anp.flipud(g))
definv(anp.fliplr,  lambda ans, x,              : lambda g: anp.fliplr(g))
definv(anp.moveaxis, lambda ans, a, source, destination: lambda g:
                    anp.moveaxis(g, destination, source))


definv(anp.matmul, None, lambda ans, A, B: anp.primitive(lambda g : _np.linalg.solve(A, g)))
definv(anp.tensordot, None, lambda ans, A, B, axes=2: anp.primitive(lambda g: _np.linalg.tensorsolve(A, g, axes)))

def inv_transpose(ans, x, axes=None):
    axes0 = axes
    if axes is not None:
        axes = anp.argsort(axes)
    return lambda g : anp.transpose(g, axes)
definv(anp.transpose, inv_transpose)

def inv_concatenate_args(argnum):
    assert argnum >= 1
    def inv(ans, axis, *args):
        sizes = [anp.shape(a)[axis] for a in args[:argnum]]
        start = sum(sizes[:-1])
        idxs = [slice(None)] * ans.ndim
        print 'Concatenate inv. Slicing from {} to {} on axis {}.'.format(start, sizes[-1], axis)
        idxs[axis] = slice(start, start + sizes[-1])
        return lambda g: g[idxs]
    return inv
definv_argnum(anp.concatenate_args, definv_argnum)

#############################



def nodeinv(node):
    if node.fun is None:
        return lambda y : y
    else:
        assert len(node.parent_argnums) == 1
        argnum = node.parent_argnums[0] if node.parent_argnums else 0
        # print 'inv is at argnum {}'.format(argnum)
        if node.fun not in primitive_invs:
            raise NotImplementedError('Inverse of {} not implemented for node {}'.format(node.fun.__name__, node))
        return primitive_invs[node.fun](argnum)(node.value, *node.args, **node.kwargs)


def backward_pass(y, end_node):
    outgrads = {end_node : y}
    for node in toposort(end_node):
        outgrad = outgrads.pop(node)
        # print 'Running inv for node {}'.format(node)
        print 'Running inv for node {}, curr shape is {}'.format(node, outgrad.shape)
        ingrads = nodeinv(node)(outgrad)
        # print 'New shape {}'.format(ingrads.shape)
        for parent in node.parents:
            outgrads[parent] = ingrads # add_outgrads(outgrads.get(parent), ingrad)
    return outgrad


class TraceStack(object):
    def __init__(self):
        self.top = -1
    @contextmanager
    def new_trace(self):
        self.top += 1
        yield self.top
        self.top -= 1
trace_stack = TraceStack()


def toposort(end_node):
    child_counts = {}
    stack = [end_node]
    while stack:
        node = stack.pop()
        if node in child_counts:
            child_counts[node] += 1
        else:
            child_counts[node] = 1
            stack.extend(node.parents)

    childless_nodes = [end_node]
    while childless_nodes:
        node = childless_nodes.pop()
        yield node
        for parent in node.parents:
            if child_counts[parent] == 1:
                childless_nodes.append(parent)
            else:
                child_counts[parent] -= 1

                
def print_trace(end_node):
    for node in toposort(end_node):
        print node



# nondiff_methods = ['all', 'any', 'argmax', 'argmin', 'argpartition',
#                    'argsort', 'nonzero', 'searchsorted', 'round']
# diff_methods = ['clip', 'compress', 'cumprod', 'cumsum', 'diagonal',
#                 'max', 'mean', 'min', 'prod', 'ptp', 'ravel', 'repeat',
#                 'reshape', 'squeeze', 'std', 'sum', 'swapaxes', 'take',
#                 'trace', 'transpose', 'var']
# for method_name in nondiff_methods + diff_methods:
# 	# pass
#     setattr(Box, method_name, local_anp[method_name])