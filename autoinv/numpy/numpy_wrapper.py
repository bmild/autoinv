import warnings
from contextlib import contextmanager
import numpy as _np
import inspect
from inspect import isfunction
import types
import itertools


def wraps(fun, namestr="{fun}", docstr="{doc}", **kwargs):  
    def _wraps(f):
        try:
            f.__name__ = namestr.format(fun=get_name(fun), **kwargs)
            f.__doc__ = docstr.format(fun=get_name(fun), doc=get_doc(fun), **kwargs)
        finally:
            return f
    return _wraps   

get_name = lambda f: getattr(f, '__name__', '[unknown name]')
get_doc  = lambda f: getattr(f, '__doc__' , '')
        
        
def primitive(f_raw):
    """
    Wraps a function so that its inverse can be specified and its invocation
    can be recorded."""
    @wraps(f_raw)
    def f_wrapped(*args, **kwargs):

        _args = []
        for arg in args:
            if isinstance(arg, Box):
                _args.append(arg.view(_np.ndarray))
            else:
                _args.append(arg)

        value = f_raw(*_args, **kwargs)

        boxes, trace = find_top_boxed_args(args)

        parent_argnums = [a for a, _ in boxes]
        parents = [b.node for _, b in boxes]

        if isinstance(value, _np.ndarray):
            value = value.view(Box)
            # print 'Boxing for function {}, # parents is {}'.format(f_raw.__name__, len(parents))
            # node_type = type(boxes[0].node)
            value.node = Node(value, f_wrapped, args, kwargs, parent_argnums, parents)
            value._trace = trace
        return value
    return f_wrapped


class Box(_np.ndarray):
    __slots__ = ['node', '_trace']
    def __init__(self):
        self._trace = -1

    # @primitive
    # def __getitem__(A, idx): return A[idx]
    __getitem__ = primitive(_np.ndarray.__getitem__)

    T = property(lambda self: transpose(self))

    def __neg__(self): return negative(self)
    def __add__(self, other): return add(     self, other)
    def __sub__(self, other): return subtract(self, other)
    def __mul__(self, other): return multiply(self, other)
    def __pow__(self, other): return power   (self, other)
    def __div__(self, other): return divide(  self, other)
    def __mod__(self, other): return mod(     self, other)
    def __truediv__(self, other): return true_divide(self, other)
    def __matmul__(self, other): return matmul(self, other)
    def __radd__(self, other): return add(     other, self)
    def __rsub__(self, other): return subtract(other, self)
    def __rmul__(self, other): return multiply(other, self)
    def __rpow__(self, other): return power(   other, self)
    def __rdiv__(self, other): return divide(  other, self)
    def __rmod__(self, other): return mod(     other, self)
    def __rtruediv__(self, other): return true_divide(other, self)
    def __rmatmul__(self, other): return matmul(other, self)
    def __eq__(self, other): return equal(self, other)
    def __ne__(self, other): return not_equal(self, other)
    def __gt__(self, other): return greater(self, other)
    def __ge__(self, other): return greater_equal(self, other)
    def __lt__(self, other): return less(self, other)
    def __le__(self, other): return less_equal(self, other)
    def __abs__(self): return abs(self)


class Node():
    newid = itertools.count().next
    
    def __init__(self, value, fun, args, kwargs, parent_argnums, parents):
        self.value = value
        self.fun = fun
        self.args = args
        self.kwargs = kwargs
        self.parent_argnums = parent_argnums
        self.parents = parents
        self.id = Node.newid()
        
    def __repr__(self):
        if self.fun is None:
            return '<Start node>'
        else:
            return '<Node {} for fn {}. Called with Nodes {} in positions {}>'.format(
                self.id,
                self.fun.__name__,
                [n.id for n in self.parents],
                self.parent_argnums,
            )
   

def find_top_boxed_args(args):
    top_trace = -1
    top_boxes = []
    for argnum, arg in enumerate(args):
        if isinstance(arg, Box):
            trace = arg._trace
            if trace > top_trace:
                top_boxes = [(argnum, arg)]
                top_trace = trace
            elif trace == top_trace:
                top_boxes.append((argnum, arg))
    return top_boxes, top_trace 
    



def wrap_namespace(old, new):
    unchanged_types = {float, int, type(None), type}
    int_types = {_np.int, _np.int8, _np.int16, _np.int32, _np.int64, _np.integer}
    function_types = {_np.ufunc, types.FunctionType, types.BuiltinFunctionType}
    for name, obj in old.items():
        
        if type(obj) in function_types:
            new[name] = primitive(obj)
        else:
            new[name] = obj


        # if obj in notrace_functions:
        #     new[name] = notrace_primitive(obj)
        # elif type(obj) in function_types:
        #     new[name] = primitive(obj)
        # elif type(obj) is type and obj in int_types:
        #     new[name] = wrap_intdtype(obj)
        # elif type(obj) in unchanged_types:
        #     new[name] = obj

wrap_namespace(_np.__dict__, globals())

@primitive
def concatenate_args(axis, *args):
    return _np.concatenate(args, axis).view(ndarray)
concatenate = lambda arr_list, axis=0 : concatenate_args(axis, *arr_list)


# nondiff_methods = ['all', 'any', 'argmax', 'argmin', 'argpartition',
#                    'argsort', 'nonzero', 'searchsorted', 'round']
# diff_methods = ['clip', 'compress', 'cumprod', 'cumsum', 'diagonal',
#                 'max', 'mean', 'min', 'prod', 'ptp', 'ravel', 'repeat',
#                 'reshape', 'squeeze', 'std', 'sum', 'swapaxes', 'take',
#                 'trace', 'transpose', 'var']
# for method_name in nondiff_methods + diff_methods:
#   # pass
#     setattr(Box, method_name, local_anp[method_name])