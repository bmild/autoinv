from __future__ import absolute_import
from ..core import definv
import ..numpy as anp


def inv_reshape(ans, x, shape, order=None):
    print 'Inverting reshape to {} using original shape {}'.format(shape, anp.shape(x))
    return lambda g : anp.reshape(g, anp.shape(x), order=order)
definv(anp.reshape, inv_reshape)


def grad_transpose(ans, x, axes=None):
    axes0 = axes
    if axes is not None:
        axes = anp.argsort(axes)
    print 'Inverting transpose {} using {}'.format(axes0, axes)
    return lambda g : anp.transpose(g, axes)
definv(anp.transpose, grad_transpose)