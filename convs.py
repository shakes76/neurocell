'''
Conv module for all convolution algorithms
'''
import jax
import jax.numpy as jnp

#JIT compiled members for performance
from functools import partial

@partial(jax.jit, static_argnums=(2,3,4,5))
def _filter(cells, nethood, dn, window_strides, lhs_dilation=None, rhs_dilation=None):
    """
    Filter the array provided by filter provided.
    """
    #get weighted sum of neighbors using convolution
    cells = jax.lax.conv_general_dilated(cells[jnp.newaxis,...,jnp.newaxis], # lhs = NHWC image tensor
                                        nethood,  # rhs = OHWI conv kernel tensor
                                        window_strides,  # window strides
                                        'SAME', # padding mode
                                        lhs_dilation,  # lhs/image dilation
                                        rhs_dilation,  # rhs/kernel dilation
                                        dn)     # dimension_numbers = lhs, rhs, out dimension permutation
    return cells[0,...,0] # NHWC

# conv operators
@partial(jax.jit, static_argnums=(2,3,4,5,6,7))
def _fast_conv(sub_cells, filters, dn, window_strides, padding_mode="SAME", lhs_dilation=None, rhs_dilation=None, feature_group_count=1):
    """
    nD conv done with filters with JAX builtin general conv func. Supports multiple filters in channel dim.
    Assumes sub_cells has matching channel dim to filters
    The dims are defined by dn and strides

    Cells is assumed to be the relevant sub-region to process
    """
    #Conv via 2D brute force
    #get weighted sum of neighbors using convolution
    sub_cells_conv = jax.lax.conv_general_dilated(sub_cells[jnp.newaxis,...], # lhs = NHWC image tensor
                                        filters,  # rhs = OHWI conv kernel tensor
                                        window_strides,  # window strides
                                        padding_mode, # padding mode
                                        lhs_dilation,  # lhs/image dilation
                                        rhs_dilation,  # rhs/kernel dilation
                                        dn,  # dimension_numbers = lhs, rhs, out dimension permutation
                                        feature_group_count, #used for grouped convolutions
                                        )
    return sub_cells_conv[0] # NHWC, remove extra batch dim, return multi-filters result
