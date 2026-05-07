'''
Forward and propagate operators for VNNs and cellular machines

Propagate functions act as learnable Green's functions for cells

@author Shakes
'''
import jax
import jax.numpy as jnp

#JIT compiled members for performance
from functools import partial

from convs import _fast_conv, _filter

#propagate funcs
@partial(jax.jit, static_argnums=(2,3,4,10,11,12,13,14))
def _propagate_full_clp(input, coords, out_offset, out_size, depth, cells, weights, biases, states, filter_ffts, activate, final_activate, norm, dn, window_strides=(1,1)):
    """
    Propagate input across cellular net based on weights etc. for all layers across entire universe
    """
    width = 0 #unused
    # length = input.shape[0]

    cells = cells.at[coords[:, 0], coords[:, 1]].set(input)

    #propagate
    for i in range(1, depth):
        cells = _forward_full(width, cells, weights, biases, states, filter_ffts, activate, norm, dn, window_strides) #not inplace
    cells = _forward_full(width, cells, weights, biases, states, filter_ffts, final_activate, norm, dn, window_strides) #final prop and activate

    return cells, jax.lax.dynamic_slice(cells, out_offset, out_size)

@partial(jax.jit, static_argnums=(1,2,3,4,10,11,12,13,14))
def _propagate_mlp(input, offset, out_offset, out_size, depth, cells, weights, biases, states, filters, activate, final_activate, norm, dn, window_strides=(1,1)):
    """
    Propagate input across cellular net based on weights etc. as a traditional MLP

    Uses column/last hyperplane based computation for efficiency and propagates up to provided depth and
    then one more for output layer.
    Last dim of filters is used as the initial starting step
    """
    zeros_tuple = (0,) * (len(offset)-1) #offset for entire hyper plane
    step = 1 #(filters.shape[-2]-1)//2 #-2 to take last dim butignore channel dim
    # print("forward step:", step)

    cells = jax.lax.dynamic_update_slice(cells, input, offset) #place input into cells

    #propagate
    index = offset[-1]+step #first col (or last hyper plane) next to input
    # print("forward index",index)
    for i in range(depth): #0 to depth-1, depth in total
        result_block = _forward_column(index+i, cells, weights, biases, states, filters, activate, norm, dn, window_strides)
        cells = jax.lax.dynamic_update_slice(cells, result_block, zeros_tuple+(index+i,)) #place back into col (or hyper plane)
    # print("forward index final",index+depth)
    result_block = _forward_column(index+depth, cells, weights, biases, states, filters, final_activate, norm, dn, window_strides)
    if not final_activate == None:
        result_block = final_activate(result_block) #force final activation, assume output layer
    cells = jax.lax.dynamic_update_slice(cells, result_block, zeros_tuple+(index+depth,)) #place back into final col (or hyper plane)

    return cells, jax.lax.dynamic_slice(cells, out_offset, out_size)

@partial(jax.jit, static_argnums=(1,2,3,4,10,11,12,13,14))
def _propagate_mlp_row(input, offset, out_offset, out_size, depth, cells, weights, biases, states, filters, activate, final_activate, norm, dn, window_strides=(1,1)):
    """
    Propagate input across cellular net based on weights etc. as a traditional MLP

    Uses row/second-last hyperplane based computation for efficiency and propagates up to provided depth and
    then one more for output layer.
    """
    zeros_tuple = (0,) * (len(offset)-2) #offset for entire hyper plane upto second last (row)

    cells = jax.lax.dynamic_update_slice(cells, input, offset) #place input into cells

    #propagate
    index = offset[-2]+1 #first row (or second-last hyper plane) next to input
    # print("forward index",index)
    for i in range(depth): #0 to depth-1, depth in total
        result_block = _forward_row(index+i, cells, weights, biases, states, filters, activate, norm, dn, window_strides)
        cells = jax.lax.dynamic_update_slice(cells, result_block, zeros_tuple+(index+i,)+(0,)) #place back into row (or hyper plane)
    # print("forward index final",index+depth)
    result_block = _forward_row(index+depth, cells, weights, biases, states, filters, final_activate, norm, dn, window_strides)
    if not final_activate == None:
        result_block = final_activate(result_block) #force final activation, assume output layer
    cells = jax.lax.dynamic_update_slice(cells, result_block, zeros_tuple+(index+depth,)+(0,)) #place back into final row (or hyper plane)

    return cells, jax.lax.dynamic_slice(cells, out_offset, out_size)

@partial(jax.jit, static_argnums=(1,2,3,4,10,11,12,13,14))
def _propagate_full_mlp(input, offset, out_offset, out_size, depth, cells, weights, biases, states, filters, activate, final_activate, norm, dn, window_strides=(1,1)):
    """
    Propagate input across cellular net based on weights etc. as a traditional MLP
    """
    cells = jax.lax.dynamic_update_slice(cells, input, offset) #place input into cells

    #propagate
    for i in range(1, depth):
        cells = _forward_full(1, cells, weights, biases, states, filters, activate, norm, dn, window_strides) #not inplace
    cells = _forward_full(1, cells, weights, biases, states, filters, final_activate, norm, dn, window_strides) #final prop and activate
    if not final_activate == None:
        cells = final_activate(cells) #force final activation, assume output layer

    return cells, jax.lax.dynamic_slice(cells, out_offset, out_size)

@partial(jax.jit, static_argnums=(2,3,4,10,11,12,13,14))
def _propagate_full_mlp_list(inputs, offsets, out_offset, out_size, depth, cells, weights, biases, states, filters, activate, final_activate, norm, dn, window_strides=(1,1)):
    """
    Propagate input across cellular net based on weights etc. as a traditional MLP, inputs are list of inputs, one output still
    """
    for input, offset in zip(inputs, offsets): #place inputs to cells
        cells = jax.lax.dynamic_update_slice(cells, input, offset)

    #propagate
    for i in range(1, depth):
        cells = _forward_full(1, cells, weights, biases, states, filters, activate, norm, dn, window_strides) #not inplace
    cells = _forward_full(1, cells, weights, biases, states, filters, final_activate, norm, dn, window_strides) #final prop and activate
    if not final_activate == None:
        cells = final_activate(cells) #force final activation, assume output layer

    return cells, jax.lax.dynamic_slice(cells, out_offset, out_size)

# forward step operators
@partial(jax.jit, static_argnums=(5,6,7))
def _forward(cells, weights, biases, states, nethood, activate, norm, dn):
    """
    Compute the forward pass by computing each neuron depending on its state.
    """
    #get weighted sum of neighbors using convolution
    cells = jax.lax.conv_general_dilated(cells[jnp.newaxis,:,:,jnp.newaxis], # lhs = NHWC image tensor
                                        nethood,  # rhs = OHWI conv kernel tensor
                                        (1,1),  # window strides
                                        'SAME', # padding mode
                                        (1,1),  # lhs/image dilation
                                        (1,1),  # rhs/kernel dilation
                                        dn)     # dimension_numbers = lhs, rhs, out dimension permutation
    cells = cells[0,:,:,0] # NHWC

    if norm:
        mean_axes = tuple(i for i in range(len(dn.lhs_spec)-2)) #nD cells need norm'ing, ignore B and C
        # print("mean axes", mean_axes)
        #norm, note no B in result due to vectorization
        cells = jax.nn.standardize(cells, axis=mean_axes)

    #norm states to 0-1 range
    states = jax.nn.sigmoid(states)
    # sub_states = (sub_states - sub_states.min()) / (sub_states.max() - sub_states.min())

    #check states
    #if passive apply activation, else signal state so weight
    cells = jnp.where(jnp.logical_and(states>=STATE_SIGNAL,states<STATE_ACTIVE), cells*weights, cells) #weight if signal
    if activate == None:
        cells = jnp.where(states>=STATE_ACTIVE, cells+biases, cells) #activate if passive, more accurate
    else:
        cells = jnp.where(states>=STATE_ACTIVE, activate(cells+biases), cells) #activate if passive, more accurate
    # cells = jnp.where(states==2, activate(cells+biases), cells) #activate if passive, faster

    return cells

@partial(jax.jit, static_argnums=(0,6,7,8,9,10))
def _forward_column(index, cells, weights, biases, states, filters, activate, norm, dn, window_strides=(1,1), mode="COMPRESS"):
    """
    Compute the forward pass by computing each neuron depending on its state for a subregion.

    This version computes the forward WRT a central row/column (provided by index) with the nethood applied to it
    according to the width of the nethood. For a nethood of width 3, that means central row/column
    plus one before and one after it.

    All operations are done on the sub-region defined by width of nethood and central index

    Supports row/column in 2D and second-last/last hyperplane in nD

    Assumes filters are odd sized @todo generalise
    """
    filter_shape = filters.shape[1:-1] #shape is (1,rows,cols,...,1), remove first and last dims
    width = filter_shape[-1] # last dim, column in 2D
    height_pad_shape = tuple(map(lambda x: (x-1)//2, filter_shape[:-1])) # half of other dims of filter
    height_pad_shape = height_pad_shape + (0,) + (0,) #padding for each dim plus nth dim and add filter dim
    height_pad = tuple([(pad_value, pad_value) for pad_value in height_pad_shape]) #pad before and after
    width_pad = max(1, (width-1)//2) # half of last dim of filter
    # print("index",index,"width_pad",width_pad, "height_pad_shape", height_pad_shape,"height_pad",height_pad)

    #selects hyper plane (n-1)D at last dim
    sub_weights = weights[...,index-width_pad:index-width_pad+width]
    sub_biases = biases[...,index-width_pad:index-width_pad+width]
    sub_states = states[...,index-width_pad:index-width_pad+width]
    # print("sub_states",sub_states.shape)

    #process only width of kernel and restore to nD if needed
    groups_count = 1
    # groups_count = filters.shape[-1]//cells.shape[-1]
    # print("groups_count", groups_count, "cells shape", cells.shape)
    if len(cells.shape) == len(filter_shape): #if 3D
        #extract sub array of filter width to conv
        # print("cells",cells.shape)
        result_block = cells[...,index-width_pad:index-width_pad+width]
        result_block = result_block[...,jnp.newaxis] # add dim to support filter channel
        #~ groups_count = 1
    else:
        #extract sub array of filter width to conv, and preserve the channel dim
        result_block = cells[...,index-width_pad:index-width_pad+width,:]
        #~ groups_count = filters.shape[-1]

    sub_weights = sub_weights[...,jnp.newaxis] #promotes to column vec (2D) or back to nD and support filter channel
    sub_biases = sub_biases[...,jnp.newaxis] #promotes to column vec (2D) or back to nD and support filter channel
    sub_states = sub_states[...,jnp.newaxis] #promotes to column vec (2D) or back to nD and support filter channel

    # print("result_block",result_block.shape, "filters", filters.shape)
    #valid mode approach in all dims, reduced complexity by deliberately squishing last hyperplane as well to current index
    #handles boundaries better?
    result_block = _fast_conv(result_block, filters, dn, window_strides, padding_mode="SAME", feature_group_count=groups_count)
    #Fast Fourier conv
    # result_block = _fast_conv_3D(result_block, filters, pad_volume=False, normalize=True, apply_antialias=False)
    # print("result_block conv",result_block.shape)
    # result_block = jnp.pad(result_block, height_pad) #pad rows to undo valid conv in rows
    #Same mode approach
    # result_block = _fast_conv(result_block, filters, dn, padding_mode="SAME")
    # result_block = result_block[:,(width_pad+1)-1].reshape(-1,1) #reshape promotes to column vec, -1 for 0 indexing
    # print("result_block padded",result_block.shape)

    if norm: # or isinstance(norm, list):
        mean_axes = tuple(i for i in range(len(dn.lhs_spec)-2)) #nD cells need norm'ing, ignore B and C
        # print("mean axes", mean_axes)
        #norm, note no B in result due to vectorization
        result_block = jax.nn.standardize(result_block, axis=mean_axes)
        # result_block = utils.iqr_median_normalize(result_block, axis=mean_axes)
        # mean_axes = tuple(i for i in range(len(dn.lhs_spec)-3)) #nD cells need norm'ing, ignore B and C
        # gamma for scaled weights standardization equivalent to batch norm
        # Calculate scaling factor for ReLU
        # gamma = utils.get_relu_gain_gamma([sub_weights.shape[-3]])
        # sub_weights = jax.nn.standardize(sub_weights, axis=-3)
        # if isinstance(norm, list): #assume layer norm wanted
        #     scale = norm[0][index] #usually learnt per layer
        #     shift = norm[1][index] #usually learnt per layer
        #     # print("layer norm:", scale, shift)
        #     result_block = result_block * scale + shift #scale and shift distribution

    #~ result_block = jnp.pad(result_block, height_pad) #pad rows to undo valid conv in rows
    #we do this after norm to ensure padding does not skew the norm after valid approach

    #norm states to 0-1 range
    sub_states = jax.nn.sigmoid(sub_states)
    #~ sub_states = jax.nn.hard_sigmoid(sub_states)
    #~ sub_states = jax.nn.leaky_relu(sub_states)
    #~ sub_states = utils.min_max_normalize(sub_states)

    #check states
    #if passive apply activation, else signal state so weight, let the grad descent decide
    #the operation is akin to adversarial states
    # print("result_block",result_block.shape, "sub_weights", sub_weights.shape)
    if activate == None:
        result_block = result_block*sub_weights*(1.0-sub_states) + (result_block+sub_biases)*sub_states
    else:
        result_block = result_block*sub_weights*(1.0-sub_states) + activate(result_block+sub_biases)*sub_states

    if mode.lower() == "compress":
        return jnp.mean(result_block, axis=-1) # averaged across filters, back to 1 channel dim
    elif mode.lower() == "expand":
        return result_block
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'COMPRESS' or 'EXPAND'.")

@partial(jax.jit, static_argnums=(0,6,7,8,9))
def _forward_row(index, cells, weights, biases, states, filters, activate, norm, dn, window_strides=(1,1)):
    """
    Compute the forward pass by computing each neuron depending on its state for a subregion.

    This version computes the forward WRT a central row/column (provided by index) with the nethood applied to it
    according to the width of the nethood. For a nethood of width 3, that means central row/column
    plus one before and one after it.

    All operations are done on the sub-region defined by width of nethood and central index

    Supports row/column in 2D and second-last/last hyperplane in nD

    Assumes filters are odd sized @todo generalise
    """
    filter_shape = filters.shape[1:-1] #shape is (1,rows,cols,...,1), remove first and last dims
    width = filter_shape[-2] # second-last dim, row in 2D
    height_pad_shape = tuple(map(lambda x: (x-1)//2, filter_shape[:-2])) # half of other dims of filter, no padding in second last dim
    height_pad_shape = height_pad_shape + (0,) + ((filter_shape[-1]-1)//2,) + (0,) #padding for each dim and add filter dim
    height_pad = tuple([(pad_value, pad_value) for pad_value in height_pad_shape]) #pad before and after
    width_pad = (width-1)//2 # half of last dim of filter
    # print("index",index,"width_pad",width_pad, "height_pad_shape", height_pad_shape,"height_pad",height_pad)

    #selects hyper plane (n-1)D at second-last dim
    # sub_cells = cells[:,index-1:index-1+width]
    sub_weights = weights[...,index,:]
    sub_biases = biases[...,index,:]
    sub_states = states[...,index,:]
    # print("sub_states",sub_states.shape)

    #extract sub array of filter width to conv
    # print("cells",cells.shape)
    result_block = cells[...,index-width_pad:index-width_pad+width,:]
    # print("result_block",result_block.shape)

    #restore to nD
    result_block = result_block[...,jnp.newaxis] # add dim to support filter channel
    sub_weights = sub_weights[...,jnp.newaxis,:,jnp.newaxis] #promotes to row vec (2D) or back to nD and support filter channel
    sub_biases = sub_biases[...,jnp.newaxis,:,jnp.newaxis] #promotes to row vec (2D) or back to nD and support filter channel
    sub_states = sub_states[...,jnp.newaxis,:,jnp.newaxis] #promotes to row vec (2D) or back to nD and support filter channel

    # if norm:
    #     # gamma for scaled weights standardization equivalent to batch norm
    #     # Calculate scaling factor for ReLU
    #     # gamma = utils.get_relu_gain_gamma([filters.shape[-2]])

    #     # Apply the standardization function to the weights
    #     mean_axes = tuple(i for i in range(len(filters.shape)-1)) #nD cells need norm'ing, ignore B and C
    #     # print("mean axes", mean_axes, "filters", filters.shape)
    #     filters = jax.nn.standardize(filters, axis=mean_axes)

    # print("result_block",result_block.shape, "filters", filters.shape)
    #valid mode approach, reduced complexity, handles boundaries better?
    result_block = _fast_conv(result_block, filters, dn, window_strides, padding_mode="VALID")
    #Fast Fourier conv
    # result_block = _fast_conv_3D(result_block, filters, pad_volume=False, normalize=True, apply_antialias=False)
    # result_block = jnp.pad(result_block, height_pad) #pad rows to undo valid conv in rows
    #Same mode approach
    # result_block = _fast_conv(result_block, filters, dn, padding_mode="SAME")
    # result_block = result_block[:,(width_pad+1)-1].reshape(-1,1) #reshape promotes to column vec, -1 for 0 indexing
    # print("result_block padded",result_block.shape)

    if norm or isinstance(norm, list):
        mean_axes = tuple(i for i in range(len(dn.lhs_spec)-2)) #nD cells need norm'ing, ignore B and C
        # print("mean axes", mean_axes)
        #norm, note no B in result due to vectorization
        result_block = jax.nn.standardize(result_block, axis=mean_axes)
        # result_block = utils.iqr_median_normalize(result_block, axis=mean_axes)
        # mean_axes = tuple(i for i in range(len(dn.lhs_spec)-3)) #nD cells need norm'ing, ignore B and C
        # gamma for scaled weights standardization equivalent to batch norm
        # Calculate scaling factor for ReLU
        # gamma = utils.get_relu_gain_gamma([sub_weights.shape[-3]])
        # sub_weights = jax.nn.standardize(sub_weights, axis=-3)
        if isinstance(norm, list): #assume layer norm wanted
            scale = norm[0][index] #usually learnt per layer
            shift = norm[1][index] #usually learnt per layer
            # print("layer norm:", scale, shift)
            result_block = result_block * scale + shift #scale and shift distribution

    result_block = jnp.pad(result_block, height_pad) #pad rows to undo valid conv in rows
    #we do this after norm to ensure padding does not skew the norm after valid approach

    #norm states to 0-1 range
    sub_states = jax.nn.sigmoid(sub_states)
    # sub_states = (sub_states - sub_states.min()) / (sub_states.max() - sub_states.min())

    #check states
    #if passive apply activation, else signal state so weight, let the grad descent decide
    #the operation is akin to adversarial states
    # print("result_block",result_block.shape, "sub_weights", sub_weights.shape)
    if activate == None:
        result_block = result_block*sub_weights*(1.0-sub_states) + (result_block+sub_biases)*sub_states
    else:
        result_block = result_block*sub_weights*(1.0-sub_states) + activate(result_block+sub_biases)*sub_states

    return jnp.mean(result_block, axis=-1) # averaged across filters, back to 1 channel dim

@partial(jax.jit, static_argnums=(0,6,7,8,9,10))
def _forward_full(width, cells, weights, biases, states, filters, activate, norm, dn, window_strides=(1,1), mode="COMPRESS"):
    """
    Compute the forward pass by computing each neuron depending on its state for a subregion of width.
    Assumes subregion width is centred within universe WRT columns and width is even.

    This version computes the forward across the full array with the nethood applied to it
    The result is a full 2D convolution with the nethood, which can be expensive.
    Multiple nethoods or filters are supported, returns the mean result of these filters

    All operations are done on the entire universe.

    Assumes filters are odd sized @todo generalise
    """
    # centre = cells.shape[1]//2
    # padding = filters.shape[1]-1
    filter_shape = filters.shape[1:-1] #shape is (1,rows,cols,...,1), remove first and last dims
    height_pad_shape = tuple(map(lambda x: (x-1)//2, filter_shape)) # half of all dims of filter
    height_pad_shape = height_pad_shape + (0,) #padding for each dim plus filter dim
    height_pad = tuple([(pad_value, pad_value) for pad_value in height_pad_shape]) #pad before and after
    # print("height_pad_shape", height_pad_shape,"height_pad",height_pad)

    # sub_weights = weights[:,centre-width//2-padding:centre+width//2+padding]
    # sub_biases = biases[:,centre-width//2-padding:centre+width//2+padding]
    # sub_states = states[:,centre-width//2-padding:centre+width//2+padding]

    #broadcasting used later for mul, can be memory intensive for large cell arrays, see compact version
    groups_count = 1
    # groups_count = filters.shape[-1]
    if len(cells.shape) == len(filter_shape): #if 3D
        cells = cells[...,jnp.newaxis] #add features/channels dim
        # groups_count = 1
    weights = weights[...,jnp.newaxis]
    biases = biases[...,jnp.newaxis]
    states = states[...,jnp.newaxis]

    #conv to eval neighbourhood
    # padding VALID mode approach, reduced complexity, handles boundaries better?
    result = _fast_conv(cells, filters, dn, window_strides, padding_mode="VALID", feature_group_count=groups_count)
    # result = jnp.pad(result, height_pad) #pad rows to undo valid conv in rows and cols
    # result = _fast_conv(cells[:,centre-width//2-padding:centre+width//2+padding], filters, dn)
    # padding SAME mode approach
    # result = _fast_conv(cells, filters, dn, window_strides) #produces result channel dims matching filters channel dims
    # print("Conv Result Shape:",result.shape)

    if norm:
        mean_axes = tuple(i for i in range(len(dn.lhs_spec)-2)) #nD cells need norm'ing, ignore B and C
        # print("mean axes", mean_axes, "result", result)
        #norm, note no B in result due to vectorization
        result = jax.nn.standardize(result, axis=mean_axes)
        # filters = utils.normalize_l2(filters, axis=mean_axes) # unit norm

    result = jnp.pad(result, height_pad) #pad rows to undo valid conv in rows and cols
    #we do this after norm to ensure padding does not skew the norm after valid approach

    #norm states to 0-1 range
    states = jax.nn.sigmoid(states)
    #~ states = jax.nn.hard_sigmoid(states)
    #~ states = jax.nn.gelu(states)
    #~ states = utils.min_max_normalize(states)

    #check states
    #if passive apply activation, else signal state so weight, let the grad descent decide
    #the operation is akin to adversarial states
    if activate == None:
        result = result*weights*(1.0-states) + (result+biases)*states
    else:
        result = result*weights*(1.0-states) + activate(result+biases)*states

    if mode.lower() == "compress":
        return jnp.mean(result, axis=-1) # averaged across filters, back to 1 channel dim
    elif mode.lower() == "expand":
        return result
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'COMPRESS' or 'EXPAND'.")
