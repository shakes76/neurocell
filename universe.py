"""
Universe object for Von Neumann networks
"""
from typing import Sequence

import jax
import jax.numpy as jnp
import jax.nn as nn
import jax.scipy.signal as jsignal
import haiku as hk
import math

import filters #local
from convs import _filter
import forward #local

#kernels for larger networks
kernel_backward =  [[1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0]]
kernel_forward =   [[0, 0, 1],
                    [0, 0, 1],
                    [0, 0, 1],
                    [0, 0, 1],
                    [0, 0, 1]]
STATE_ACTIVE = 0.7
STATE_SIGNAL = 0.2
STATE_ACTIVE_VALUE = 1.0
STATE_SIGNAL_VALUE = 0.0

class VectorCells(hk.initializers.Initializer):
    """Initializes by sampling from a normal distribution and setting vertical/horizontal cells."""
    def __init__(self, in_sizes, out_sizes, in_offsets, out_offsets, is_states=False, clear=False, random_init=1.0, col=True):
        """
        Constructs a :class:`RandomNormal` like initializer with vector cells.

        in_sizes etc. can be single or a list. in_sizes and in_offsets must be same in number, likewise out_*
        """
        self.in_sizes = in_sizes
        self.out_sizes = out_sizes
        self.in_offsets = in_offsets
        self.out_offsets = out_offsets
        self.is_states = is_states
        self.clear = clear
        self.random_init = random_init
        self.col = col
        # self.initializer = jax.nn.initializers.uniform(self.random_init)
        self.initializer = jax.nn.initializers.variance_scaling(self.random_init, mode="fan_in", distribution="truncated_normal")
        # self.initializer = jax.nn.initializers.truncated_normal(stddev=self.random_init)
        # self.initializer = jax.nn.initializers.xavier_uniform()
        # self.initializer = jax.nn.initializers.orthogonal()

    def clear_cells_vertical(self, cells, offset, after=False):
        """
        Clears cells vertically upto and including offset positions, assumes 2D
        """
        if after:
            result = cells.at[:,offset[1]:].set(0)  #clear cells after
        else:
            result = cells.at[:,:offset[1]+1].set(0) #clear cells before
        return result

    def clear_cells_horizontal(self, cells, offset, after=False):
        """
        Clears cells horizontally upto and including offset positions, assumes 2D
        """
        if after:
            result = cells.at[offset[0]:,:].set(0)  #clear cells after
        else:
            result = cells.at[:offset[0]+1,:].set(0) #clear cells before
        return result

    def __call__(self, shape: Sequence[int], dtype) -> jax.Array:
        if self.random_init > 0 or not self.random_init is None:
            cells = self.initializer(hk.next_rng_key(), shape, jnp.float32)
        else:
            cells = jnp.zeros(shape, dtype=dtype)

        if self.is_states:
            #norm states to 0-1 range
            cells = jax.nn.sigmoid(cells)
            # cells = (cells - cells.min()) / (cells.max() - cells.min())

        if not isinstance(self.in_sizes, list):
            self.in_sizes = [self.in_sizes] #assume was one element, convert to list
            self.in_offsets = [self.in_offsets] #assume was one element, convert to list
        if not isinstance(self.out_sizes, list):
            self.out_sizes = [self.out_sizes] #assume was one element, convert to list
            self.out_offsets = [self.out_offsets] #assume was one element, convert to list

        for in_size, in_offset in zip(self.in_sizes, self.in_offsets):
            if self.is_states:
                # cells = jnp.ones(shape, dtype=dtype)
                inputs = jnp.full(in_size, STATE_ACTIVE_VALUE, dtype=dtype)
            else:
                # real_dtype = jnp.finfo(dtype).dtype
                # stddev = 1. / jnp.sqrt(max(shape[0], shape[1]))
                # upper = 3.0
                # lower = -3.0
                # cells = stddev * jax.random.truncated_normal(hk.next_rng_key(), lower, upper, shape, real_dtype)
                # cells = 1.0 * jax.random.normal(hk.next_rng_key(), shape)
                inputs = jnp.ones(in_size, dtype=dtype)

            if self.clear:
                if self.col:
                    cells = self.clear_cells_vertical(cells, in_offset, after=False)
                else:
                    cells = self.clear_cells_horizontal(cells, in_offset, after=False)

            cells = jax.lax.dynamic_update_slice(cells, inputs, in_offset)

        for out_size, out_offset in zip(self.out_sizes, self.out_offsets):
            if self.is_states:
                # cells = jnp.ones(shape, dtype=dtype)
                outputs = jnp.full(out_size, STATE_ACTIVE_VALUE, dtype=dtype)
            else:
                # real_dtype = jnp.finfo(dtype).dtype
                # stddev = 1. / jnp.sqrt(max(shape[0], shape[1]))
                # upper = 3.0
                # lower = -3.0
                # cells = stddev * jax.random.truncated_normal(hk.next_rng_key(), lower, upper, shape, real_dtype)
                # cells = 1.0 * jax.random.normal(hk.next_rng_key(), shape)
                outputs = jnp.ones(out_size, dtype=dtype)

            if self.clear:
                if self.col:
                    cells = self.clear_cells_vertical(cells, out_offset, after=True)
                else:
                    cells = self.clear_cells_horizontal(cells, out_offset, after=True)

            cells = jax.lax.dynamic_update_slice(cells, outputs, out_offset)

        return cells

class VerticalCells(VectorCells):
    """Initializes by sampling from a normal distribution and setting vertical cells."""
    def __init__(self, in_sizes, out_sizes, in_offsets, out_offsets, is_states=False, clear=False, random_init=1.0):
        """
        Constructs a :class:`RandomNormal` like initializer with vertical cells. See base class VectorCells.
        """
        super().__init__(in_sizes, out_sizes, in_offsets, out_offsets, is_states, clear, random_init, col=True)

class HorizontalCells(VectorCells):
    """Initializes by sampling from a normal distribution and setting horizontal cells."""
    def __init__(self, in_sizes, out_sizes, in_offsets, out_offsets, is_states=False, clear=False, random_init=1.0):
        """
        Constructs a :class:`RandomNormal` like initializer with horizontal cells. See base class VectorCells.
        """
        super().__init__(in_sizes, out_sizes, in_offsets, out_offsets, is_states, clear, random_init, col=False)

class CoordinateCells(hk.initializers.Initializer):
    """Initializes by sampling from a normal distribution and setting cells through 2D coordinates provided."""
    def __init__(self, in_coords, out_coords, out_offset, out_size, is_states=False, clear=True, random_init=1.0):
        """
        Constructs a :class:`RandomNormal` like initializer with cells with coordinates.
        """
        self.in_coords = in_coords
        self.out_coords = out_coords
        self.out_offset = out_offset #unused
        self.out_size = out_size #unused
        self.is_states = is_states
        self.clear = clear
        self.random_init = random_init
        self.min_coord = in_coords[:, 1].min()
        self.max_coord = out_coords[:, 1].max()
        self.random_init = random_init
        # self.initializer = jax.nn.initializers.uniform(self.random_init)
        self.initializer = jax.nn.initializers.variance_scaling(self.random_init, mode="fan_in", distribution="truncated_normal")

    def clear_cells_outside(self, cells, coord, after=False):
        """
        Clears cells upto and including coords positions padded by 1
        """
        if after:
            result = cells.at[:,coord+1:].set(0)  #clear cells after
        else:
            result = cells.at[:,:coord].set(0) #clear cells before
        return result

    def __call__(self, shape: Sequence[int], dtype) -> jax.Array:
        n = len(self.in_coords)
        if self.is_states:
            # cells = jnp.ones(shape, dtype=dtype)
            inputs = STATE_ACTIVE_VALUE
            outputs = STATE_ACTIVE_VALUE
        else:
            # real_dtype = jnp.finfo(dtype).dtype
            # stddev = 1. / jnp.sqrt(max(shape[0], shape[1]))
            # upper = 3.0
            # lower = -3.0
            # cells = stddev * jax.random.truncated_normal(hk.next_rng_key(), lower, upper, shape, real_dtype)
            # cells = 1.0 * jax.random.normal(hk.next_rng_key(), shape)
            inputs = STATE_SIGNAL_VALUE
            outputs = STATE_SIGNAL_VALUE

        if self.random_init > 0 or not self.random_init is None:
            cells = self.initializer(hk.next_rng_key(), shape, jnp.float32)
        else:
            cells = jnp.zeros(shape, dtype=dtype)

        if self.is_states:
            #norm states to 0-1 range
            cells = jax.nn.sigmoid(cells)
            # cells = (cells - cells.min()) / (cells.max() - cells.min())

        if self.clear:
            cells = self.clear_cells_outside(cells, self.min_coord, after=False)
            cells = self.clear_cells_outside(cells, self.max_coord, after=True)

        cells = cells.at[self.in_coords[:, 0], self.in_coords[:, 1]].set(inputs)
        cells = cells.at[self.out_coords[:, 0], self.out_coords[:, 1]].set(outputs)

        return cells

class FiltersCells(hk.initializers.Initializer):
    """Initializes by sampling from a normal distribution and setting filters."""
    def __init__(self, is_forward=True, random_init=1.0):
        self.is_forward = is_forward
        self.random_init = random_init
        # self.initializer = jax.nn.initializers.uniform(self.random_init)
        self.initializer = jax.nn.initializers.variance_scaling(self.random_init, mode="fan_in", distribution="truncated_normal")

    def __call__(self, shape: Sequence[int], dtype) -> jax.Array:
        # if not self.is_forward:
        #     nethood = filters.backward(dtype=dtype, norm=False)[jnp.newaxis,:,:,jnp.newaxis] # 8 connected kernel
        # else:
        #     nethood = filters.forward(dtype=dtype, norm=False)[jnp.newaxis,:,:,jnp.newaxis] # 8 connected kernel

        # nethood = nethood + 0.1 * jax.random.normal(hk.next_rng_key(), nethood.shape)
        if self.random_init > 0 or not self.random_init is None:
            nethood = self.initializer(hk.next_rng_key(), shape, jnp.float32)
        else:
            nethood = jnp.ones(shape, dtype=dtype)

        return nethood

class UniverseLP(hk.Module):
    """
    Represents cellular structured universe of Von Neumann networks (VNNs) as a layered perceptron (LP)

    You must use the initialize_* members to create the type of layered perceptron, e.g. MLP etc. first
    Then call usual JAX/Haiku type networks calls that will call this classes implementation of the VNNs
    """
    def __init__(self, N, sharpen=True, dtype=jnp.float32, name=None):
        super().__init__(name=name)
        if isinstance(N, (list, tuple)): #size provided by user
            self.N = max(N)
            self.grid_size = N
        else:
            self.N = N
            self.grid_size = (N,N)
        self.sharpen = sharpen
        self.dtype = dtype
        self.states_dtype = jnp.float32
        # self.type_MLP = False
        # self.type_CLP = False
        #filters for spatial derivatives
        # self.nethood_backward = jnp.array(kernel_backward, dtype=dtype)[jnp.newaxis,:,:,jnp.newaxis] # 8 connected kernel, NHWC
        # self.nethood_forward = jnp.array(kernel_forward, dtype=dtype)[jnp.newaxis,:,:,jnp.newaxis] # 8 connected kernel, NHWC
        # self.nethood_backward = filters.backward(dtype=jnp.float32, norm=False)[jnp.newaxis,:,:,jnp.newaxis] # 8 connected kernel
        # self.nethood_forward = filters.forward(dtype=jnp.float32, norm=False)[jnp.newaxis,:,:,jnp.newaxis] # 8 connected kernel
        self.smooth_filter = filters.smooth(dtype=jnp.float32, norm=False)[jnp.newaxis,:,:,jnp.newaxis] # 8 connected kernel
        self.sharpen_filter = filters.sharpen(dtype=jnp.float32, norm=True)[jnp.newaxis,:,:,jnp.newaxis] # 8 connected kernel
        # print("filter:", self.nethood_forward)
        self.filters_shape = None
        self.img_shape = (1,N,N,1)
        self.window_stride = (1,1)
        self.forward_op = None

    def initialize(self, in_sizes, out_sizes, in_offsets, out_offsets, depth, kernel_size=(3,3), directions=1):
        """
        Init all cells with a series of input and outputs as vectors, i.e. lists of vectors are provided, single values are also supported.

        Assumes inputs or outputs are of same length, different number of inputs and outputs can be provided.
        """
        self.in_size = in_sizes
        self.out_size = out_sizes
        self.in_offset = in_offsets
        self.out_offset = out_offsets
        self.kernel_size = kernel_size
        self.depth = depth
        self.directions = directions
        # print("depth:", self.depth)

        cells = jnp.zeros(self.grid_size, dtype=self.dtype)
        hk.set_state("universe", cells)

        init_scale = 1./jnp.sqrt(self.depth*self.N)
        init_filter_scale = 1./jnp.sqrt(self.kernel_size[0]*self.kernel_size[1]*self.directions) #shape is (rows,cols) before array creation

        #init states
        self.s_init = VectorCells(in_sizes, out_sizes, in_offsets, out_offsets, is_states=True, clear=False, random_init=init_scale)
        # states = self.s_init(shape=self.grid_size, dtype=self.states_dtype)
        # passives = jnp.full(layer_size, STATE_ACTIVE, dtype=self.states_dtype)
        # self.states = jax.lax.dynamic_update_slice(states, passives, layer_offset)
        self.states = hk.get_parameter("s", shape=self.grid_size, dtype=self.states_dtype, init=self.s_init)
        # print("States:", self.states)
        # hk.set_state("states", self.states)
        # we need input and out shaped array to force input/output states to be active, used later
        # assumes all ins and outs are same size by taking first element, TODO: remove this restriction
        self.input_states = []
        for in_size in in_sizes:
            self.input_states.append(jnp.full(in_size, STATE_ACTIVE_VALUE, dtype=self.states_dtype))
        self.output_states = jnp.full(self.out_size, STATE_ACTIVE_VALUE, dtype=self.states_dtype)
        self.output_weights = jnp.full(self.out_size, 1., dtype=jnp.float32)

        #init weights (random)
        self.w_init = VectorCells(in_sizes, out_sizes, in_offsets, out_offsets, is_states=False, clear=False, random_init=init_scale)
        self.weights = hk.get_parameter("w", shape=self.grid_size, dtype=self.dtype, init=self.w_init)

        #init biases
        self.b_init = jnp.zeros
        self.biases = hk.get_parameter("b", shape=self.grid_size, dtype=self.dtype, init=self.b_init)

        #init filters
        self.filters_shape = (1,kernel_size[0],kernel_size[1],self.directions)
        self.f_init = FiltersCells(is_forward=False, random_init=init_filter_scale) # 8 connected kernel
        # nethood_forward = filters.forward(dtype=jnp.float32, norm=False)[jnp.newaxis,:,:,jnp.newaxis] # 8 connected kernel
        self.filters = hk.get_parameter("f", shape=self.filters_shape, dtype=self.dtype, init=self.f_init)

        self.dn = jax.lax.conv_dimension_numbers(self.img_shape,     # only ndim matters, not shape
                                self.filters.shape,  # only ndim matters, not shape
                                ('NHWC', 'IHWO', 'NHWC'))  # the important bit

        self.forward_op = jax.vmap(forward._propagate_full_mlp_list, in_axes=[0, None, None, None, None, None, None, None, None, None, None, None, None, None, None])

        return hk.get_state("universe")

    def initialize_mlp(self, in_size, out_size, in_offset, out_offset, depth, kernel_size=(3,3), directions=1):
        """
        Init all cells as a typical vertical set of layers like a traditional MLP
        """
        self.in_size = in_size
        self.out_size = out_size
        self.in_offset = in_offset
        self.out_offset = out_offset
        self.kernel_size = kernel_size
        self.depth = depth
        # self.depth = self.out_offset[1]-self.in_offset[1]
        self.directions = directions
        # print("depth:", self.depth)

        cells = jnp.zeros(self.grid_size, dtype=self.dtype)
        hk.set_state("universe", cells)

        init_scale = 1./jnp.sqrt(self.depth*self.N)
        init_filter_scale = 1./jnp.sqrt(math.prod(self.kernel_size)*self.directions) #shape is (rows,cols) before array creation

        #init states
        self.s_init = VerticalCells(in_size, out_size, in_offset, out_offset, is_states=True, clear=True, random_init=init_scale)
        # states = self.s_init(shape=self.grid_size, dtype=self.states_dtype)
        # passives = jnp.full(layer_size, STATE_ACTIVE, dtype=self.states_dtype)
        # self.states = jax.lax.dynamic_update_slice(states, passives, layer_offset)
        self.states = hk.get_parameter("s", shape=self.grid_size, dtype=self.states_dtype, init=self.s_init)
        # print("States:", self.states)
        # hk.set_state("states", self.states)
        # we need input and out shaped array to force input/output states to be active, used later
        self.input_states = jnp.full(self.in_size, STATE_ACTIVE_VALUE, dtype=self.states_dtype)
        self.output_states = jnp.full(self.out_size, STATE_ACTIVE_VALUE, dtype=self.states_dtype)
        self.output_weights = jnp.full(self.out_size, 1., dtype=jnp.float32)

        #init weights (random)
        self.w_init = VerticalCells(in_size, out_size, in_offset, out_offset, is_states=False, clear=True, random_init=init_scale)
        self.weights = hk.get_parameter("w", shape=self.grid_size, dtype=self.dtype, init=self.w_init)

        #init biases
        self.b_init = jnp.zeros
        self.biases = hk.get_parameter("b", shape=self.grid_size, dtype=self.dtype, init=self.b_init)

        #init filters
        self.filters_shape = (1,kernel_size[0],kernel_size[1],self.directions)
        self.f_init = FiltersCells(is_forward=False, random_init=init_filter_scale) # 8 connected kernel
        # nethood_forward = filters.forward(dtype=jnp.float32, norm=False)[jnp.newaxis,:,:,jnp.newaxis] # 8 connected kernel
        self.filters = hk.get_parameter("f", shape=self.filters_shape, dtype=self.dtype, init=self.f_init)

        self.dn = jax.lax.conv_dimension_numbers(self.img_shape,     # only ndim matters, not shape
                                self.filters.shape,  # only ndim matters, not shape
                                ('NHWC', 'IHWO', 'NHWC'))  # the important bit

        self.forward_op = jax.vmap(forward._propagate_mlp, in_axes=[0, None, None, None, None, None, None, None, None, None, None, None, None, None, None])

        return hk.get_state("universe")

    def initialize_mlp_row(self, in_size, out_size, in_offset, out_offset, depth, kernel_size=(3,3), directions=1):
        """
        Init all cells as a typical vertical set of layers like a traditional MLP
        """
        self.in_size = in_size
        self.out_size = out_size
        self.in_offset = in_offset
        self.out_offset = out_offset
        self.kernel_size = kernel_size
        self.depth = depth
        # self.depth = self.out_offset[1]-self.in_offset[1]
        self.directions = directions
        # print("depth:", self.depth)

        cells = jnp.zeros(self.grid_size, dtype=self.dtype)
        hk.set_state("universe", cells)

        init_scale = 1./jnp.sqrt(self.depth*self.N)
        init_filter_scale = 1./jnp.sqrt(math.prod(self.kernel_size)*self.directions) #shape is (rows,cols) before array creation

        #init states
        self.s_init = HorizontalCells(in_size, out_size, in_offset, out_offset, is_states=True, clear=True, random_init=init_scale)
        # states = self.s_init(shape=self.grid_size, dtype=self.states_dtype)
        # passives = jnp.full(layer_size, STATE_ACTIVE, dtype=self.states_dtype)
        # self.states = jax.lax.dynamic_update_slice(states, passives, layer_offset)
        self.states = hk.get_parameter("s", shape=self.grid_size, dtype=self.states_dtype, init=self.s_init)
        # print("States:", self.states)
        # hk.set_state("states", self.states)
        # we need input and out shaped array to force input/output states to be active, used later
        self.input_states = jnp.full(self.in_size, STATE_ACTIVE_VALUE, dtype=self.states_dtype)
        self.output_states = jnp.full(self.out_size, STATE_ACTIVE_VALUE, dtype=self.states_dtype)
        self.output_weights = jnp.full(self.out_size, 1., dtype=jnp.float32)

        #init weights (random)
        self.w_init = HorizontalCells(in_size, out_size, in_offset, out_offset, is_states=False, clear=True, random_init=init_scale)
        self.weights = hk.get_parameter("w", shape=self.grid_size, dtype=self.dtype, init=self.w_init)

        #init biases
        self.b_init = jnp.zeros
        self.biases = hk.get_parameter("b", shape=self.grid_size, dtype=self.dtype, init=self.b_init)

        #init filters
        self.filters_shape = (1,kernel_size[0],kernel_size[1],self.directions)
        self.f_init = FiltersCells(is_forward=False, random_init=init_filter_scale) # 8 connected kernel
        # nethood_forward = filters.forward(dtype=jnp.float32, norm=False)[jnp.newaxis,:,:,jnp.newaxis] # 8 connected kernel
        self.filters = hk.get_parameter("f", shape=self.filters_shape, dtype=self.dtype, init=self.f_init)

        self.dn = jax.lax.conv_dimension_numbers(self.img_shape,     # only ndim matters, not shape
                                self.filters.shape,  # only ndim matters, not shape
                                ('NHWC', 'IHWO', 'NHWC'))  # the important bit

        self.forward_op = jax.vmap(forward._propagate_mlp_row, in_axes=[0, None, None, None, None, None, None, None, None, None, None, None, None, None, None])

        return hk.get_state("universe")

    def initialize_clp(self, in_coords, out_coords, out_offset, out_size, layer_coords, depth, kernel_size=(3,3), directions=1):
        """
        Init all cells as an open circle-like shape
        """
        self.in_offset = in_coords
        self.out_coords = out_coords
        self.out_offset = out_offset
        self.out_size = out_size
        # self.layer_coords = layer_coords
        self.kernel_size = kernel_size
        self.depth = depth
        # print("depth:", depth)
        self.directions = directions

        cells = jnp.zeros(self.grid_size, dtype=self.dtype)
        hk.set_state("universe", cells)

        #init states (integers)
        self.s_init = CoordinateCells(self.in_offset, self.out_coords, out_offset, out_size, is_states=True, clear=False)
        # states = self.s_init(shape=self.grid_size, dtype=self.states_dtype)
        # self.states = states.at[self.layer_coords[:, 0], self.layer_coords[:, 1]].set(2)
        self.states = hk.get_parameter("s", shape=self.grid_size, dtype=self.states_dtype, init=self.s_init)
        # print("States:", self.states)
        # hk.set_state("states", self.states)
        # we need input and out shaped array to force input/output states to be active, used later
        self.input_states = None
        self.output_states = jnp.full(self.out_size, STATE_ACTIVE_VALUE, dtype=self.states_dtype)
        self.output_weights = None

        #init weights (random)
        self.w_init = CoordinateCells(self.in_offset, self.out_coords, out_offset, out_size, is_states=False, clear=False)
        self.weights = hk.get_parameter("w", shape=self.grid_size, dtype=self.dtype, init=self.w_init)

        #init biases
        self.b_init = jnp.zeros
        self.biases = hk.get_parameter("b", shape=self.grid_size, dtype=self.dtype, init=self.b_init)

        #init filters
        self.filters_shape = (1,kernel_size[0],kernel_size[1],self.directions)
        self.f_init = FiltersCells(is_forward=False) # 8 connected kernel
        # nethood_forward = filters.forward(dtype=jnp.float32, norm=False)[jnp.newaxis,:,:,jnp.newaxis] # 8 connected kernel
        self.filters = hk.get_parameter("f", shape=self.filters_shape, dtype=self.dtype, init=self.f_init)

        self.dn = jax.lax.conv_dimension_numbers(self.img_shape,     # only ndim matters, not shape
                                self.filters.shape,  # only ndim matters, not shape
                                ('NHWC', 'IHWO', 'NHWC'))  # the important bit

        self.forward_op = jax.vmap(forward._propagate_full_clp, in_axes=[0, None, None, None, None, None, None, None, None, None, None, None, None, None, None])

        return hk.get_state("universe")

    def get_weights(self):
        return self.weights

    def get_biases(self):
        return self.biases

    def get_states(self):
        return self.states

    def get_filters(self):
        return self.filters

    def get_state(self):
        return hk.get_state("universe")

    # def forward_propagate(self, activate=nn.sigmoid, norm=True):
    #     self.universe = self.forward(self.universe, self.weights, self.biases, activate=activate, norm=norm)

    def __call__(self, inputs, activate=nn.tanh, final_activate=nn.sigmoid, norm=True):
        # get params
        # states = self.states
        states = hk.get_parameter("s", shape=self.grid_size, dtype=self.states_dtype, init=self.s_init)
        weights = hk.get_parameter("w", shape=self.grid_size, dtype=self.dtype, init=self.w_init)
        biases = hk.get_parameter("b", shape=self.grid_size, dtype=self.dtype, init=self.b_init)

        #reset cells
        # cells = jnp.zeros(self.grid_size, dtype=self.dtype)
        cells = hk.get_state("universe")
        cells = cells.at[...].set(0.)  #clear cells
        # hk.set_state("universe", cells)

        #smooth/sharpen weights
        if self.sharpen:
            weights = _filter(weights, self.sharpen_filter, self.dn, self.window_stride)

        #precompute FFTs of filters
        filters = hk.get_parameter("f", shape=self.filters_shape, dtype=self.dtype, init=self.f_init)
        # filters = filters[0,:,:,0]
        # length = self.kernel_size[0]
        # # FFTs of filter
        # filter_ffts = _fft_1D(self.N, length, filters)

        #set input
        # if self.type_MLP:
        #     cells = jax.lax.dynamic_update_slice(cells, inputs, self.in_offset)
        # else:
        #     cells = cells.at[self.in_coords[:, 0], self.in_coords[:, 1]].set(inputs)
        # hk.set_state("universe", cells)

        # force input/output states to be active
        if not self.input_states is None:
            if isinstance(self.input_states, list):
                for input_state, in_offset in zip(self.input_states, self.in_offset):
                    states = jax.lax.dynamic_update_slice(states, input_state, in_offset)
            else:
                states = jax.lax.dynamic_update_slice(states, self.input_states, self.in_offset)
        if not self.output_states is None:
            states = jax.lax.dynamic_update_slice(states, self.output_states, self.out_offset)
        if not self.output_weights is None:
            weights = jax.lax.dynamic_update_slice(weights, self.output_weights, self.out_offset)

        #forward
        # cells = forward._propagate_fast(self.in_offset[1], self.depth, cells, weights, biases, states, filters, activate, final_activate, norm, self.dn)
        # cells = forward._propagate_fast_mlp(inputs, self.in_offset, self.depth, cells, weights, biases, states, filters, activate, final_activate, norm, self.dn)
        # cells = forward._propagate_fast_1D(self.in_offset[1], self.depth, cells, weights, biases, states, filter_ffts, activate, final_activate, norm)
        # forward_propagate_fast = jax.vmap(forward._propagate_fast_mlp, in_axes=[0, None, None, None, None, None, None, None, None, None, None, None, None, None])
        cells, outputs = self.forward_op(inputs, self.in_offset, self.out_offset, self.out_size, self.depth, cells, weights, biases, states, filters, activate, final_activate, norm, self.dn, self.window_stride)
        hk.set_state("universe", cells) #remove vmap batch dim

        #read output
        # if self.type_MLP:
        #     return jax.lax.dynamic_slice(cells, self.out_offset, self.out_size)
        # else:
        #     # value = cells[self.out_coords[:, 0], self.out_coords[:, 1]]
        #     # print("out:", value)
        #     # return value
        #     return cells[self.out_coords[:, 0], self.out_coords[:, 1]]
        return outputs

    def save(self, model_name):
        jnp.savez(model_name, weights=self.weights, biases=self.biases, states=self.states, allow_pickle=False)

    def load(self, filename):
        with jnp.load(filename, allow_pickle=False) as data:
            self.weights = data['weights']
            self.biases = data['biases']
            self.states = data['states']
            self.filters = data['filters']
        self.N = self.weights.shape[0]
        self.grid_size = self.weights.shape
        self.dtype = self.weights.dtype
        self.universe = jnp.zeros(self.grid_size, dtype=self.dtype)
