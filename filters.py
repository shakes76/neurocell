# -*- coding: utf-8 -*-
"""
Discrete filters module

Note that convolution is the time reversed correlation with the kernel,
hence convolution with forward is equivalent to correlation with backward

Notes:
Integer values are used for the kernels and they are not normalised in general

See: https://en.wikipedia.org/wiki/Kernel_(image_processing)
"""
import jax.numpy as jnp

def smooth(dtype=jnp.int32, norm=False):
    '''
    Returns the window or smoothing kernel
    '''
    kernel =   [[1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]]
    if norm:
        kernel[1][1] = 8
    kernel = jnp.array(kernel, dtype=dtype)
    # print(kernel.shape)
    return kernel

def backward(dtype=jnp.int32, norm=False):
    '''
    Returns the backward propagation kernel
    '''
    kernel =   [[1, 0, 0],
                [1, 0, 0],
                [1, 0, 0]]
    if norm:
        kernel[1][1] = 3
    kernel = jnp.array(kernel, dtype=dtype)
    # print(kernel.shape)
    return kernel

def forward(dtype=jnp.int32, norm=False):
    '''
    Returns the forward propagation kernel
    '''
    kernel =   [[0, 0, 1],
                [0, 0, 1],
                [0, 0, 1]]
    if norm:
        kernel[1][1] = 3
    kernel = jnp.array(kernel, dtype=dtype)
    # print(kernel.shape, kernel)
    return kernel

def forwardPropagation(dtype=jnp.int32, norm=False):
    '''
    Returns the forward derivative kernel
    '''
    kernel =   [[0, 0, -1],
                [0, 1, -1],
                [0, 0, -1]]
    if norm:
        kernel[1][1] = 3
    kernel = jnp.array(kernel, dtype=dtype)
    # print(kernel.shape)
    return kernel

def backwardPropagation(dtype=jnp.int32, norm=False):
    '''
    Returns the backward derivative kernel
    '''
    kernel =   [[-1, 0, 0],
                [-1, 1, 0],
                [-1, 0, 0]]
    if norm:
        kernel[1][1] = 3
    kernel = jnp.array(kernel, dtype=dtype)
    # print(kernel.shape)
    return kernel

def vonNeumann(dtype=jnp.int32, norm=False):
    '''
    Returns the backward and forward propagation kernel
    '''
    kernel =   [[1, 0, 1],
                [1, 1, 1],
                [1, 0, 1]]
    if norm:
        kernel[1][1] = 6
    kernel = jnp.array(kernel, dtype=dtype)
    # print(kernel.shape)
    return kernel

def gaussian(dtype=jnp.int32):
    '''
    Returns the Gaussian smoothing kernel
    '''
    kernel =   [[1, 2, 1],
                [2, 4, 2],
                [1, 2, 1]]
    kernel = jnp.array(kernel, dtype=dtype)
    # print(kernel.shape)
    return kernel

def sobelVertical(dtype=jnp.int32):
    '''
    Returns the Sobel vertical kernel, usually used for edge detection
    '''
    kernel =   [[1, 0, -1],
                [2, 0, -2],
                [1, 0, -1]]
    kernel = jnp.array(kernel, dtype=dtype)
    # print(kernel.shape)
    return kernel

def sobelHorizontal(dtype=jnp.int32):
    '''
    Returns the Sobel horizontal kernel, usually used for edge detection
    '''
    kernel =   [[1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]]
    kernel = jnp.array(kernel, dtype=dtype)
    # print(kernel.shape)
    return kernel

def edge1(dtype=jnp.int32):
    '''
    Returns a diagonal edge kernel, usually used for edge detection
    [[1, 0, -1],
    [0, 0, 0],
    [-1, 0, 1]]
    '''
    kernel =   [[1, 0, -1],
                [0, 0, 0],
                [-1, 0, 1]]
    kernel = jnp.array(kernel, dtype=dtype)
    # print(kernel.shape)
    return kernel

def edge2(dtype=jnp.int32):
    '''
    Returns a center diagonal edge kernel, usually used for edge detection
    [[1, 0, -1],
    [0, -4, 0],
    [-1, 0, 1]]
    '''
    kernel =   [[1, 0, 1],
                [0, -4, 0],
                [1, 0, 1]]
    kernel = jnp.array(kernel, dtype=dtype)
    # print(kernel.shape)
    return kernel

def edge3(dtype=jnp.int32):
    '''
    Returns an edge kernel, usually used for edge detection
    [[-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]]
    '''
    kernel =   [[-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1]]
    kernel = jnp.array(kernel, dtype=dtype)
    # print(kernel.shape)
    return kernel

def haar(dtype=jnp.int32):
    '''
    Returns the Haar kernel, usually used for edge detection
    [[1, 1],
    [1, -1]]
    '''
    kernel =   [[1, 1],
                [1, -1]]
    kernel = jnp.array(kernel, dtype=dtype)
    # print(kernel.shape)
    return kernel

def sharpen(dtype=jnp.int32, norm=False):
    '''
    Returns the sharpen kernel, usually used for edge enhancement
    [[0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]]
    '''
    kernel =   [[0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]]
    kernel = jnp.array(kernel, dtype=dtype)
    if norm:
        kernel = kernel / 5.
    # print(kernel.shape)
    return kernel

def sharpen_3D(dtype=jnp.int32, norm=False):
    '''
    Returns the 3D sharpen kernel, usually used for edge enhancement
    [[0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]]
    '''
    kernel =   [[[0, 0, 0],
                [0, -1, 0],
                [0, 0, 0]],
                [[0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]],
                [[0, 0, 0],
                [0, -1, 0],
                [0, 0, 0]]]
    kernel = jnp.array(kernel, dtype=dtype)
    if norm:
        kernel = kernel / 5.
    # print("sharpen shape:", kernel.shape)
    return kernel

def filter(N, filterType='smooth', x0 = 0, y0 = 0):
    '''
    Returns an NxN image of the kernel requested by filterType translated by (x0,y0)
    filterType
    - smooth or window (default)
    - gauss
    - sobelV or sobelH
    - edge1
    - edge2
    - edge3
    - haar
    - sharpen
    '''
    #create kernel
    kernelImage = jnp.zeros((N,N), dtype=jnp.int32)

    if filterType == 'sobelV':
        kernel = sobelVertical()
    elif filterType == 'sobelH':
        kernel = sobelHorizontal()
    elif filterType == 'edge1':
        kernel = edge1()
    elif filterType == 'edge2':
        kernel = edge2()
    elif filterType == 'edge3':
        kernel = edge3()
    elif filterType == 'haar':
        kernel = haar()
    elif filterType == 'sharpen':
        kernel = sharpen()
    elif filterType == 'gauss':
        kernel = gaussian()
    else:
        kernel = smooth()

    kernelImage[x0:x0+kernel.shape[0], y0:y0+kernel.shape[1]] = kernel
    # print(kernelImage)
    return kernelImage
