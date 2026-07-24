'''
Evolve a series of grid cells via rules
'''
import numpy as np
import scipy.signal as signal

import jax.numpy as jnp
import jax.scipy.signal as jsignal

import filters

neighborhood = np.ones((3,3), np.int32) # 8 connected kernel
neighborhood[1,1] = 0 #do not count centre pixel

def evolve(grid, finite=True, alive_value=1, dead_value=0):
    '''
    Given the current states of the cells, apply the GoL rules:
    - Any live cell with fewer than two live neighbors dies, as if by underpopulation.
    - Any live cell with two or three live neighbors lives on to the next generation.
    - Any live cell with more than three live neighbors dies, as if by overpopulation.
    - Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction
    '''
    if finite:
        boundary='wrap'
    else:
        boundary='fill'

    #get weighted sum of neighbors using convolution
    weights = np.around(signal.fftconvolve(grid, neighborhood, mode='same'))

    #implement the GoL rules by thresholding the weights
    for i, row in enumerate(grid):
        for j, col in enumerate(row):
            if col == alive_value:
                if int(weights[i,j]) < 2: #rule 1
                    weights[i,j] = dead_value
                elif int(weights[i,j]) == 2 or weights[i,j] == 3: #rule 2
                    weights[i,j] = alive_value
                elif int(weights[i,j]) > 3: #rule 3
                    weights[i,j] = dead_value
            else:
                if int(weights[i,j]) == 3: #rule 4
                    weights[i,j] = alive_value
                else:
                    weights[i,j] = dead_value

    #update the grid
    return weights

def evolve_fast(grid, alive_value=1, dead_value=0):
    '''
    Given the current states of the cells, apply the GoL rules:
    - Any live cell with fewer than two live neighbors dies, as if by underpopulation.
    - Any live cell with two or three live neighbors lives on to the next generation.
    - Any live cell with more than three live neighbors dies, as if by overpopulation.
    - Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction
    '''
    #get weighted sum of neighbors using convolution
    x = jsignal.convolve2d(grid, neighborhood, mode='same')
    #x = jnp.around(jsignal.fftconvolve(grid, neighborhood, mode='same'))

    #implement the GoL rules by thresholding the weights
    #If point alive then count of 2 and 3 is allowed, else just 3
    #weights = np.where(np.logical_or(3 < x, x < 2), 0, 1)
    weights = jnp.where(x==3, alive_value, dead_value) #always alive
    weights += jnp.where(jnp.logical_and(grid==alive_value,x==2), alive_value, dead_value)

    #update the grid
    return weights
