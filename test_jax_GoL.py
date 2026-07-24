'''
Implementation of the Game of Life (GoL) in JAX
'''
import utils #local
import patterns #local
import evolve
#import jax.numpy as jnp
import numpy as np

utils.jax_status()

N = 256 
insert_point = (32,32)
grid_size = (N,N) 
universe = np.zeros(grid_size) 

#universe = patterns.insertGlider(universe, index=insert_point)
universe = patterns.insertGliderGun(universe, index=insert_point)

#plot cells
import matplotlib.pyplot as plt
import matplotlib.animation as animation


fig = plt.figure()

plt.gray()

img = plt.imshow(universe, animated=True)

def animate(i):
    """perform animation step"""
    global universe

    #universe = evolve.evolve(universe)
    universe = evolve.evolve_fast(universe) 

    img.set_array(universe)
    
    return img,

interval = 10 #ms

#animate 24 frames with interval between them calling animate function at each frame
ani = animation.FuncAnimation(fig, animate, frames=24, interval=interval, blit=True)

plt.show()

