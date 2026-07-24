'''
Different patterns of cells known
''' 

import jax.numpy as jnp
import rle

#Neural patterns
def insertInputVertical(grid, input, index=(0,0)):
    '''
    Insert a line of input receptors, where input is a 1D array
    '''
    for i, value in enumerate(input):
        grid[index[0]+i, index[1]] = value
    
    return grid

#GoL Patterns
def insertBlinker(grid, index=(0,0), value=1):
    '''
    Insert a blinker oscillator construct at the index position
    '''
    grid[index[0], index[1]+1] = value
    grid[index[0]+1, index[1]+1] = value
    grid[index[0]+2, index[1]+1] = value

    return grid
    
def insertGlider(grid, index=(0,0), value=1):
    '''
    Insert a glider construct at the index position
    '''
    grid[index[0], index[1]+1] = value
    grid[index[0]+1, index[1]+2] = value
    grid[index[0]+2, index[1]] = value
    grid[index[0]+2, index[1]+1] = value
    grid[index[0]+2, index[1]+2] = value

    return grid
    
def insertGliderGun(grid, index=(0,0), value=1):
    '''
    Insert a glider construct at the index position
    '''
    grid[index[0]+1, index[1]+25] = value
    
    grid[index[0]+2, index[1]+23] = value
    grid[index[0]+2, index[1]+25] = value
    
    grid[index[0]+3, index[1]+13] = value
    grid[index[0]+3, index[1]+14] = value
    grid[index[0]+3, index[1]+21] = value
    grid[index[0]+3, index[1]+22] = value
    grid[index[0]+3, index[1]+35] = value
    grid[index[0]+3, index[1]+36] = value
    
    grid[index[0]+4, index[1]+12] = value
    grid[index[0]+4, index[1]+16] = value
    grid[index[0]+4, index[1]+21] = value
    grid[index[0]+4, index[1]+22] = value
    grid[index[0]+4, index[1]+35] = value
    grid[index[0]+4, index[1]+36] = value
    
    grid[index[0]+5, index[1]+1] = value
    grid[index[0]+5, index[1]+2] = value
    grid[index[0]+5, index[1]+11] = value
    grid[index[0]+5, index[1]+17] = value
    grid[index[0]+5, index[1]+21] = value
    grid[index[0]+5, index[1]+22] = value
    
    grid[index[0]+6, index[1]+1] = value
    grid[index[0]+6, index[1]+2] = value
    grid[index[0]+6, index[1]+11] = value
    grid[index[0]+6, index[1]+15] = value
    grid[index[0]+6, index[1]+17] = value
    grid[index[0]+6, index[1]+18] = value
    grid[index[0]+6, index[1]+23] = value
    grid[index[0]+6, index[1]+25] = value
    
    grid[index[0]+7, index[1]+11] = value
    grid[index[0]+7, index[1]+17] = value
    grid[index[0]+7, index[1]+25] = value
    
    grid[index[0]+8, index[1]+12] = value
    grid[index[0]+8, index[1]+16] = value
    
    grid[index[0]+9, index[1]+13] = value
    grid[index[0]+9, index[1]+14] = value

    return grid

def insertFromPlainText(grid, txtString, value=1, pad=0, index=(0,0)):
    '''
    Assumes txtString contains the entire pattern as a human readable pattern
    '''
    i = 0
    j = 0
    commentCase = False
    for cell in txtString:
        if cell == '\n':
            commentCase = False
            i += 1
            j = 0
        elif commentCase:
            pass
        elif cell == '!':
            commentCase = True
        elif cell == 'O' or cell == 'o':
            grid[index[0]+i][index[1]+j] = value
            j += 1
        elif cell == '.':
            j += 1
    grid = jnp.pad(grid, pad, 'constant', constant_values=0)
    return grid

def insertFromRLE(grid, rleString, value=1, pad=0, index=(0,0)):
    '''
    Given string loaded from RLE file, populate the game grid
    '''
    parsed_text = rle.RunLengthEncodedParser(rleString)
    human_string = parsed_text.human_friendly_pattern
    insertFromPlainText(grid, human_string, value, pad, index)
