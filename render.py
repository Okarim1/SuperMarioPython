# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 19:55:07 2018

@author: Okarim
"""

import numpy as np
import matplotlib.pyplot as plt

def render_level(array, ncols):
    
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result

def make_array(texto):
    from PIL import Image
    path='Tiles\\SMBTiles\\'
    
    level=[]
    for t in list(texto):
        for c in t:
            if c == '-':
                level.append(np.array(np.asarray(Image.open(path+'sky.png').convert('RGB'))))
            elif c == '#':
                level.append(np.array(np.asarray(Image.open(path+'tileset_tile33.png').convert('RGB'))))
            elif c == '?':
                level.append(np.array(np.asarray(Image.open(path+'tileset_tile24.png').convert('RGB'))))
            elif c == 'B':
                level.append(np.array(np.asarray(Image.open(path+'tileset_tile01.png').convert('RGB'))))
            elif c == 'M':
                level.append(np.array(np.asarray(Image.open(path+'tileset_tileMushroom.png').convert('RGB'))))
            elif c == 'p':
                level.append(np.array(np.asarray(Image.open(path+'pipe.png').convert('RGB'))))
            elif c == 'P':
                level.append(np.array(np.asarray(Image.open(path+'pipe_r.png').convert('RGB'))))
            elif c == '[':
                level.append(np.array(np.asarray(Image.open(path+'pipe_ul.png').convert('RGB'))))
            elif c == ']':
                level.append(np.array(np.asarray(Image.open(path+'pipe_ur.png').convert('RGB'))))
            elif c == 'g':
                level.append(np.array(np.asarray(Image.open(path+'goomba.png').convert('RGB'))))
            elif c == '+':
                level.append(np.array(np.asarray(Image.open(path+'tileset_tileMushroomLL2.png').convert('RGB'))))
            elif c == 'O':
                level.append(np.array(np.asarray(Image.open(path+'tileset_tile58.png').convert('RGB'))))
            elif c == 'o':
                level.append(np.array(np.asarray(Image.open(path+'tileset_tile57.png').convert('RGB'))))
            elif c == 'k':
                level.append(np.array(np.asarray(Image.open(path+'turtle2.png').convert('RGB'))))
            elif c == 'K':
                level.append(np.array(np.asarray(Image.open(path+'fly4.png').convert('RGB'))))
            elif c == '*':
                level.append(np.array(np.asarray(Image.open(path+'tileset_tileMushroomLL3.png').convert('RGB'))))
            elif c == 'V':
                level.append(np.array(np.asarray(Image.open(path+'PPR.png').convert('RGB'))))
            elif c == 'y':
                level.append(np.array(np.asarray(Image.open(path+'y1.png').convert('RGB'))))
            elif c == 'Y':
                level.append(np.array(np.asarray(Image.open(path+'Y2.png').convert('RGB'))))
            elif c == 'H':
                level.append(np.array(np.asarray(Image.open(path+'tileset_tile24.png').convert('RGB'))))
            elif c == 'h':
                level.append(np.array(np.asarray(Image.open(path+'hammerbro.png').convert('RGB'))))
            elif c == 'C':
                level.append(np.array(np.asarray(Image.open(path+'tileset_tile09.png').convert('RGB'))))
            elif c == 'c':
                level.append(np.array(np.asarray(Image.open(path+'tileset_tile42.png').convert('RGB'))))
            elif c == 't':
                level.append(np.array(np.asarray(Image.open(path+'t.png').convert('RGB'))))
            elif c == 'l':
                level.append(np.array(np.asarray(Image.open(path+'l.png').convert('RGB'))))
            elif c == 'm':
                level.append(np.array(np.asarray(Image.open(path+'m.png').convert('RGB'))))
            else:
                level.append(np.array(np.asarray(Image.open(path+'URg.png').convert('RGB'))))
    return np.array(level)


dataset_path = "ComputerLevels\\"
text=np.loadtxt(dataset_path+"5.txt", dtype=str, comments="~")

array = make_array(text)
result = render_level(array, ncols=len(text[0]))
plt.figure(figsize = (100,14))
plt.imshow(result,interpolation='nearest')

plt.show()