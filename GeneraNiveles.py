# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 18:03:30 2018

@author: angel
"""

from subprocess import call
import shutil

iteraciones=100


for i in range(iteraciones):
    if i%10==0:
        print("i0: "+str(i))
    call(['python.exe', 'SuperMario_Worlds.py', '512_Worlds_Path', '0'])
    shutil.copy('testfile.txt', "Worlds0/w0_{}.txt".format(i))
    
for i in range(iteraciones):
    if i%10==0:
        print("i1: "+str(i))
    call(['python.exe', 'SuperMario_Worlds.py', '512_Worlds_Path', '1'])
    shutil.copy('testfile.txt', "Worlds1/w1_{}.txt".format(i))
    
for i in range(iteraciones):
    if i%10==0:
        print("i2: "+str(i))
    call(['python.exe', 'SuperMario_Worlds.py', '512_Worlds_Path', '2'])
    shutil.copy('testfile.txt', "Worlds2/w2_{}.txt".format(i))