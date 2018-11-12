# -*- coding: utf-8 -*-
"""

@author: Okarim
"""
import numpy as np
import tensorflow as tf
import datetime
import os
from pathlib import Path
import random
import time

def bottomToTop(texto):
    """
    texto=nivel en formato de texto
    secuencia= secuencia de caracteres obtenidos al recorrer el nivel de abajo hacia arriba
    """
    secuencia=[]
    for i in range(len(texto[0])):
        for j in reversed(range(len(texto))):
            secuencia.append(texto[j][i])
    return secuencia

def snaking(texto):
    """
    texto=nivel en formato de texto
    secuencia= secuencia de caracteres obtenidos al recorrer el nivel serpenteando
    """
    secuencia=[]
    for i in range(len(texto[0])):
        for j in range(len(texto)):
            if(i%2==0):
                secuencia.append(texto[j][i])
            else:
                secuencia.append(texto[len(texto)-j-1][i])
    return secuencia

class Dataset():

  def __init__(self, filepath):
     
    self.text = []
     
    p = Path(filepath).glob('**/*')
    files = [x for x in p if x.is_file()]

    for f in files:
        texto=np.loadtxt(f, dtype=str, comments="~")
        self.seq1=bottomToTop(texto)
        self.text.append(self.seq1)
        
    self.idx_char = sorted(list(set(str(self.text))))
    self.char_idx = {c: i for i, c in enumerate(self.idx_char)}

  def batch(self, batch_size, seq_size):
    space = range(len(self.text) - seq_size - 1)
    sampled = random.sample(space, batch_size)
    X, Y = [], []
    for s in sampled:
      seq = self.text[s:s+seq_size]
      X.append(self.encode(seq))
      seq = self.text[s+1:s+1+seq_size]
      Y.append(self.encode(seq))
    
    return X, Y

  def decode(self, text):
    return [self.idx_char[c] for c in text]

  def encode(self, text):
    return [self.char_idx[c] for c in text]

if __name__ == '__main__':
    texto=np.loadtxt("Levels/mario-1-1.txt", dtype=str, comments="~")
    sec=bottomToTop(texto)
#    print(sec[:30])
#    
#    sec=snaking(texto)
#    print(sec[:30])
    
    ds=Dataset("Levels")
    
#    dropout = tf.placeholder(tf.float32)
#    
#    cell=tf.nn.rnn_cell.LSTMCell(1)
#    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - dropout)
#    
#    data = tf.placeholder(tf.float32, [None, None, 33])
#    output, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
#    
#    output, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
#    output = tf.transpose(output, [1, 0, 2])
#    last = tf.gather(output, int(output.get_shape()[0]) - 1)
#    
#    out_size = target.get_shape()[2].value
#    logit = tf.contrib.layers.fully_connected(
#        last, out_size, activation_fn=None)
#    prediction = tf.nn.softmax(logit)
#    loss = tf.losses.softmax_cross_entropy(target, logit)
#    
#    out_size = target.get_shape()[2].value
#    logit = tf.contrib.layers.fully_connected(
#        output, out_size, activation_fn=None)
#    prediction = tf.nn.softmax(logit)
  
  