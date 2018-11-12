# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 10:53:29 2018

@author: Okarim
"""
import numpy as np
import tensorflow as tf
import datetime
import os
from pathlib import Path
import random
import time
from sklearn.model_selection import train_test_split


from tensorflow.keras.datasets import imdb
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

def timestamp():
  return datetime.datetime.fromtimestamp(time.time()).strftime('%y%m%d-%H%M%S')

class Dataset():
  def __init__(self, filepath):
    self.text = []
    p = Path(filepath).glob('**/*')
    files = [x for x in p if x.is_file()]
    for f in files:
        texto=np.loadtxt(f, dtype=str, comments="~")
        self.seq=bottomToTop(texto)
        self.text.append(self.seq)
    self.X_train, self.X_test = train_test_split(self.text, test_size=0.2)
    self.idx_char = sorted(list(set(str(self.text))))
    self.char_idx = {c: i for i, c in enumerate(self.idx_char)}

  def batch(self):
    for t in self.X_train:
      X=[self.encode(t[:-1])]
      Y=[self.encode(t[1:])]
      yield X, Y

  def test(self):
    for t in self.X_test:
      X=[self.encode(t[:-1])]
      Y=[self.encode(t[1:])]
      yield X, Y
  
  def decode(self, text):
    return [self.idx_char[c] for c in text]

  def encode(self, text):
    return [self.char_idx[c] for c in text]

class LSTM:
  def __init__(self, alpha_size, cell_size, num_layers):
    self.alpha_size = alpha_size
    self.cell_size = cell_size
    self.num_layers = num_layers
    self.state_size = self.cell_size * self.num_layers
    self._input()
    self._model()
    self._output()
    self._loss()
    self._metrics()
    self._summaries()
    self._optimizer()
    self.sess = tf.Session()
    self.last_state = None

  def __del__(self):
    if hasattr(self, 'sess'):
      del self.sess
      
      

  def _input(self,):
    """ Defines inputs """
    with tf.variable_scope('input'):
      # [batch_size,seq_size]
      self.X = tf.placeholder(tf.int32, [None, None], name='X')
      # [batch_size,seq_size] -> 
      # [batch_size,seq_size,alpha_size]
      self.X_1h = tf.one_hot(self.X, self.alpha_size)
      # [batch_size,seq_size]
      self.Y_true = tf.placeholder(tf.int32, [None, None], name='Y_true')
      # [batch_size,seq_size] -> 
      # [batch_size,seq_size,alpha_size]
      self.Y_true_1h = tf.one_hot(self.Y_true, self.alpha_size)
      # rnn initial state
      # [batch_size,cell_size*layers_num]
      self.init_state = tf.placeholder(tf.float32,
        [None, self.state_size], name='init_state')

  def _model(self):
    """ Defines the model """
    with tf.variable_scope('model'):
      with tf.variable_scope('rnn'):
        # define rnn layers (weights) 
        rnn_layers = [tf.nn.rnn_cell.GRUCell(self.cell_size)
            for _ in range(self.num_layers)]
        # chains rnn layers
        rnn = tf.nn.rnn_cell.MultiRNNCell(rnn_layers, state_is_tuple=False)
        # [batch_size,seq_size,alpha_size] ->
        # [batch_size,seq_size,cell_size]
        Y_rnn, self.state = tf.nn.dynamic_rnn(rnn, self.X_1h,
          initial_state=self.init_state, dtype=tf.float32)
        self.seq_size = tf.shape(Y_rnn)[1]
      with tf.variable_scope('fc'):
        # Convert each sequence to multiple examples, one per step
        # [batch_size,seq_size,cell_size] ->
        # [batch_size*seq_size,cell_size]
        Y_rnn = tf.reshape(Y_rnn, [-1, self.cell_size])
        # [batch_size*seq_size,cell_size] -> 
        # [batch_size*seq_size,alpha_size]
        self.L_flat = tf.layers.dense(Y_rnn, self.alpha_size)
        self.S_flat = tf.nn.softmax(self.L_flat)
        # [batch_size*seq_size,alpha_size] -> 
        # [batch_size,seq_size,alpha_size]
        Y_shape = [-1, self.seq_size, self.alpha_size]
        self.L = tf.reshape(self.L_flat, Y_shape, name='L')
        self.S = tf.reshape(self.S_flat, Y_shape, name='S')

  def _output(self):
    """ Define model output """
    with tf.variable_scope('output'):
      # [batch_size*seq_size,alpha_size] ->
      # [batch_size*seq_size]
      self.Y_pred_flat = tf.argmax(self.S_flat, 1)
      # [batch_size*seq_size] ->
      # [batch_size,seq_size]
      Y_shape = [-1, self.seq_size]
      self.Y_pred = tf.reshape(self.Y_pred_flat, Y_shape)
      self.Y_pred = tf.cast(self.Y_pred, tf.int32, name='Y_pred')

  def _loss(self):
    """ Define loss function. """
    with tf.variable_scope('loss'):
      # [batch_size,seq_size,alpha_size] ->
      # [batch_size*seq_size,alpha_size]
      Y_true_flat = tf.reshape(self.Y_true_1h, [-1, self.alpha_size])
      # cs(labels=[batch_size*seq_size,alpha_size]
      #    logits=[batch_size*seq_size,alpha_size])
      loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=Y_true_flat, logits=self.L_flat)
      self.loss = tf.reduce_mean(loss, name='loss')

  def _metrics(self):
    """ Add metrics. """
    with tf.variable_scope('metrics'):
      equal = tf.cast(tf.equal(self.Y_true, self.Y_pred), tf.float32)
      self.acc = tf.reduce_mean(equal, name='acc')

  def _summaries(self):
    """ Add summaries for Tensorboard. """
    with tf.variable_scope('summaries'):
      tf.summary.scalar('loss', self.loss)
      tf.summary.scalar('acc', self.acc)
      self.summary = tf.summary.merge_all()

  def _optimizer(self):
    """ Setup optimizer. """
    with tf.variable_scope('optimizer'):
      grad = tf.train.AdamOptimizer(learning_rate=0.001)
      self.opt = grad.minimize(self.loss)

  def _evaluate(self, sess, X, Y_true):
    """ Perform model evaluation over X, Y. """
    if self.last_state is None:
      self.last_state = np.zeros([len(X), self.state_size])
    # init_state = np.zeros([len(X), self.state_size])
    feed = {self.X: X, self.Y_true: Y_true, 
        self.init_state: self.last_state}
    fetches = [self.loss, self.acc, self.summary]
    return sess.run(fetches, feed)

  def _train_step(self, sess, X, Y_true):
    """ Run one training step. """
    if self.last_state is None:
      self.last_state = np.zeros([len(X), self.state_size])
    # init_state = np.zeros([len(X), self.state_size])
    feed = {self.X: X, self.Y_true: Y_true, 
        self.init_state: self.last_state}
    fetches = [self.opt, self.state]
    _, self.last_state = sess.run(fetches, feed)

  def train(self, ds, epochs=1):
    """ Train the model. """
    # output message
    msg = "I{:4d} loss: {:5.3f}, acc: {:4.2f}"
    # initialize variables (params)
    self.sess.run(tf.global_variables_initializer())

    # writers for TensorBoard
    ts = timestamp()
    writer_trn = tf.summary.FileWriter(
        'graphs/11_lstm_class/{}/trn'.format(ts))
    writer_trn.add_graph(self.sess.graph)

    print("Training {}".format(ts))
    for j in range(epochs):
        for i, (X, Y_true) in enumerate(ds.batch()):
            # evaluation
            if not i % 5:
                err_trn, acc_trn, sum_trn = 0, 0, 0
                for k, (X_evl, Y_evl) in enumerate(ds.test()):    
                    e, a, s = self._evaluate(self.sess, X_evl, Y_evl)
                    err_trn+=e
                    acc_trn+=a
                    sum_trn=s
                err_trn/=len(ds.X_test)
                acc_trn/=len(ds.X_test)
                writer_trn.add_summary(sum_trn, i)
                print(msg.format(i, err_trn, acc_trn))
            # train step
            #X, Y_true = ds.batch()
            self._train_step(self.sess, X, Y_true)

      # final evaluation
    #X_evl, Y_evl = ds.batch()
    
    err_trn, acc_trn, sum_trn = self._evaluate(self.sess, X, Y_true)
    writer_trn.add_summary(sum_trn, i)
    print(msg.format(i, err_trn, acc_trn))

  def generate(self, X, reset=False):
    if reset:
      init_state = np.zeros([len(X), self.state_size])
    else:
      init_state = self.last_state
    feed = {self.X: X, self.init_state: init_state}
    S, self.last_state = self.sess.run([self.S, self.state], feed)
    return S

def main(args):
  # data loading
  dataset_path = "Levels"
  ds = Dataset(dataset_path)
  
  # training
  alpha_size = len(ds.idx_char) #Vocabulario
  cell_size = 64
  num_layers = 3
  epochs= 2
  print("Dataset {}, alpahbet size: {}".format(dataset_path, alpha_size))
  model = LSTM(alpha_size, cell_size, num_layers)
  model.train(ds, epochs)

  # generation
  #text_seed = "#-------------#-"
  texto=np.loadtxt("Levels/mario-1-1.txt", dtype=str, comments="~")
  sec=bottomToTop(texto)
  text_seed=sec[:42]
  print(text_seed)
  
  composition_size = 1000
  composition = []
  for i, char in enumerate(text_seed):
    idx = ds.encode(char)
    composition.append(idx[0])
    S = model.generate([idx], i==0)

  for i in range(composition_size):
    prob_dist = S[0][0]
    char = np.random.choice(ds.idx_char, p=prob_dist)
    idx = ds.encode([char])
    composition.append(idx[0])
    S = model.generate([idx])

  composition = ds.decode(composition)
  composition = ''.join(composition)
  print("Our composition:")
  print('%s' % composition)

  return model

if __name__ == '__main__':    
    model=main(0)