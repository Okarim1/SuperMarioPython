# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 10:53:29 2018

@author: Okarim
"""
import numpy as np
import tensorflow as tf
import datetime
from pathlib import Path
import time
from sklearn.model_selection import train_test_split

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
    self.n=len(files)
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
  def __init__(self, alpha_size, cell_size, num_layers, dropout):
    self.dropout=dropout
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
    #config = tf.ConfigProto(device_count = {'GPU': 0})
    #self.sess = tf.Session(config=config)
    self.sess = tf.Session()
    self.last_state = None

  def __del__(self):
    if hasattr(self, 'sess'):
      del self.sess

  def _input(self,):
    """ Define las entradas """
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
      self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

  def _model(self):
    """ Define el modelo de la red """
    with tf.variable_scope('model'):
      with tf.variable_scope('rnn'):
        # define las capas de la rnn
        rnn_layers = [tf.nn.rnn_cell.DropoutWrapper( tf.nn.rnn_cell.GRUCell(self.cell_size),
         output_keep_prob=1.0-self.keep_prob)
            for _ in range(self.num_layers)]
        # Encadena las capas de la rnn
        rnn = tf.nn.rnn_cell.MultiRNNCell(rnn_layers, state_is_tuple=False)
        # [batch_size,seq_size,alpha_size] ->
        # [batch_size,seq_size,cell_size]
        Y_rnn, self.state = tf.nn.dynamic_rnn(rnn, self.X_1h,
          initial_state=self.init_state, dtype=tf.float32)
        self.seq_size = tf.shape(Y_rnn)[1]
      with tf.variable_scope('fc'):
        # Convierte cada secuencia a multiples ejemplos, uno por paso
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
    """ Define el modelo de salida """
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
    """ Define la funcion de perdida. """
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
    """ Agregar metricas. """
    with tf.variable_scope('metrics'):
      equal = tf.cast(tf.equal(self.Y_true, self.Y_pred), tf.float32)
      self.acc = tf.reduce_mean(equal, name='acc')

  def _summaries(self):
    """ Agregar resumenes para Tensorboard. """
    with tf.variable_scope('summaries'):
      tf.summary.scalar('loss', self.loss)
      tf.summary.scalar('acc', self.acc)
      self.summary = tf.summary.merge_all()

  def _optimizer(self):
    """ Preparar el optimizador. """
    with tf.variable_scope('optimizer'):
      grad = tf.train.AdamOptimizer(learning_rate=0.001)
      self.opt = grad.minimize(self.loss)

  def _evaluate(self, sess, X, Y_true):
    """ Realizar la evaluacion del modelo sobre X, Y. """
    self.last_state = np.zeros([len(X), self.state_size])
    #if self.last_state is None:
    #  self.last_state = np.zeros([len(X), self.state_size])
    # init_state = np.zeros([len(X), self.state_size])
    feed = {self.X: X, self.Y_true: Y_true, 
        self.init_state: self.last_state, self.keep_prob : 0.0}
    fetches = [self.loss, self.acc, self.summary]
    return sess.run(fetches, feed)

  def _train_step(self, sess, X, Y_true):
    """ Run one training step. """
    self.last_state = np.zeros([len(X), self.state_size])
    #if self.last_state is None:
    #  self.last_state = np.zeros([len(X), self.state_size])
    # init_state = np.zeros([len(X), self.state_size])
    n=200 #200 data points for BPTT
    for i in range(0, len(X), n): 
      stepX=X[i:i+n]
      stepY=Y_true[i:i+n]
      feed = {self.X: stepX, self.Y_true: stepY, 
         self.init_state: self.last_state, self.keep_prob : self.dropout}
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
    best_acc = 0
    best_err = 1.0
    saver = tf.train.Saver()
    save_path = 'checkpoints/best_acc'
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

                if (acc_trn > best_acc) or (acc_trn == best_acc and err_trn < best_err) :
                  best_acc=acc_trn
                  best_err = err_trn
                  saver.save(sess=self.sess, save_path=save_path)
                writer_trn.add_summary(sum_trn, i+j*len(ds.X_train))
                print(msg.format(i+j*ds.n, err_trn, acc_trn))
            # train step
            #X, Y_true = ds.batch()
            self._train_step(self.sess, X, Y_true)

      # final evaluation
    err_trn, acc_trn, sum_trn = 0, 0, 0
    for k, (X_evl, Y_evl) in enumerate(ds.test()):    
        e, a, s = self._evaluate(self.sess, X_evl, Y_evl)
        err_trn+=e
        acc_trn+=a
        sum_trn=s
    err_trn/=len(ds.X_test)
    acc_trn/=len(ds.X_test)
    writer_trn.add_summary(sum_trn, i+j*len(ds.X_train))
    print(msg.format(i+j*ds.n, err_trn, acc_trn))

  def restore(self, ds, save_path):
    msg = "I{:4d} loss: {:5.3f}, acc: {:4.2f}"
    self.sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(self.sess, save_path+"/best_acc")
    err_trn, acc_trn, sum_trn = 0, 0, 0
    for k, (X_evl, Y_evl) in enumerate(ds.test()):    
        e, a, s = self._evaluate(self.sess, X_evl, Y_evl)
        err_trn+=e
        acc_trn+=a
        sum_trn=s
    err_trn/=len(ds.X_test)
    acc_trn/=len(ds.X_test)
    print(msg.format(0, err_trn, acc_trn))

  def generate(self, X, reset=False):
    if reset:
      init_state = np.zeros([len(X), self.state_size])
    else:
      init_state = self.last_state
    feed = {self.X: X, self.init_state: init_state, self.keep_prob : 0.0}
    S, self.last_state = self.sess.run([self.S, self.state], feed)
    return S

def sample_from_probabilities(ds, probabilities, minProb=0.1):
  """Roll the dice to produce a random integer in the [0..ALPHASIZE] range,
  according to the provided probabilities. If topn is specified, only the
  topn highest probabilities are taken into account.
  :param probabilities: a list of size ALPHASIZE with individual probabilities
  :param topn: the number of highest probabilities to consider. Defaults to all of them.
  :return: a random integer
  """
  p = np.squeeze(probabilities)
  p[np.where(p<minProb)]=0
  #p[np.argsort(p)[:-topn]] = 0
  p = p / np.sum(p)
  return np.random.choice(ds.idx_char, p=p)[0]

def main(args):
  # Cargar los datos
  dataset_path = "simpleLevels"
  ds = Dataset(dataset_path)
  alpha_size = len(ds.idx_char) #Vocabulario
  cell_size = 512
  num_layers = 3
  epochs= 1000
  dropout = 0.5
  if len(args) <= 1:
    # Entrenamiento
    print("Dataset {}, alpahbet size: {}".format(dataset_path, alpha_size))
    model = LSTM(alpha_size, cell_size, num_layers, dropout)
    model.train(ds, epochs)
  else:
    with tf.Session() as sess:
      model = LSTM(alpha_size, cell_size, num_layers, dropout)
      model.restore(ds, args[1])

  # generation
  #text_seed = "#-------------#-------------"
  texto=np.loadtxt(dataset_path+"/mario-1-1.txt", dtype=str, comments="~")
  sec=bottomToTop(texto)
  text_seed=sec[:28]

  composition_size = 1400
  composition = []
  for i, char in enumerate(text_seed):
    idx = ds.encode(char)
    composition.append(idx[0])
    S = model.generate([idx], i==0)

  for i in range(composition_size):
    prob_dist = S[0][0]
    char = np.random.choice(ds.idx_char, p=prob_dist)
    char = sample_from_probabilities(ds, prob_dist, minProb=0.01)
    idx = ds.encode([char])
    composition.append(idx[0])
    S = model.generate([idx])

  composition = ds.decode(composition)
  composition = ''.join(composition)
  #print("New level:")
  #print('%s' % composition)

  print("New level:")
  for i in range(13, -1, -1):
      print(composition[i::14])

  return 

if __name__ == '__main__':
  import sys
  sys.exit(main(sys.argv))