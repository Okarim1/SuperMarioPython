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

def snaking(texto, direction=0):
    """
    texto=nivel en formato de texto
    secuencia= secuencia de caracteres obtenidos al recorrer el nivel serpenteando
    direction = dirección inicial del snaking (0 = arriba-abajo, 1 = abajo-arraba)
    """
    secuencia=[]
    for i in range(len(texto[0])):
        for j in range(len(texto)):
            if(i%2==direction):
                secuencia.append(texto[j][i])
            else:
                secuencia.append(texto[len(texto)-j-1][i])
    return secuencia

def timestamp():
  return datetime.datetime.fromtimestamp(time.time()).strftime('%y%m%d-%H%M%S')

class Dataset():
  def __init__(self, filepath, snak=False, include_path=False, reduce=False):
    self.text = []
    p = Path(filepath).glob('**/*')
    files = [x for x in p if x.is_file()]
    self.n=len(files)
    for f in files:
        texto=np.loadtxt(f, dtype=str, comments="~")
        
        if reduce:
            new_text=[]
            for i in range(len(texto)):
                t=np.array(list(texto[i]))
                t[np.where(t == 'd')]='p'
                t[np.where(t == 'D')]='P'
                t[np.where(t == '{')]='['
                t[np.where(t == '}')]=']'
                t[np.where(t == 'v')]='#'
                t[np.where(t == '|')]='-'
                t[np.where(t == 'X')]='V'
                t[np.where(t == 'H')]='?'
                t[np.where(t == ',')]='-'
                new_text.append(''.join(t))
            texto=new_text
        if include_path:
	        camino=np.loadtxt("Player_Path/"+f.name, dtype=str, comments="~")
	        new_text=[]
	        for i in range(len(texto)):
	            t=np.array(list(texto[i]))
	            c=np.array(list(camino[i]))
	            t[np.where(c == 'x')]='m'
	            new_text.append(''.join(t))
	        texto=new_text
            
        if snak:
          self.seq=snaking(texto,0)
          self.text.append(self.seq)
          self.seq=snaking(texto,1)
          self.text.append(self.seq)
        else:
          self.seq=bottomToTop(texto)
          self.text.append((self.seq, [int(f.parents[0].name)-1]*len(self.seq)))
    self.X_train, self.X_test = train_test_split(self.text, test_size=0.2)
    self.idx_char = sorted(list(set(str(self.text))))
    self.char_idx = {c: i for i, c in enumerate(self.idx_char)}
    for x in self.X_test:
      print(x[1][0])

  def batch(self):
    for t in self.X_train:
      t2=t[0]
      W=t[1]
      X=[self.encode(t2[:-1]), W[:-1]]
      Y=self.encode(t2[1:])
      yield [X], [Y]

  def test(self):
    for t in self.X_test:
      t2=t[0]
      W=t[1]
      X=[self.encode(t2[:-1]), W[:-1]]
      Y=self.encode(t2[1:])
      yield [X], [Y]
  
  def decode(self, text):
    return [self.idx_char[c] for c in text]

  def encode(self, text):
    return [self.char_idx[c] for c in text]

class Generador:
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
      self.X = tf.placeholder(tf.int32, [1, 2, None], name='X')
      self.W_1h = tf.one_hot([self.X[0][1]], 3)
      
      # [batch_size,seq_size] -> 
      # [batch_size,seq_size,alpha_size]
      self.X_1 = tf.one_hot([self.X[0][0]], self.alpha_size)
      self.X_1h = tf.concat([self.X_1, self.W_1h], 2)
      # [batch_size,seq_size]
      self.Y_true = tf.placeholder(tf.int32, [1, None], name='Y_true')
      #self.Wy_1h = tf.one_hot([self.Y_true[0][1]], 3)
      # [batch_size,seq_size] -> 
      # [batch_size,seq_size,alpha_size]
      self.Y_true_1h = tf.one_hot(self.Y_true, self.alpha_size)
      #self.Y_true_1h = tf.concat([self.Y_true_1, self.Wy_1h], 2)
      # rnn initial state
      # [batch_size,cell_size*layers_num]
      self.init_state = tf.placeholder(tf.float32,
        [None, self.state_size], name='init_state')
      self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
      #self.alpha_size+=3      
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
    stepX=[[X[0][0][:], X[0][1][:]]]
    stepY=[Y_true[0][:]]
    feed = {self.X: stepX, self.Y_true: stepY, 
    self.init_state: self.last_state, self.keep_prob : self.dropout}
    fetches = [self.opt, self.state]
    _, self.last_state = sess.run(fetches, feed)

    #n=200 #200 data points for BPTT
    #for i in range(0, len(X[0][0]), n): 
    #  stepX=[[X[0][0][i:i+n], X[0][1][i:i+n]]]
    #  stepY=[Y_true[0][i:i+n]]
    #  feed = {self.X: stepX, self.Y_true: stepY, 
    #     self.init_state: self.last_state, self.keep_prob : self.dropout}
    #  fetches = [self.opt, self.state]
    #  _, self.last_state = sess.run(fetches, feed)

  def train(self, ds, epochs=1):
    """ Train the model. """
    # output message
    msg = "I{:4d} loss: {:5.4f}, acc: {:4.4f}"
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
            if not i % 1:
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
                print(msg.format(i+j*len(ds.X_train), err_trn, acc_trn))
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
    print(msg.format(i+j*len(ds.X_train), err_trn, acc_trn))

  def restore(self, ds, save_path, evaluate=False):
    msg = "I{:4d} loss: {:5.3f}, acc: {:4.2f}"
    self.sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(self.sess, save_path+"/best_acc")
    if evaluate:
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
    """
    Corre la red una vez para obtener la siguiente salida.
    """
    if reset:
      init_state = np.zeros([len(X), self.state_size])
    else:
      init_state = self.last_state
    feed = {self.X: X, self.init_state: init_state, self.keep_prob : 0.0}
    S, self.last_state = self.sess.run([self.S, self.state], feed)
    return S

def sample_from_probabilities(ds, probabilities, minProb=0.1):
  """Tira los dados para producir un entero aleatorio en el rango de ds.idx_char, 
  de acuerdo con las probabilidades proporcionadas. 
  Si se especifica minProb, solo se tienen en cuenta las probabilidades que sean mayores a ese valor.
  ds = dataset
  probabilities = lista de probabilidades para cada loseta
  minProb = probabilidad minima a considerar
  
  return carácter (loseta) escogido
  """
  p = np.squeeze(probabilities)
  if 1 in p:
  	p[np.where(p<1)]=0
  else:
  	p[np.where(p<minProb)]=0
  #p=p*p
  p = p/np.sum(p)
  return np.random.choice(ds.idx_char, p=p)[0]

def print_bottomToTop(composition):
  file = open('testfile.txt','w')
  level=[]
  for i in range(13, -1, -1):
  	level.append(composition[i::14]+'\n')
  for i in range(4):
  	if 'm' in level[-1]:
  		level.append(level[0])
  		level.remove(level[0])
  print(''.join(level))
  file.write(''.join(level))
  return

def print_snaking(composition):
  new_comp = ''
  j=0
  for i in range(0, len(composition), 14):
    if j%2 == 0:
      if i == 0:
        new_comp+=composition[13::-1]
      else:
        new_comp+=composition[i+13:i-1:-1]
    else:
      new_comp+=composition[i:i+14]
    j+=1
  print_bottomToTop(new_comp)
  return 

def main(args):
  # Cargar los datos
  dataset_path = "Levels"
  snak = False
  include_path=True
  reduce = True
  ds = Dataset(dataset_path, snak, include_path, reduce)
  alpha_size = len(ds.idx_char) #Vocabulario
  cell_size = 512
  num_layers = 3
  epochs= 1000
  dropout = 0.5

  if len(args) <= 1:
    # Entrenamiento
    print("Dataset {}, alpahbet size: {}".format(dataset_path, alpha_size))
    model = Generador(alpha_size, cell_size, num_layers, dropout)
    model.train(ds, epochs)
    world=0
  else:
    with tf.Session() as sess:
      model = Generador(alpha_size, cell_size, num_layers, dropout)
      model.restore(ds, args[1])
      world=int(args[2])

  #err_trn, acc_trn, sum_trn = 0, 0, 0
  #for k, (X_evl, Y_evl) in enumerate(ds.test()):    
  #  e, a, s = model._evaluate(model.sess, X_evl, Y_evl)
  #  err_trn+=e
  #  acc_trn+=a
  #  sum_trn=s
  #err_trn/=len(ds.X_test)
  #acc_trn/=len(ds.X_test)

  #print(err_trn, acc_trn)

  # Generación
  if include_path:
    text_seed = "#-------------#-------------#m------------"
  else:
    text_seed = "#-------------#-------------#-------------"
  composition_size = 2800 #Tamaño del nivel a crear
  mp=0.01 #Minima probabilidad a considerar
  composition = []
  #Ingresa el texto semilla a partir del cual se creará el nuevo nivel
  for i, char in enumerate(text_seed):
    idx = ds.encode(char)
    composition.append(idx[0])
    S = model.generate([[idx, [world]]], i==0)

  if include_path:
    index_m=ds.idx_char.index('m')
  index_f=ds.idx_char.index('#')
  index_pr=ds.idx_char.index(']')
  index_pl=ds.idx_char.index('[')
  index_p1=ds.idx_char.index('p')
  index_p2=ds.idx_char.index('P')
  index_cd=ds.idx_char.index('c')
  index_cu=ds.idx_char.index('C')
  index_yd=ds.idx_char.index('y')
  index_yu=ds.idx_char.index('Y')
  index_q=ds.idx_char.index('?')
  
  #Obtiene los siguientes (composition_size) caracteres para generar el nivel
  for i in range(composition_size):
    prob_dist = S[0][0]
    #char = np.random.choice(ds.idx_char, p=prob_dist)
    if not(snak):
        if include_path:
          if len(composition) % 14 == 0:
            prob_dist[index_f]+=prob_dist[index_m]
            prob_dist[index_m]=0
          if (len(composition)+1) % 14 == 0:
            prob_dist[index_q]+=prob_dist[index_m]
            prob_dist[index_m]=0
        if composition[-14] == index_pl:
            prob_dist[index_pr]=1
        else:
            prob_dist[index_pr]=0
        if composition[-14] == index_p1:
            prob_dist[index_p2]=1
        else:
            prob_dist[index_p2]=0
        if composition[-1] == index_yd:
            prob_dist[index_yu]=1
        else:
            prob_dist[index_yu]=0
    char = sample_from_probabilities(ds, prob_dist, minProb=mp)
    idx = ds.encode([char])
    composition.append(idx[0])
    #for i, char in enumerate(composition):
	#    S = model.generate([idx], i==0)
    S = model.generate([[idx, [world]]])

  composition = ds.decode(composition)
  composition = ''.join(composition)
  #print("New level:")
  #print('%s' % composition)

  print("New level:")
  if snak:
    print_snaking(composition)
  else:
    print_bottomToTop(composition)

  return 

if __name__ == '__main__':
  import sys
  sys.exit(main(sys.argv))
  #main([0,'512_BTT_reduce'])