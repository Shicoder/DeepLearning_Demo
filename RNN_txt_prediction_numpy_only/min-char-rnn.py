# -*- coding:utf-8 -*-
"""
Minimal character-level Vanilla RNN model.
原作者：Andrej Karpathy (@karpathy)
"""

## add comments by weixsong
## reference page [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
## 中文注释：DivinerShi
## 三层神经网络.
## 输入层: one hot vector, dim: vocab * 1
## 隐藏层: LSTM, hidden vector: hidden_size * 1
## 输出层: Softmax, vocab * 1, the probabilities distribution of each character


import numpy as np

# 读入数据，简单文本文件
data = open('input.txt', 'r').read()

# set()可以去重，分别进行计数
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print ('data has %d characters, %d unique.' % (data_size, vocab_size))

# 建立字典
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# 超参数设置
hidden_size = 100 # 隐藏层大小
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

# 定义权值矩阵
## RNN/LSTM
## 本文实现了基本的RNN
## # update the hidden state
## self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))
## # compute the output vector
## y = np.dot(self.W_hy, self.h)
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias


## 计算梯度和损失
## cross-entropy loss is used
## actually, here the author use cross-entropy as error,
## but in the backpropagation the author use sum of squared error (Quadratic cost) to do back propagation.
## be careful about this trick. 
## this is because the output layer is a linear layer.
## TRICK: Using the quadratic cost when we have linear neurons in the output layer, z[i] = a[i]
def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  ## record each hidden state of
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass for each training data point
  for t in range(len(inputs)):
    #one-hot编码
    xs[t] = np.zeros((vocab_size, 1))
    xs[t][inputs[t]] = 1
    
    ## 计算当前隐状态值
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)
    ##计算输出层值
    ys[t] = np.dot(Why, hs[t]) + by
    ## 计算softmax
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
    loss += -np.log(ps[t][targets[t], 0])

  #梯度反传
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(range(len(inputs))):
    ## 计算梯度误差
    ## dE/dy[j] = y[j] - t[j]
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1
    
    ##计算输出层的梯度
    ## dE/dy[j]*dy[j]/dWhy[j,k] = dE/dy[j] * h[k]
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    
    ## 计算隐藏层输入梯度
    dh = np.dot(Why.T, dy) + dhnext
    ## dtanh(x)/dx = 1 - tanh(x) * tanh(x)
    dhraw = (1 - hs[t] * hs[t]) * dh
    dbh += dhraw
    
    ## 计算隐藏层和输入层梯度
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)

  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # RNN反传是线性的，容易发生梯度爆炸，做梯度截断。

  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

## given a hidden RNN state, and a input char id, predict the coming n chars
#给定一个隐状态，一个输入字符编码，预测下n个字符。
def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model
  h is memory state, seed_ix is seed letter for first time step
  """

  ## one-hot
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1

  ixes = []
  for t in range(n):
    ## self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    ## y = np.dot(self.W_hy, self.h)
    y = np.dot(Why, h) + by
    ## softmax
    p = np.exp(y) / np.sum(np.exp(y))
    ## sample according to probability distribution
    ix = np.random.choice(range(vocab_size), p=p.ravel())

    ## update input x
    ## use the new sampled result as last input, then predict next char again.
    x = np.zeros((vocab_size, 1))
    x[ix] = 1

    ixes.append(ix)

  return ixes


## 计数
n = 0
## 数据起始点
p = 0

mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0

## main loop
while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p + seq_length + 1 >= len(data) or n == 0:
    # reset RNN memory
    ## hprev is the hiddden state of RNN
    hprev = np.zeros((hidden_size, 1))
    # go from start of data
    p = 0

  #输入输出差一个字符
  inputs = [char_to_ix[ch] for ch in data[p : p + seq_length]]
  targets = [char_to_ix[ch] for ch in data[p + 1 : p + seq_length + 1]]

  # 预测
  if n % 100 == 0:
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print ('---- sample -----')
    print ('----\n %s \n----' % (txt, ))

  # 前向
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0:
    print ('iter %d, loss: %f' % (n, smooth_loss))
  
  # Adagrad优化
  ## mem += dparam * dparam
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                [dWxh, dWhh, dWhy, dbh, dby],
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    ##  Adagrad
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length
  n += 1
