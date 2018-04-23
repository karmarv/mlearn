
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# In[2]:


# Linear line
# x_vals = np.linspace(start=0, stop=5, num=50)
# y_vals = 2 * x_vals + 1
# input_shape = 1

# Line with noise
sigma = 5
x_vals = np.linspace(start=0, stop=5, num=50)
y_vals = 2 * x_vals + 1 + sigma* np.random.randn(50)
input_shape = 1


# In[3]:


plt.plot(x_vals, y_vals, 'r')
plt.show()


# In[4]:


def visualize_line(w_val):
    x_grid = np.linspace(start=0, stop=5, num=50)
    y_guessed = w_val[0] * x_grid + w_val[1]

    plt.plot(x_grid, y_guessed, 'b')
    plt.plot(x_vals, y_vals, 'r')
    
    plt.ylim((np.min(y_vals),np.max(y_vals)))
    plt.xlim((0,5))

    plt.show()


# In[5]:


x_tf = tf.placeholder(tf.float64, [])
y_tf = tf.placeholder(tf.float64, [])

w = np.zeros([2])
w_tf = tf.Variable(initial_value=w, dtype=tf.float64)


# In[6]:


yhat_tf =  tf.reduce_sum(w_tf[0] * x_tf) + w_tf[1]
loss = 1/2. * (y_tf - yhat_tf) ** 2


# In[7]:


lr = 0.001
optimization = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)


# In[8]:


sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)


# In[9]:


epochs = 15
for ee in range(epochs):
    epoch_loss = 0.
    for ii in range(x_vals.shape[0]):
        x_i = x_vals[ii]
        y_i = y_vals[ii]

        run_dict = {'optimization': optimization, 'loss':loss}
        feed_dict = {x_tf:x_i, y_tf:y_i}
        out_dict = sess.run(run_dict, feed_dict=feed_dict)
        epoch_loss += out_dict['loss']

        # w_trained = sess.run(w_tf)
        # print('Current loss: %.5f' %  out_dict['loss'])
        # visualize_line(w_trained)

    w_trained = sess.run(w_tf)
    print('Epoch loss: %.5f' % epoch_loss)
    visualize_line(w_trained)


# In[10]:


print('w: %.2f, b: %.2f' % (w_trained[0], w_trained[1]))

