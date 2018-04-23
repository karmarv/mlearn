
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
get_ipython().magic(u'matplotlib inline')
np.random.seed(10)


# In[2]:


import os
# mask visible GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf


# In[3]:


# Define our points
# easy constant, separable points
X1 = [[-2,4], [4,1]]
Y1 = [-1, -1]

X2 = [[1,6], [2,4], [6,2]]
Y2 = [1, 1, 1]

# Random separable
# mean = [1,1]
# cov = [[1,0],[0,1]]
# size = 50
# X1 = np.random.multivariate_normal(mean, cov, size)
# Y1 = np.ones([size]) * -1

# mean = [6,6]
# cov = [[1,0],[0,1]]
# size = 50
# X2 = np.random.multivariate_normal(mean, cov, size)
# Y2 = np.ones([size]) 


# Random XOR
# mean = [1,1]
# cov = [[1,0],[0,1]]
# size = 50
# X1 = np.random.multivariate_normal(mean, cov, size)
# mean = [5,5]
# X1 = np.concatenate([X1, np.random.multivariate_normal(mean, cov, size)], axis=0)
# Y1 = np.ones([size*2]) * -1

# mean = [1,5]
# cov = [[1,0],[0,1]]
# size = 50
# X2 = np.random.multivariate_normal(mean, cov, size)
# Y2 = np.ones([size]) 
# mean = [5,1]
# X2 = np.concatenate([X2, np.random.multivariate_normal(mean, cov, size)], axis=0)
# Y2 = np.ones([size*2]) * 1


# In[4]:


# Helper visualization function
def visualize(points1, points2, line=[0,0,0]):
	for ii, sample in enumerate(points1):
		plt.scatter(sample[0], sample[1], s=120, marker='_', c='g')

	for ii, sample in enumerate(points2):
		plt.scatter(sample[0], sample[1], s=120, marker='+', c='r')	

	w1, w2, b = line
	
	plt.plot([-5, 10], [-(b+-5*w1)/(w2+0.01), -(b+10*w1)/(w2+0.01)])

	plt.ylim((-5,10))
	plt.xlim((-5,10))


# In[5]:


# Initialize our perceptron
w = [0.,0.]
b = 0.
visualize(X1, X2, [w[0], w[1], b])
plt.show()


# In[6]:


# set our learning rate
lr = 1
epochs = 20

# combine our points and cast everything into numpy arrays
X = np.concatenate([X1, X2], axis=0)
Y = np.concatenate([Y1, Y2], axis=0)
w = np.array(w)

# random_indices = np.random.permutation(Y.shape[0])
# X = X[random_indices, :]
# Y = Y[random_indices]


# In[7]:


# Define tensorflow graph

# create tensorflow placeholders for the inputs
x_tf = tf.placeholder(tf.float64, [2,])
y_tf = tf.placeholder(tf.float64, [])

# create tf variables
w_tf = tf.Variable(initial_value=w, dtype=tf.float64)
b_tf = tf.Variable(initial_value=b, dtype=tf.float64)

y_hat_tf = tf.reduce_sum(x_tf * w_tf) + b_tf

# Define the loss function
# -0.00001 is to solve the gradient division issue of reduce max.
# this way if (y_tf * y_hat_tf) == 0, gradient wont be divided
loss_tf = tf.reduce_max([-0.00001, - (y_tf * y_hat_tf)])

grad_tf = tf.gradients(loss_tf, [w_tf, b_tf])

# Define our optimizer
optimization = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss_tf)


# In[8]:


# Set session
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options)#,log_device_placement=True)
sess = tf.Session(config=config)
# sess = tf.Session()

# Initialize Variable values
init_op = tf.global_variables_initializer()
sess.run(init_op)


# In[9]:


losses = []
# Train the weights
for ee in range(epochs):
    epoch_loss = 0.
    for ii in range(Y.shape[0]):
        x = X[ii,:]
        y = Y[ii]
        
        # Basic way of running things
#         _, grad, current_loss = sess.run([optimization, grad_tf, loss_tf], feed_dict={x_tf:x, y_tf:y})
#         epoch_loss += current_loss

        # Better
        run_dict = {'optimization': optimization, 'gradient':grad_tf, 'loss':loss_tf}
        feed_dict = {x_tf:x, y_tf:y}
        out_dict = sess.run(run_dict, feed_dict=feed_dict)
        epoch_loss += out_dict['loss']
        
        


    w_trained, b_trained = sess.run([w_tf, b_tf]) 
    visualize(X1, X2, [w_trained[0], w_trained[1], b_trained])
    plt.show()


# In[10]:


w_trained, b_trained = sess.run([w_tf, b_tf])
visualize(X1, X2, [w_trained[0], w_trained[1], b_trained])
plt.show()


# In[11]:


print([w_trained[0], w_trained[1], b_trained])

