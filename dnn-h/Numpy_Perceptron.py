
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
get_ipython().magic(u'matplotlib inline')
np.random.seed(10)


# In[2]:


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


# In[3]:


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


# In[4]:


# Initialize our perceptron
w = [0.,0.]
b = 0.
visualize(X1, X2, [w[0], w[1], b])
plt.show()


# In[5]:


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


# In[6]:


losses = []
for ee in range(epochs):
	epoch_loss = 0.
	for ii in range(Y.shape[0]):
		x = X[ii,:]
		y = Y[ii]
		y_hat = np.sum(w * x)+ b

		loss = max(0, 0-y_hat*y)
		epoch_loss += loss

		if (y_hat * y) <= 0.:
			w = w + lr * x * y
			b = b + lr * 1 * y

		# plt.scatter(x[0], x[1], s=120, marker='*')
	losses.append(epoch_loss)
	visualize(X1, X2, [w[0], w[1], b])
	plt.show()


visualize(X1, X2, [w[0], w[1], b])
plt.show()


# In[7]:


plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.show()


# In[8]:


print([w[0], w[1], b])

