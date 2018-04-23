
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# In[3]:


# img_index = 2
for img_index in range(10):
    img_to_show = mnist.train.images[img_index]
    img_to_show = np.reshape(img_to_show, [28,28])
    plt.imshow(img_to_show)
    plt.show()


# In[4]:


# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


# In[5]:


# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax


# In[6]:


# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))


# In[7]:


# Gradient Descent
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# In[8]:


# Test model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[9]:


# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# In[10]:


sess =  tf.Session()
sess.run(init)


# In[11]:


# Parameters
training_epochs = 25
batch_size = 100
display_step = 1
# Training cycle
for epoch in range(training_epochs):
    avg_cost = 0.
    avg_accuracy = 0.
    total_batch = int(mnist.train.num_examples/batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c, a = sess.run([optimizer, cost, accuracy], feed_dict={x: batch_xs,
                                                      y: batch_ys})
        # Compute average loss
        avg_cost += c / total_batch
        
        # Compute Accuracy So far
        avg_accuracy += a / total_batch
    # Display logs per epoch step
    if (epoch+1) % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost), "accuracy=", "{:.2f}".format(avg_accuracy))


# In[12]:


print("Accuracy:", sess.run(accuracy, {x: mnist.test.images, y: mnist.test.labels}))


# In[13]:


# img_index = 2
for img_index in range(10):
    test_img = mnist.test.images[img_index]
    test_label = mnist.test.labels[img_index]
    img_to_show = np.reshape(test_img, [28,28])
    predicted_label = sess.run(pred, {x:np.expand_dims(test_img,0)})
    print("Model's Predicted Label: %i" %np.argmax(predicted_label))
    plt.imshow(img_to_show)
    plt.show()

