
# coding: utf-8

# 
# 
# 
# #### Topics: Sparsity (PCA and Compressive Sensing)
# #### Assigned: Wednesday May 23
# #### Due: Sunday June 10 by midnight
# 

# In[ ]:





# In[1]:


# -*- coding: utf-8 -*-
import numpy as np
from math import *
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm


# Params
n_inpoints = 200


# In[2]:


# Component1
def generateClass1(): 
    theta0 = 0
    lmb01 = 2
    lmb02 = 1
    m0 = (0,  0)
    # computing u * u.T and later multiplying with lambda
    cov01 = [[(cos(theta0))**2,    cos(theta0)*sin(theta0)],
             [(sin(theta0))*cos(theta0), (sin(theta0))**2]]
    cov02 = [[(sin(theta0))**2,    -(cos(theta0)*sin(theta0))],
             [-(cos(theta0)*sin(theta0)), (cos(theta0))**2]]
    cov0 = lmb01*np.matrix(cov01) + lmb02*np.matrix(cov02)
    print('Component1 = Mean: ',m0,' ','\t Cov :',cov0.flatten())
    cov0_det = np.linalg.det(cov0)
    x0, y0 = np.random.multivariate_normal(m0, cov0, int(n_inpoints)).T
    return x0,y0


# Component2
def generateClass2():
    theta1a = -3*pi/4
    lmb1a1 = 2
    lmb1a2 = 1/4
    m1a = (-2, 1)
    cov1a = [[(cos(theta1a))**2,    cos(theta1a)*sin(theta1a)],
             [(sin(theta1a))*cos(theta1a), (sin(theta1a))**2]]
    cov2a = [[(sin(theta1a))**2,    -(cos(theta1a)*sin(theta1a))],
             [-(cos(theta1a)*sin(theta1a)), (cos(theta1a))**2]]
    cov1a = lmb1a1*np.matrix(cov1a) + lmb1a2*np.matrix(cov2a)
    cov1a_det = np.linalg.det(cov1a)
    x1a, y1a = np.random.multivariate_normal(m1a, cov1a, int(n_inpoints)).T
    print('Component2 = Mean: ',m1a,' ','\t Cov :',cov1a.flatten())
    return x1a,y1a

# Component3
def generateClass3():
    theta1b = pi/4
    lmb1b1 = 3
    lmb1b2 = 1
    m1b = (3, 2)
    cov1b = [[(cos(theta1b))**2,    cos(theta1b)*sin(theta1b)],
             [(sin(theta1b))*cos(theta1b), (sin(theta1b))**2]]
    cov2b = [[(sin(theta1b))**2,    -(cos(theta1b)*sin(theta1b))],
             [-(cos(theta1b)*sin(theta1b)), (cos(theta1b))**2]]
    cov1b = lmb1b1*np.matrix(cov1b) + lmb1b2*np.matrix(cov2b)
    cov1b_det = np.linalg.det(cov1b)
    x1b, y1b = np.random.multivariate_normal(m1b, cov1b, int(n_inpoints)).T
    print('Component3 = Mean: ',m1b,' ','\t Cov :',cov1b.flatten())
    return x1b,y1b

x1, y1 = generateClass1()
x2, y2 = generateClass2()
x3, y3 = generateClass3()


# In[12]:


# Mixture Density of 3 component Gaussian
cnt_m1 = [int(ceil(n_inpoints*(1/2))),int(floor(n_inpoints*(1/6))),int(ceil(n_inpoints*(1/3)))] 
print('Coefficients count for 3 GMM components: ',cnt_m1)

# Randomly pick mixture components 
idx_m1 = np.argsort(np.random.random(n_inpoints))[:cnt_m1[0]]
idx_m2 = np.argsort(np.random.random(n_inpoints))[:cnt_m1[1]]
idx_m3 = np.argsort(np.random.random(n_inpoints))[:cnt_m1[2]]

# Combine the sampled arrays 
X1 = np.concatenate((np.array(y1[idx_m1]), np.array(y2[idx_m2]), np.array(y3[idx_m3])), axis = 0)
X2 = np.concatenate((np.array(x1[idx_m1]), np.array(x2[idx_m2]), np.array(x3[idx_m3])), axis = 0)
Z = np.array([1,0,0]*int(cnt_m1[0]) + [0,1,0]*int(cnt_m1[1])+ [0,0,1]*int(cnt_m1[2])).reshape(n_inpoints,3)
Z_val = np.array([1]*int(cnt_m1[0]) + [2]*int(cnt_m1[1])+ [3]*int(cnt_m1[2])).reshape(n_inpoints,1)
print('Component identifiers for GMM mixture: ', Z_val.shape)

X = np.vstack((X1, X2)).T
print("Data, X(x1,x2) Shape:",X.shape, ', Y Shape', Z_val.shape)
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121)
ax1.scatter(x1, y1, color = 'r',marker='x', label = 'Comp 1')
ax1.scatter(x2, y2, color = 'b',marker='^', label = 'Comp 2')
ax1.scatter(x3, y3, color = 'g',marker='o', label = 'Comp 3')
ax1.set_title('3 Components of gaussian Mixture')
ax1.legend()

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X[:,0], X[:,1], Z_val, s=int(Z_val.shape[0]), c='m', marker='o')
ax2.set_title('Visualize the cluster')
ax2.set_xlabel('X1')
ax2.set_ylabel('X2')
ax2.set_zlabel('Z')
fig.subplots_adjust()


# Generate a random vector u in d dimensions as follows: The components of u are i.i.d., with
# - P [u[i] = 0] = 2=3; P [u[i] = +1] = 1=6; P [u[i] = −1] = 1=6
# 
# 
# 

# In[16]:


d = 30
ul = 7

def checkCorrelation(uj1, uj2):
    udot = np.dot(uj1, uj2)
    return udot

# Generate the IID 
def generateMultiDimGaussian():
    uj_cnts = [int(ceil(d*(2/3))),int(floor(d*(1/6))),int(ceil(d*(1/6)))]
    uj_vals = np.array([0]*int(uj_cnts[0]) + [1]*int(uj_cnts[1])+ [-1]*int(uj_cnts[2]))
    print('Number of Data points: ',uj_cnts)
    u = np.zeros([ul,d])
    np.random.shuffle(uj_vals)
    u[0] = uj_vals
    for j in np.arange(1, ul):
        #print('Prev: ', u[j-1])
        np.random.shuffle(uj_vals)      # Shuffled
        u[j] = uj_vals
        #print('Curr: ', u[j])
        corr = checkCorrelation(u[j-1],u[j])
        #if too correlated then use another
        if(corr*2 > 1):
            np.random.shuffle(uj_vals)      # Shuffled again
        print(j,' Corr => ',corr)
    return u

# Uj be i.i.d
uj = generateMultiDimGaussian()
#print(uj)


# Generate d-dimensional data samples for a Gaussian mixture distribution with 3 equiprobable components
# - Zm  : Standard Gaussian (N(0, 1)) distribution
# - N   : noise vector" N ∼ N(0, σ2Id) (default value σ2 = 0:01)
# - Component 1: Generate X = u1 + Z1u2 + Z2u3 + N.
# - Component 2: Generate X = 2u4 + sqrt(2)Z1u5 + Z2u6 + N.
# - Component 3: Generate X = sqrt(2)u6 + Z1(u1 + u2) + (1/sqrt(2))Z2u5 + N

# In[17]:


num_data = 50
Zm = [np.random.normal(0,1,num_data), np.random.normal(0,1,num_data)]
N_cov = np.eye(d) * (0.01)
N_mu = np.zeros([1,d])[0]

print('Zm:',Zm[0].shape,', N:',N_cov.shape)
print('Uj:',uj[0].shape)
# Function for each component function
Xm1 = lambda z1, z2, n: (uj[0].reshape(d,1) + z1*uj[1].reshape(d,1) + z2*uj[2].reshape(d,1) + n)
Xm2 = lambda z1, z2, n: (2*uj[3].reshape(d,1) + 1.414 * z1*uj[4].reshape(d,1) + z2*uj[5].reshape(d,1) + n)
Xm3 = lambda z1, z2, n: (1.414*uj[5].reshape(d,1) + z1*(uj[0].reshape(d,1)+uj[1].reshape(d,1)) + (1/1.414)*z2*uj[4].reshape(d,1) + n)
X = np.zeros([num_data, d])

X1 = np.zeros([d,num_data])
X2 = np.zeros([d,num_data])
X3 = np.zeros([d,num_data])

# Assign the values based on the three component function
for i in range(0, num_data):
    N = np.random.multivariate_normal(N_mu, N_cov, 1)  
    X1[:,i] = Xm1(Zm[0][i], Zm[1][i], N.T).T
    X2[:,i] = Xm2(Zm[0][i], Zm[1][i], N.T).T
    X3[:,i] = Xm3(Zm[0][i], Zm[1][i], N.T).T

print('X1:',X1)
print('X2:',X2)
print('X3:',X3)

