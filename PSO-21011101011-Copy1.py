#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

class Particle:
    def __init__(self, no_dim, x_range, v_range):

        self.x = np.random.uniform(
            x_range[0], x_range[1], (no_dim,)
        )  
        self.v = np.random.uniform(
            v_range[0], v_range[1], (no_dim,)
        )  
        self.pbest = np.inf
        self.pbestpos = np.zeros((no_dim,))


class Swarm:


    def __init__(self, no_particle, no_dim, x_range, v_range, iw_range, c):

        self.p = np.array(
            [Particle(no_dim, x_range, v_range) for i in range(no_particle)]
        )
        self.gbest = np.inf
        self.gbestpos = np.zeros((no_dim,))
        self.x_range = x_range
        self.v_range = v_range
        self.iw_range = iw_range
        self.c0 = c[0]
        self.c1 = c[1]
        self.no_dim = no_dim

    def optimize(self, function, X, Y, print_step, iter):

        for i in range(iter):
            for particle in self.p:
                fitness = function(X, Y, particle.x)

                if fitness < particle.pbest:
                    particle.pbest = fitness
                    particle.pbestpos = particle.x.copy()

                if fitness < self.gbest:
                    self.gbest = fitness
                    self.gbestpos = particle.x.copy()

            for particle in self.p:
                # Here iw is inertia weight...
                iw = np.random.uniform(self.iw_range[0], self.iw_range[1], 1)[0]
                particle.v = (
                    iw * particle.v
                    + (
                        self.c0
                        * np.random.uniform(0.0, 1.0, (self.no_dim,))
                        * (particle.pbestpos - particle.x)
                    )
                    + (
                        self.c1
                        * np.random.uniform(0.0, 1.0, (self.no_dim,))
                        * (self.gbestpos - particle.x)
                    )
                )
                particle.x = particle.x + particle.v

            if i % print_step == 0:
                print("iteration#: ", i + 1, " loss: ", fitness)

        print("global best loss: ", self.gbest)

    def get_best_solution(self):

        return self.gbestpos


# In[12]:


import pandas as pd
import numpy as np


df = pd.read_csv('Bank_Personal_Loan_Modelling.csv')

X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values


INPUT_NODES = X.shape[1]
HIDDEN_NODES = 20
OUTPUT_NODES = len(np.unique(Y))


def one_hot_encode(Y):
    num_unique = len(np.unique(np.array(Y)))
    zeros = np.zeros((len(Y), num_unique))
    zeros[range(len(Y)), Y] = 1
    return zeros


def softmax(logits):

    exps = np.exp(logits)
    return exps / np.sum(exps, axis=1, keepdims=True)


def Negative_Likelihood(probs, Y):

    num_samples = len(probs)
    corect_logprobs = -np.log(probs[range(num_samples), Y])
    return np.sum(corect_logprobs) / num_samples


def Cross_Entropy(probs, Y):

    num_samples = len(probs)
    ind_loss = np.max(-1 * Y * np.log(probs + 1e-12), axis=1)
    return np.sum(ind_loss) / num_samples


def forward_pass(X, Y, W):


    if isinstance(W, Particle):
        W = W.x

    w1 = W[0 : INPUT_NODES * HIDDEN_NODES].reshape((INPUT_NODES, HIDDEN_NODES))
    b1 = W[
        INPUT_NODES * HIDDEN_NODES : (INPUT_NODES * HIDDEN_NODES) + HIDDEN_NODES
    ].reshape((HIDDEN_NODES,))
    w2 = W[
        (INPUT_NODES * HIDDEN_NODES)
        + HIDDEN_NODES : (INPUT_NODES * HIDDEN_NODES)
        + HIDDEN_NODES
        + (HIDDEN_NODES * OUTPUT_NODES)
    ].reshape((HIDDEN_NODES, OUTPUT_NODES))
    b2 = W[
        (INPUT_NODES * HIDDEN_NODES)
        + HIDDEN_NODES
        + (HIDDEN_NODES * OUTPUT_NODES) : (INPUT_NODES * HIDDEN_NODES)
        + HIDDEN_NODES
        + (HIDDEN_NODES * OUTPUT_NODES)
        + OUTPUT_NODES
    ].reshape((OUTPUT_NODES,))

    z1 = np.dot(X, w1) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(a1, w2) + b2
    logits = z2

    probs = softmax(logits)

    # Here we can calculate Categorical Cross Entropy from probs in case we have one-hot encoded vector
    # ,or calculate Negative Log Likelihood from logits without one-hot encoded vector

    # We're going to calculate Negative Likelihood, because we didn't one-hot encoded Y target...
    return Negative_Likelihood(probs, Y)
    # return Cross_Entropy(probs, Y) #used in case of one-hot vector target Y...


def predict(X, W):

    w1 = W[0 : INPUT_NODES * HIDDEN_NODES].reshape((INPUT_NODES, HIDDEN_NODES))
    b1 = W[
        INPUT_NODES * HIDDEN_NODES : (INPUT_NODES * HIDDEN_NODES) + HIDDEN_NODES
    ].reshape((HIDDEN_NODES,))
    w2 = W[
        (INPUT_NODES * HIDDEN_NODES)
        + HIDDEN_NODES : (INPUT_NODES * HIDDEN_NODES)
        + HIDDEN_NODES
        + (HIDDEN_NODES * OUTPUT_NODES)
    ].reshape((HIDDEN_NODES, OUTPUT_NODES))
    b2 = W[
        (INPUT_NODES * HIDDEN_NODES)
        + HIDDEN_NODES
        + (HIDDEN_NODES * OUTPUT_NODES) : (INPUT_NODES * HIDDEN_NODES)
        + HIDDEN_NODES
        + (HIDDEN_NODES * OUTPUT_NODES)
        + OUTPUT_NODES
    ].reshape((OUTPUT_NODES,))

    z1 = np.dot(X, w1) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(a1, w2) + b2
    logits = z2

    probs = softmax(logits)
    Y_pred = np.argmax(probs, axis=1)
    return Y_pred


def get_accuracy(Y, Y_pred):

    return (Y == Y_pred).mean()
    # return (np.argmax(Y, axis=1) == Y_pred).mean() #used in case of one-hot vector and loss is Negative Likelihood.


if __name__ == "__main__":
    no_solution = 100
    no_dim = (
        (INPUT_NODES * HIDDEN_NODES)
        + HIDDEN_NODES
        + (HIDDEN_NODES * OUTPUT_NODES)
        + OUTPUT_NODES
    )
    w_range = (0.0, 1.0)
    lr_range = (0.0, 1.0)
    iw_range = (0.9, 0.9)  # iw -> inertial weight...
    c = (0.5, 0.3)  # c[0] -> cognitive factor, c[1] -> social factor...

    s = Swarm(no_solution, no_dim, w_range, lr_range, iw_range, c)
    # Y = one_hot_encode(Y) #Encode here...
    s.optimize(forward_pass, X, Y, 100, 1000)
    W = s.get_best_solution()
    Y_pred = predict(X, W)
    accuracy = get_accuracy(Y, Y_pred)
    print("Accuracy: %.3f" % (accuracy*100))


# In[ ]:





# In[ ]:




