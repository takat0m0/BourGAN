# -*- coding:utf-8 -*-

'''
we here cite https://github.com/a554b554/BourGAN/tree/master/src/bourgan
'''

import os
import sys
import numpy as np

def mindist(x_id, idxset, distmat):
    """
    helper function to find the minimal distance in a given point set
    Args:
        x_id (int): id for reference point
        idxset (list): ids for the point set to test
        distmat (ndarray): distance matrix for all points
    
    Returns:
        mindist (float): minimal distance
    """
    mindist = np.inf
    for y_id in idxset[0]:
        d = distmat[x_id, y_id]
        if d < mindist:
            mindist = d
    if mindist == np.inf:
        mindist = 0
    return mindist


def bourgain_embedding(data, p, m, distmat):
    """
    bourgain embedding main function.
    Args:
        data (ndarray): Input data for embedding. Shape must be nxm, 
                        where n is the number of data points, m is the data dimension.
        p, m (float): bourgain embedding hyperparameters.
        distmat (ndarray): Distance matrix for data, shape must be nxn.
    
    Returns:
        ans (ndarray): results for bourgain embedding, shape must be nxk, where k is the
            latent space dimension.
    """
    assert(p>0 and p<1)
    assert(isinstance(m, int))
    n = data.shape[0]
    K = np.ceil(np.log(n)/np.log(1/p))
    S={}
    for j in range(int(K)):
        for i in range(m):
            S[str(i)+str('_')+str(j)]=[]
            prob = np.power(p, j+1)
            rand_num = np.random.rand(n)
            good = rand_num<prob
            good = np.argwhere(good==True).reshape((-1))
            S[str(i)+str('_')+str(j)].append(good)


    ans = np.zeros((n, int(K)*m))

    for (c, x) in enumerate(data):
        fx = np.zeros((m, int(K)))
        for i in range(fx.shape[0]):
            for j in range(fx.shape[1]):
                fx[i, j] = mindist(c, S[str(i)+str('_')+str(j)], distmat)

        fx = fx.reshape(-1)
        ans[c, :] = fx

    ans = ans - np.mean(ans, axis=0)
    dists = np.linalg.norm(ans, ord='fro')/np.sqrt(ans.shape[0])
    ans = ans/dists * np.sqrt(ans.shape[1])
    return ans


class BourganianData(object):
    p = 0.5
    m = 5
    eps = 0.01
    num_data = 1000
    
    def __init__(self, data_sample_function):
        tmp_data = data_sample_function(self.num_data)
        d_mat = self._calc_pair_distance(tmp_data)
        self._data = bourgain_embedding(tmp_data, self.p, self.m, d_mat)
        print(self._data.shape)
        
    def _calc_pair_distance(self, target_data):
        ret = []
        for i in target_data:
            tmp = [np.linalg.norm(i - j) for j in target_data]
            ret.append(tmp)
        return np.asarray(ret, dtype = np.float32)
    
    def get_z_dim(self):
        return self._data.shape[1]
    
    def __call__(self, batch_size):
        idx = np.random.choice(self.num_data, batch_size)
        sampled_data = self._data[idx, :]
        noise = np.random.normal(scale=self.eps, size=sampled_data.shape)
        sampled_data = sampled_data + noise
        return sampled_data        
    
from data_sample import gaussian_mixture_lattice

if __name__ == '__main__':
    tmp = BourganianData(gaussian_mixture_lattice)
    hoge = tmp(100)
    print(hoge.shape)
