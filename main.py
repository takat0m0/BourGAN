# -*- coding:utf-8 -*-

import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from model import Model
from data_sample import gaussian_mixture_lattice, gaussian_mixture_B, gaussian_mixture
from util import plot_scatter
from embedding import BourganianData

if __name__ == u'__main__':

    #this_sampler = gaussian_mixture_B
    this_sampler = gaussian_mixture
    
    # parameter
    epoch_num = 200

    batch_size = 256
    num_one_epoch = 50

    # get data
    b_data = BourganianData(this_sampler)
    z_dim = b_data.get_z_dim()
    
    # make model
    print('-- make model --')
    model = Model(z_dim)
    model.set_model()

    # training
    print('-- begin training --')
    
    with tf.Session() as sess:
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in range(epoch_num):
            print('** epoch {} begin **'.format(epoch))
            g_obj = 0.0
            d_obj = 0.0
            
            # plot p_g
            #batch_z = np.random.normal(0, 1, [10000, z_dim]).astype(np.float32)
            batch_z = b_data(10000)
            tmp = model.generate(sess, batch_z)
            plot_scatter(tmp, 'result', epoch, None)
 
            for step in range(num_one_epoch):
                
                # draw from p_z
                #batch_z = np.random.normal(0, 1, [batch_size, z_dim]).astype(np.float32)
                batch_z  = b_data(batch_size)
                batch_z2 = b_data(batch_size)

                # draw from p_data
                #batch_inputs = gaussian_mixture(batch_size)
                #batch_inputs = gaussian_mixture_B(batch_size)
                batch_inputs = this_sampler(batch_size)
                
                # train discriminator
                d_obj += model.training_disc(sess, batch_z, batch_inputs)

                # train generator
                g_obj += model.training_gen(sess,  batch_z, batch_z2)
                
            print('epoch:{}, d_obj = {}, g_obj = {}'.format(epoch,
                                                            d_obj/num_one_epoch,
                                                            g_obj/num_one_epoch))
            
            saver.save(sess, './model.dump')
