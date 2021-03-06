# -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf

from generator import Generator
from discriminator import Discriminator

from tf_util import flatten

class Model(object):
    def __init__(self, z_dim):

        self.z_dim = z_dim
        self.true_input_dim = 2
        self.lr = 0.001
        self.beta = 0.1
        
        # generator config
        gen_layer = [z_dim, 128, 128, self.true_input_dim]

        #discriminato config
        disc_layer = [self.true_input_dim, 128, 128, 1]

        # -- generator -----
        self.gen = Generator([u'gen_deconv'], gen_layer)

        # -- discriminator --
        self.disc = Discriminator([u'disc_conv'], disc_layer)

    def set_model(self):

        # -- define place holder -------
        self.z = tf.placeholder(tf.float32, [None, self.z_dim])
        self.z2 = tf.placeholder(tf.float32, [None, self.z_dim])
        self.true_input= tf.placeholder(tf.float32, [None, self.true_input_dim])
        
        # -- generator -----------------
        gen_out = self.gen.set_model(self.z, True, False)
        g_logits = self.disc.set_model(gen_out, True, False)
        gen_out2 = self.gen.set_model(self.z2, True, True)
        
        gen_dist = 0.5 * tf.log(
            tf.reduce_sum(tf.square(gen_out - gen_out2), axis = 1)
        )
        
        z_dist = 0.5 * tf.log(
            tf.reduce_sum(tf.square(self.z - self.z2), axis = 1)
        )
        
        self.g_obj =  tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = g_logits,
                labels = tf.ones_like(g_logits)
            )
        )
        self.g_obj += self.beta * tf.reduce_mean(tf.square(gen_dist - z_dist))

        self.train_gen  = tf.train.AdamOptimizer(self.lr, beta1 = 0.5).minimize(self.g_obj, var_list = self.gen.get_variables())
        
        # -- discriminator --------
        d_logits = self.disc.set_model(self.true_input, True, True)

        d_obj_true = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(            
                logits = d_logits,
                labels = tf.ones_like(d_logits)
            )
        )
        d_obj_false = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(            
                logits = g_logits,
                labels = tf.zeros_like(g_logits)
            )
        )
        self.d_obj = d_obj_true + d_obj_false
        self.train_disc = tf.train.AdamOptimizer(self.lr, beta1 = 0.5).minimize(self.d_obj, var_list = self.disc.get_variables())
        
        # -- for figure generation -------
        self.gen_figs = self.gen.set_model(self.z, False, True)
        self.sigmoid_d = tf.nn.sigmoid(self.disc.set_model(self.true_input
                                       ,False, True))
        
    def training_gen(self, sess, z_list, z_list2):
        _, g_obj = sess.run([self.train_gen, self.g_obj],
                            feed_dict = {self.z: z_list,
                                         self.z2: z_list2})
        return g_obj
        
    def training_disc(self, sess, z_list, figs):
        _, d_obj = sess.run([self.train_disc, self.d_obj],
                            feed_dict = {
                                         self.z: z_list,
                                         self.true_input:figs})
        return d_obj
    
    def generate(self, sess, z):
        ret_ = sess.run(self.gen_figs,
                       feed_dict = {self.z: z})
        return ret_
    def get_disc_value(self, sess, inputs):
        ret = sess.run(self.sigmoid_d,
                       feed_dict = {self.true_input:inputs})
        return ret
if __name__ == u'__main__':
    model = Model(z_dim = 30)
    model.set_model()
    
