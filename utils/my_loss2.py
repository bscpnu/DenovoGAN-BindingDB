import keras
from keras import backend as K
from keras import objectives
import numpy as np
import tensorflow as tf

class calc_output_with_los(keras.layers.Layer):

    def vae_loss_full(self, x_orig, x_pred,db, dt, disc_in,disc_out, in_tgt, out_tgt):
        x_ori = x_orig
        x_pre = x_pred

        x_orig = K.flatten(x_orig)
        x_pred = K.flatten(x_pred)

        recon_loss = objectives.mse(x_orig, x_pred)

        x_okurung = tf.gather_nd(x_ori, K.cast(db,"int32"))
        xhat_okurung = tf.gather_nd(x_pre, K.cast(db,"int32"))

        x_tkurung = tf.gather_nd(x_ori, K.cast(dt,"int32"))
        xhat_tkurung = tf.gather_nd(x_pre, K.cast(dt,"int32"))

        bracket_loss = K.mean(K.abs(x_okurung-xhat_okurung) - K.abs(x_tkurung-xhat_tkurung))

        disc_in = K.flatten(disc_in)
        disc_out = K.flatten(disc_out)
        disc_loss = K.binary_crossentropy(disc_in, disc_out)

        in_tgt = K.flatten(in_tgt)
        out_tgt = K.flatten(out_tgt)
        err_latent = objectives.mse(in_tgt, out_tgt)

        return 100*recon_loss + bracket_loss + disc_loss + err_latent

    def call(self, inputs):
        x_orig = inputs[0]
        x_pred = inputs[1]
        db = inputs[2]
        dt = inputs[3]
        disc_in = inputs[4]
        disc_out = inputs[5]
        in_tgt = inputs[6]
        out_tgt = inputs[7]


        loss = self.vae_loss_full(x_orig, x_pred,db, dt, disc_in,disc_out, in_tgt, out_tgt)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x_orig