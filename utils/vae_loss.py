import keras
from keras import backend as K
from keras import objectives

class calc_output_with_los(keras.layers.Layer):
    def vae_loss(self, x, z_decoded, z_mean, z_log_sigma):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        reconstruction_loss = keras.losses.binary_crossentropy(x, z_decoded)
        reconstruction_loss *= 784
        kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        loss_final = vae_loss

        return loss_final

    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        z_mean = inputs[2]
        z_log_sigma = inputs[3]
        loss = self.vae_loss(x, z_decoded, z_mean, z_log_sigma)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return K.mean(loss)