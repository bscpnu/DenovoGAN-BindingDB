import pandas as pd
import numpy as np
from utils import converter_direct_follow_tgt as tgtcov
import keras
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Conv2DTranspose
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU, ReLU, Lambda
from keras import backend as K
from tensorflow.keras import layers
import vae_loss

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], 80),
                              mean=0., stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon

class CustomSaver(keras.callbacks.Callback):
    def __init__(self, encoder, decoder):
        """ Save params in constructor
        """
        self.encoder = encoder
        self.decoder = decoder

    def on_epoch_end(self, epoch, logs={}):
        if epoch % 20 == 0:  # or save after some epoch, each k-th epoch etc.
            self.encoder.save("encoder_tgt_vae/encoder_{}.h5".format(epoch))
            self.decoder.save("decoder_tgt_vae/decoder_{}.h5".format(epoch))

def load_data():
    df1 = pd.read_csv('data/semi_atom2.csv')
    df2 = pd.read_csv('data/data_smiles.csv')
    df2 = df2[0:15000]
    df = pd.concat([df1['0'], df2['target'], df2['canon_smiles']], axis=1)
    df = df.dropna()
    return df


def define_generator(compound_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    latent_size = 80
    # image input
    in_target = Input(shape=compound_shape)
    # encoder model
    g = Conv2D(512, kernel_size=(3, 3), strides=2, padding="valid")(in_target)
    g = BatchNormalization()(g)
    g = ReLU()(g)
    g = Conv2D(256, kernel_size=(3, 3), strides=2, padding="valid", kernel_initializer=init)(g)
    g = BatchNormalization()(g)
    g = ReLU()(g)
    g = Conv2D(128, kernel_size=(2, 2), strides=2, padding="valid", kernel_initializer=init)(g)
    g = BatchNormalization()(g)
    g = ReLU()(g)
    g = Conv2D(64, kernel_size=(3, 3), strides=2, padding="same", kernel_initializer=init)(g)
    g = BatchNormalization()(g)
    g = ReLU()(g)
    shape = K.int_shape(g)
    g = Flatten()(g)
    g = BatchNormalization()(g)
    x = Dense(latent_size, name='latent_vector', activation='sigmoid')(g)

    z_mean = layers.Dense(latent_size, name="z_mean")(x)
    z_log_var = layers.Dense(latent_size, name="z_log_var")(x)
    z = layers.Lambda(sampling)([z_mean, z_log_var])
    encoder = keras.Model(in_target, [z_mean, z_log_var, z], name="encoder")
    print("encoder ---------------------------")
    print(encoder.summary())

    latent_inputs = Input(shape=(latent_size,), name='decoder_input')
    x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    x = Conv2DTranspose(64, kernel_size=(2, 2), strides=2, padding="valid", kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2DTranspose(128, kernel_size=(2, 2), strides=2, padding="valid", kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2DTranspose(256, kernel_size=(3, 3), strides=2, padding="valid", kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2DTranspose(512, kernel_size=(3, 3), strides=2, padding="valid", kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2DTranspose(1, kernel_size=(4, 4), strides=1, padding="valid", kernel_initializer=init)(x)
    out = Activation('sigmoid')(x)
    decoder = Model(latent_inputs, out)
    print("decoder ------------------------")
    z_decoded = decoder(z)

    # calculate the cooperative loss
    outputs = vae_loss.calc_output_with_los()([in_target, z_decoded, z_mean, z_log_var])

    # initiate super-encoder model
    autoencoder = Model(in_target, outputs)
    opt = Adam()
    autoencoder.compile(loss=None, optimizer=opt)

    return autoencoder, encoder, decoder

if __name__ == '__main__':
    df = load_data()
    print(df)

    # target data
    ls_target = []
    for tgt in df['target']:
        mat = tgtcov.target_encoder(tgt)
        ls_target.append(mat)
        # dec = tgtcov.target_decoder(mat)
    ls_target = np.array(ls_target)
    ls_target = ls_target.reshape((ls_target.shape[0], ls_target.shape[1], ls_target.shape[2], 1))
    ls_target = ls_target.astype('float32')
    ls_target /= np.amax(ls_target)

    print("target shape = ", ls_target.shape)

    target_shape = ls_target.shape[1:]

    g_model, encoder, decoder = define_generator(target_shape)
    # checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', verbose=1, monitor='val_loss',
    #                             save_best_only=True, mode='auto')
    encoder.load_weights('model_vae_target/encoder/encoder_180.h5')
    latent_pred = encoder.predict(ls_target)[2]

    df_latent = pd.DataFrame(latent_pred)
    df_latent.to_csv("data_prepro/target_latent.csv", index=False)






