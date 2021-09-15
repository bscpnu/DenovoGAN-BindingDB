import pandas as pd
import numpy as np
from utils import convert_to_oh as smiconv
import keras
from utils import vae_loss
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from tensorflow.keras import layers
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

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], 20),
                              mean=0., stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon

def load_data():
    df1 = pd.read_csv('data_prepro/semi_atom.csv')
    df2 = pd.read_csv('data_prepro/data_smiles.csv')
    df2 = df2[0:15000]
    df = pd.concat([df1['0'], df2['target'], df2['canon_smiles']], axis=1)
    return df


def define_generator(compound_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    latent_size = 20
    # image input
    in_target = Input(shape=compound_shape)
    # encoder model
    g = Conv2D(512, kernel_size=(3,3), strides=2, padding="valid")(in_target)
    g = BatchNormalization()(g)
    g = Dropout(0.3)(g)
    g = LeakyReLU(alpha=0.025)(g)
    g = Conv2D(512, kernel_size=(3, 3), strides=2, padding="valid", kernel_initializer=init)(g)
    g = BatchNormalization()(g)
    g = Dropout(0.3)(g)
    g = LeakyReLU(alpha=0.025)(g)
    g = Conv2D(256, kernel_size=(3,3), strides=2, padding="valid", kernel_initializer=init)(g)
    g = BatchNormalization()(g)
    g = Dropout(0.3)(g)
    g = LeakyReLU(alpha=0.025)(g)
    g = Conv2D(128, kernel_size=(2,2), strides=2, padding="valid", kernel_initializer=init)(g)
    g = BatchNormalization()(g)
    g = Dropout(0.3)(g)
    g = LeakyReLU(alpha=0.025)(g)
    g = Conv2D(64, kernel_size=(3,3), strides=2, padding="same", kernel_initializer=init)(g)
    g = LeakyReLU(alpha=0.025)(g)
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
    x = Reshape((3, 4, 64))(x)
    x = Conv2DTranspose(64, kernel_size=(2, 2), strides=2, padding="same", kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.025)(x)
    x = Conv2DTranspose(128, kernel_size=(2, 5), strides=2, padding="same", kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.025)(x)
    x = Conv2DTranspose(256, kernel_size=(3, 6), strides=3, padding="valid", kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.025)(x)
    x = Conv2DTranspose(512, kernel_size=(5, 6), strides=2, padding="valid", kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2DTranspose(1, kernel_size=(26, 15), strides=1, padding="valid", kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    out = Activation('sigmoid')(x)
    decoder = Model(latent_inputs, out)
    print("decoder ------------------------")
    print(decoder.summary())
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

    s = pd.Series(df["0"])
    s = s.str.split(",", expand=True)
    df2 = pd.DataFrame(s)
    print(df2)

    # smiles data
    ls_compound = []
    for idx, smi in df2.iterrows():
        ss = smi.values
        smi2 = ss[ss != np.array(None)]
        mat = smiconv.smiles_encoder(smi2)
        ls_compound.append(mat)
        # dec = smiconv.smiles_decoder(mat)
    ls_compound = np.array(ls_compound)
    ls_compound = ls_compound.reshape((ls_compound.shape[0], ls_compound.shape[1], ls_compound.shape[2], 1))

    compound_shape = ls_compound.shape[1:]

    g_model, encoder, decoder = define_generator(compound_shape)

    encoder.load_weights('encoder_vae_comp/encoder/encoder_180.h5')
    latent_pred = encoder.predict(ls_compound)[2]

    df_latent = pd.DataFrame(latent_pred)
    df_latent.to_csv("data_prepro/compound_latent.csv", index=False)










