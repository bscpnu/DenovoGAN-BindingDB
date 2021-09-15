import numpy as np
import pandas as pd
from utils import convert_to_oh as smiconv
import keras
from keras.models import load_model
from keras.optimizers import Adam, RMSprop
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Conv2DTranspose
from keras.layers import Flatten
from keras.layers import Activation
from numpy import zeros
from numpy import ones
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU, ReLU, Lambda
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from utils import converter_direct_follow_tgt as tgtcov
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from utils import my_loss2

def load_data():
    df1 = pd.read_csv('data_prepro/semi_atom.csv')
    df1 = df1[0:10000]
    #df1 = df1[10000:15000]
    df2 = pd.read_csv('data_prepro/data_smiles.csv')
    df2 = df2[0:10000]
    #df2 = df2[10000:15000]
    df3 = pd.read_csv('data_prepro/open_bracket2.csv')
    df3 = df3[0:10000]
    #df3 = df3[10000:15000]
    df4 = pd.read_csv('data_prepro/close_bracket2.csv')
    #df4 = df4[10000:15000]
    df4 = df4[0:10000]
    df5 = pd.read_csv('data_prepro/compound_latent.csv')
    #df5 = df5[10000:15000]
    df5 = df5[0:10000]
    df6 = pd.read_csv('data_prepro/target_latent.csv')
    df6 = df6[0:10000]
    #df6 = df6[10000:15000]
    print("df3 = ", df3.count())
    print("df4 = ", df4.count())
    df = pd.concat([df1['0'], df2['target'], df2['canon_smiles']], axis=1)
    return df, df3, df4, df5, df6


def discriminator(compound_shape):
    init = RandomNormal(stddev=0.02)
    in_compound = Input(shape=compound_shape)
    # encoder model
    g = Conv2D(512, kernel_size=(3, 3), strides=2, padding="valid")(in_compound)
    g = BatchNormalization()(g)
    g = ReLU()(g)
    g = Conv2D(512, kernel_size=(3, 3), strides=2, padding="valid", kernel_initializer=init)(g)
    g = BatchNormalization()(g)
    g = ReLU()(g)
    g = Conv2D(256, kernel_size=(3, 3), strides=2, padding="valid", kernel_initializer=init)(g)
    g = BatchNormalization()(g)
    g = ReLU()(g)
    g = Conv2D(128, kernel_size=(2, 2), strides=2, padding="valid", kernel_initializer=init)(g)
    g = BatchNormalization()(g)
    g = ReLU()(g)
    g = Conv2D(64, kernel_size=(3, 3), strides=2, padding="same", kernel_initializer=init)(g)
    g = ReLU()(g)
    g = Flatten()(g)
    g = BatchNormalization()(g)
    out = Dense(4, name='latent_vector', activation='sigmoid')(g)
    disc = Model(in_compound, out)
    opt = RMSprop()
    disc.compile(loss=['binary_crossentropy'], optimizer=opt, loss_weights=[0.5])
    return disc


def ae_compound():
    # weight initialization
    init = RandomNormal(stddev=0.02)
    latent_size = 100

    latent_inputs = Input(shape=(latent_size,), name='decoder_input')
    x = Dense(3 * 4 * 64)(latent_inputs)
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

    # return model
    gen = Model(latent_inputs, out)
    print(gen.summary())
    opt = Adam()
    gen.compile(loss='mse', optimizer=opt)

    return gen

def encoder(compound_shape):
    init = RandomNormal(stddev=0.02)
    latent_size = 100
    # image input
    in_target = Input(shape=compound_shape)
    # encoder model
    g = Conv2D(512, kernel_size=(3, 3), strides=2, padding="valid")(in_target)
    g = BatchNormalization()(g)
    g = ReLU()(g)
    g = Conv2D(512, kernel_size=(3, 3), strides=2, padding="valid", kernel_initializer=init)(g)
    g = BatchNormalization()(g)
    g = ReLU()(g)
    g = Conv2D(256, kernel_size=(3, 3), strides=2, padding="valid", kernel_initializer=init)(g)
    g = BatchNormalization()(g)
    g = ReLU()(g)
    g = Conv2D(128, kernel_size=(2, 2), strides=2, padding="valid", kernel_initializer=init)(g)
    g = BatchNormalization()(g)
    g = ReLU()(g)
    g = Conv2D(64, kernel_size=(3, 3), strides=2, padding="same", kernel_initializer=init)(g)
    g = ReLU()(g)
    g = Flatten()(g)
    g = BatchNormalization()(g)
    z = Dense(latent_size, activation='sigmoid')(g)
    encoder = Model(in_target, z, name='encoder')
    opt = Adam()
    encoder.compile(optimizer=opt,loss='mse')
    return encoder

def ae_combine(latent_shape, gen, disc, enc, compound_shape):
    # in_compound = Input(shape=compound_shape)
    # make weights in the discriminator not trainable
    for layer in disc.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False

    in_target = Input(shape=latent_shape)
    in_comp = Input(shape=compound_shape)
    in_disc = Input(shape=(4,))
    in_b = Input(shape=(1,))
    in_t = Input(shape=(1,))

    gen_out = gen(in_target)
    gen_out2 = Reshape([gen_out.shape[1], gen_out.shape[2]])(gen_out)
#
    print("gen_out.shape = " ,gen_out.shape)
    enc_out = enc(gen_out)
    d_out = disc(gen_out)
    outputs = my_loss2.calc_output_with_los()([in_comp, gen_out2, in_b, in_t, in_disc, d_out, in_target, enc_out])
    ae_model = Model([in_target, in_comp,in_disc, in_b, in_t], outputs)
    opt = Adam()
    ae_model.compile(loss=None, optimizer=opt)

    return ae_model


if __name__ == '__main__':
    n_epochs = 200
    batch_size = 50
    df, bkurung, tkurung, df_comp, df_tgt = load_data()

    list_b = []
    for index, row in bkurung.iterrows():
        data = np.stack((list([row['0'], 2]), list([row['1'], 2]), list([row['2'], 2]), list([row['2'], 2])))
        list_b.append(data)
    list_b2 = np.array(list_b, dtype=int)
    list_t = []

    for index, row in tkurung.iterrows():
        data = np.stack((list([row['0'], 2]), list([row['1'], 2]), list([row['2'], 2]), list([row['2'], 2])))
        list_t.append(data)
    list_t2 = np.array(list_t, dtype=int)
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
    compound_shape2 = ls_compound.shape[1:-1]

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
    latent_shape = (100,)

    # ae1, encoder1, decoder1 = ae_target(target_shape)
    gen = ae_compound()
    disc = discriminator(compound_shape)
    enc = encoder(compound_shape)
    gan = ae_combine(latent_shape, gen, disc, enc, compound_shape2)

    latent_tgt = df_tgt.values
    #latent_comp = df_comp.values
    latent_comp = np.random.uniform(low=-1.215913, high=1.2117409, size=(len(latent_tgt),20))
    print("latent comp sahep = ", latent_comp.shape)

    latent_vec = np.concatenate([latent_tgt, latent_comp], axis=1)
    print("latent vec = ", latent_vec.shape)

    gan_mod = load_model('gan_model/gan_20.h5', compile=False)

    out_gan = gan_mod.predict(latent_vec)

    pred_comp_oh = np.where(out_gan > 0.5, 1, 0)

    # target data
    ls_compound = ls_compound.reshape(ls_compound.shape[0], ls_compound.shape[1], ls_compound.shape[2])
    actual_comp = []
    for comp in ls_compound:
        dec = smiconv.smiles_decoder(comp)
        actual_comp.append(dec)
    actual_compound = np.array(actual_comp)

    df_act = pd.DataFrame(actual_compound, columns=['actual'])
    # target data
    pred_compound = []

    pred_list = pred_comp_oh.reshape(pred_comp_oh.shape[0], pred_comp_oh.shape[1], pred_comp_oh.shape[2])
    for comp in pred_list:
        dec = smiconv.smiles_decoder(comp)
        pred_compound.append(dec)
    pred_compound = np.array(pred_compound)

    df_pred = pd.DataFrame(pred_compound, columns=['predicted'])

    df_save = pd.concat([df_pred, df_act], axis=1)
    df_save.to_csv("gan_result.csv")

    print("mse = ", mean_squared_error(ls_compound.flatten(), pred_list.flatten()))