import numpy as np
import pandas as pd
from utils import convert_to_oh as smiconv
import keras
from keras.regularizers import l2
from keras.models import load_model
from keras.optimizers import Adam, RMSprop
from keras.initializers import RandomNormal
import math
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
from utils import my_loss2
from keras import backend as K

def load_data():
    df1 = pd.read_csv('data_prepro/semi_atom.csv')
    df1 = df1[0:10000]
    df2 = pd.read_csv('data_prepro/data_smiles.csv')
    df2 = df2[0:10000]
    df3 = pd.read_csv('data_prepro/open_bracket.csv')
    df3 = df3[0:10000]
    df4 = pd.read_csv('data_prepro/closed_bracket.csv')
    df4 = df4[0:10000]
    df5 = pd.read_csv('data_prepro/compound_latent.csv')
    df5 = df5[0:10000]
    df6 = pd.read_csv('data_prepro/target_latent.csv')
    df6 = df6[0:10000]
    print("df3 = ", df3.count())
    print("df4 = ",df4.count())
    df = pd.concat([df1['0'], df2['target'], df2['canon_smiles']], axis=1)
    return df, df3, df4, df5, df6

def discriminator(compound_shape):
    init = RandomNormal(stddev=0.02)
    in_compound = Input(shape=compound_shape)
    # encoder model
    g = Conv2D(512, kernel_size=(3, 3), strides=2, padding="valid")(in_compound)
    g = BatchNormalization()(g)
    g = LeakyReLU(alpha=0.025)(g)
    g = Conv2D(512, kernel_size=(3, 3), strides=2, padding="valid", kernel_initializer=init)(g)
    g = BatchNormalization()(g)
    g = LeakyReLU(alpha=0.025)(g)
    g = Conv2D(256, kernel_size=(3, 3), strides=2, padding="valid", kernel_initializer=init)(g)
    g = BatchNormalization()(g)
    g = LeakyReLU(alpha=0.025)(g)
    g = Conv2D(128, kernel_size=(2, 2), strides=2, padding="valid", kernel_initializer=init)(g)
    g = BatchNormalization()(g)
    g = LeakyReLU(alpha=0.025)(g)
    g = Conv2D(64, kernel_size=(3, 3), strides=2, padding="same", kernel_initializer=init)(g)
    g = ReLU()(g)
    g = Flatten()(g)
    g = BatchNormalization()(g)
    out = Dense(4, name='latent_vector', activation='sigmoid')(g)
    disc = Model(in_compound, out)
    opt = RMSprop()
    disc.compile(loss=['binary_crossentropy'], optimizer=opt, loss_weights=[0.5])
    return disc


def ae_compound(latent_size):
    # weight initialization
    init = RandomNormal(stddev=0.02)

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

def encoder(compound_shape, latent_size):
    init = RandomNormal(stddev=0.02)
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

    enc_out = enc(gen_out)
    print("enc_out.shape = ", enc_out.shape)
    print("in_target.shape = ", in_target.shape)
    d_out = disc(gen_out)
    outputs = my_loss2.calc_output_with_los()([in_comp, gen_out2, in_b, in_t, in_disc, d_out, in_target, enc_out])
    ae_model = Model([in_target, in_comp,in_disc, in_b, in_t], outputs)
    opt = Adam()
    ae_model.compile(loss=None, optimizer=opt)

    return ae_model

if __name__ == '__main__':
    n_epochs = 200
    batch_size = 50
    df, bkurung, tkurung, df_tgt, df_comp = load_data()

    list_b = []
    for index, row in bkurung.iterrows():
        data = np.stack((list([row['0'], 2]), list([row['1'], 2]), list([row['2'], 2]), list([row['2'], 2])))
        list_b.append(data)
    list_b2 = np.array(list_b, dtype=int)
    print("list_b2 = ", type(list_b2))
    list_t= []
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
    latent_size = 100

    # ae1, encoder1, decoder1 = ae_target(target_shape)
    gen = ae_compound(latent_size)
    disc = discriminator(compound_shape)
    enc = encoder(compound_shape, latent_size)
    gan = ae_combine(latent_shape, gen, disc, enc, compound_shape2)
    latent_tgt = df_tgt.values
    latent_comp = df_comp.values

    latent_vec = np.concatenate([latent_tgt, latent_comp], axis=1)
    print("latent vec = ", latent_vec.shape)

    bat_per_epo = int(len(ls_compound) / batch_size)
    n_steps = bat_per_epo * n_epochs
    j = 0
    a = 0
    for i in range(n_steps):
        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random batch of images
        idx = np.random.randint(0, ls_compound.shape[0], batch_size)
        compound_data = ls_compound[idx]
        compound_data2 = compound_data.reshape(compound_data.shape[0],compound_data.shape[1],compound_data.shape[2])
        target_data = ls_target[idx]
        latent_data = latent_vec[idx]
        db = list_b2[idx]
        dt = list_t2[idx]

        y_fake = zeros((len(target_data), 4))
        y_real = ones((len(target_data), 4))

        # Train the discriminator
        disc.trainable = True
        d_loss_real = disc.train_on_batch(compound_data, y_real)
        d_loss_fake = disc.train_on_batch(gen.predict(latent_data), y_fake)

        # d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------

        # Train the generator
        #g_loss = gan.train_on_batch([latent_data, compound_data, y_real], compound_data)
        g_loss = gan.train_on_batch([latent_data, compound_data2, y_real, db, dt], compound_data)

        print('>%d, d_loss_real[%.5f] d_loss_fake[%.5f] g_loss[%.5f]' % (a, d_loss_real, d_loss_fake, g_loss))
        # Plot the progress (every 10th epoch)

        if (i + 1) % (bat_per_epo * 10) == 0:
            epoch = j+1
            # calculate learning rate:
            #print("learning rate before = ",K.eval(gan.optimizer.lr))
            #current_learning_rate = calculate_learning_rate(epoch, K.eval(gan.optimizer.lr))
            #print("learning rate after = ", K.eval(gan.optimizer.lr))
            # train model:
            #K.set_value(gan.optimizer.lr, current_learning_rate)  # set new learning_rate
            # sample_images(latent_dim, decoder, epoch)
            # print("%d [D loss real: %f, D loss fake: %f] [G loss: %f]" % (

            gen.save("gan_model/gan_{}.h5".format(j + 1))
            j = j + 1
        a=a+1

