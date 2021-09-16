import numpy as np
import pandas as pd
from utils import convert_to_oh as smiconv
import keras
from keras.optimizers import Adam
from utils import converter_direct_follow_tgt as tgtcov
from keras.initializers import RandomNormal
from sklearn.preprocessing import MinMaxScaler
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
from keras.callbacks import ModelCheckpoint

# custom model saver per given epoch
class CustomSaver(keras.callbacks.Callback):
    def __init__(self, encoder, decoder):
        """ Save params in constructor
        """
        self.encoder = encoder
        self.decoder = decoder

    def on_epoch_end(self, epoch, logs={}):
        if epoch % 20 == 0:  # or save after some epoch, each k-th epoch etc.
            self.encoder.save("model_regressor_encoder/encoder_{}.h5".format(epoch))
            self.decoder.save("model_regressor_decoder/decoder_{}.h5".format(epoch))

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def load_data():
    df = pd.read_csv('NewBinding.csv')
    return df

def define_reg_compound(compound_shape, target_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_compound = Input(shape=compound_shape)
    # encoder model
    g1 = Conv2D(512, kernel_size=(3, 3), strides=2, padding="valid",activation='relu')(in_compound)
    g1 = BatchNormalization()(g1)
    g1 = ReLU()(g1)
    g1 = Conv2D(256, kernel_size=(3, 3), strides=2, padding="valid", activation='relu')(g1)
    g1 = BatchNormalization()(g1)
    g1 = ReLU()(g1)
    g1 = Conv2D(128, kernel_size=(2, 2), strides=2, padding="valid", activation='relu')(g1)
    g1 = BatchNormalization()(g1)
    g1 = ReLU()(g1)
    g1 = Conv2D(64, kernel_size=(3, 3), strides=2, padding="same", activation='relu')(g1)
    g1 = ReLU()(g1)
    g1 = Flatten()(g1)
    g1 = BatchNormalization()(g1)
    out1 = Dense(50, name='latent_vector1', activation='relu')(g1)

    in_target = Input(shape=target_shape)
    # encoder model
    g2 = Conv2D(512, kernel_size=(3, 3), strides=2, padding="valid",activation='relu')(in_target)
    g2 = BatchNormalization()(g2)
    g2 = ReLU()(g2)
    g2 = Conv2D(256, kernel_size=(3, 3), strides=2, padding="valid", activation='relu')(g2)
    g2 = BatchNormalization()(g2)
    g2 = ReLU()(g2)
    g2 = Conv2D(128, kernel_size=(2, 2), strides=2, padding="valid", activation='relu')(g2)
    g2 = BatchNormalization()(g2)
    g2 = ReLU()(g2)
    g2 = Conv2D(64, kernel_size=(3, 3), strides=2, padding="same", activation='relu')(g2)
    g2 = BatchNormalization()(g2)
    g2 = ReLU()(g2)
    g2 = Flatten()(g2)
    #g2 = BatchNormalization()(g2)
    out2 = Dense(50, name='latent_vector2', activation='relu')(g2)

    combined = Concatenate()([out1, out2])
    g = Dense(256, activation='relu')(combined)
    g = Dense(128, activation='relu')(g)
    g = Dense(64, activation='relu')(g)
    out = Dense(1, activation='sigmoid')(g)

    reg = Model([in_compound, in_target], out)
    opt = Adam()
    reg.compile(optimizer=opt, loss=['mse'], metrics=['mse'])

    return reg

if __name__ == '__main__':

    df = load_data()

    ls_compound = []
    for smi in df['compound']:
        mat = smiconv.smiles_encoder(smi)
        ls_compound.append(mat)
        # dec = smiconv.smiles_decoder(mat)
    ls_compound = np.array(ls_compound)

    ls_compound = ls_compound.reshape((ls_compound.shape[0], ls_compound.shape[1], ls_compound.shape[2], 1))
    print("compound shape = ", ls_compound.shape)


    compound_shape = ls_compound.shape[1:]

    # remove if target has a numberic value
    for tgt in df['target']:
        if (hasNumbers(tgt) == True):
            df = df[df['target'] != tgt]

    # target to all uppercase
    df['target'] = df['target'].str.upper()

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

    ls_ic50 = np.array(df['IC50']).reshape(-1, 1)
    scaler = MinMaxScaler().fit(ls_ic50)
    binding_df = scaler.transform(ls_ic50)

    reg = define_reg_compound(compound_shape, target_shape)

    checkpoint = ModelCheckpoint('model_regressor-{epoch:03d}.h5', verbose=1, monitor='val_loss',
                                 save_best_only=True, mode='auto')

    reg.fit([ls_compound, ls_target], binding_df, epochs=200, batch_size=50, verbose=1, validation_split=0.3,callbacks=[checkpoint])