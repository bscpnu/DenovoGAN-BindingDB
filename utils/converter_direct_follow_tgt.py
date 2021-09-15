import numpy as np

TARGET_CHARS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

targt2index = dict((c, i) for i, c in enumerate(TARGET_CHARS))
index2targt = dict((i, c) for i, c in enumerate(TARGET_CHARS))


# maxlen target = 1484 based on bindingDB dataset
def target_encoder(target):
    X = np.zeros((len(TARGET_CHARS), len(TARGET_CHARS)), dtype=np.uint16)
    for i, c in enumerate(target):
        #X[targt2index[c], targt2index[c+1]] += 1
        if i < len(target)-1:
            #print("current : ", c)
            #print("next : ", target[i+1])
            X[targt2index[c], targt2index[target[i+1]]] += 1
    return X


def target_decoder(X):
    target = ''
    X = X.argmax(axis=-1)
