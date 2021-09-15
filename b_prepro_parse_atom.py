# step two
# parsing atom
# the output data_prepro/onehot_atom.h5

import pandas as pd
import numpy as np
from utils import zinc_grammar
from utils import molecule_vae
import nltk
import h5py

MAX_LEN=282
NCHARS = len(zinc_grammar.GCFG.productions())

def load_data():
    df = pd.read_csv('data_prepro/data_smiles.csv')
    return df

def to_one_hot(smiles):
    """ Encode a list of smiles strings to one-hot vectors """
    assert type(smiles) == list
    prod_map = {}
    for ix, prod in enumerate(zinc_grammar.GCFG.productions()):
        prod_map[prod] = ix
    tokenize = molecule_vae.get_zinc_tokenizer(zinc_grammar.GCFG)
    tokens = map(tokenize, smiles)
    parser = nltk.ChartParser(zinc_grammar.GCFG)

    parse_trees = []
    for t in tokens:
        while True:
            try:
                parse_trees.append(next(parser.parse(t)))
                #print(next(parser.parse(t)))
            except StopIteration:
                # one of the iterables has no more left.
                break
            break
    productions_seq = [tree.productions() for tree in parse_trees]
    #print("productions_seq = ", productions_seq)
    indices = [np.array([prod_map[prod] for prod in entry], dtype=np.int16) for entry in productions_seq]
    #print("length of index = ", len(indices))
    one_hot = np.zeros((len(indices), MAX_LEN, NCHARS), dtype=np.float16)
    for i in range(len(indices)):
        num_productions = len(indices[i])
        print("num_prod = ", num_productions)
        one_hot[i][np.arange(num_productions),indices[i]] = 1.
        one_hot[i][np.arange(num_productions, MAX_LEN),-1] = 1.
    return one_hot

if __name__ == '__main__':
    df = load_data()
    df = df[0:15000]
    print(df)

    L = df['canon_smiles'].values
    L = np.array(L)
    list_comp = L.tolist()

    print(type(L))

    OH = np.zeros((len(list_comp), MAX_LEN, NCHARS))
    onehot = to_one_hot(list_comp[0:len(list_comp)])
    print(onehot)
    print(onehot.shape)

    h5f = h5py.File('data_prepro/onehot_atom.h5', 'w')
    h5f.create_dataset('data', data=onehot)
    h5f.close()