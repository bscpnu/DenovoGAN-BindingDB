#step three
# convert compound to char

from __future__ import print_function
from utils import zinc_grammar
import h5py

import nltk
import numpy as np

uni_atom = []

def get_zinc_tokenizer(cfg):
    long_tokens = filter(lambda a: len(a) > 1, cfg._lexical_index.keys())
    replacements = ['$', '%', '^', '&', 'A', 'D', 'E', 'G', 'J', 'M', 'Q', 'U', 'W', 'X', 'Y', 'f', 'h', 'j']
    assert len(list(long_tokens)) == len(replacements)
    for token in replacements:
        assert not cfg._lexical_index.__contains__(token)
    def tokenize(smiles):
        long_tokens = filter(lambda a: len(a) > 1, cfg._lexical_index.keys())
        for i, token in enumerate(long_tokens):
            smiles = smiles.replace(token, replacements[i])
        tokens = []
        for token in smiles:
            if token in list(replacements):
                ix = replacements.index(token)
                long_tokens2 = filter(lambda a: len(a) > 1, cfg._lexical_index.keys())
                tokens.append(list(long_tokens2)[ix])
            else:
                tokens.append(token)
        return tokens

    return tokenize


def pop_or_nothing(S):
    try:
        return S.pop()
    except:
        return 'Nothing'

#CC(C)(C)NC(=O)[C@@H]1C[C@@H](CCN1C[C@@H](O)[C@H](Cc1ccccc1)NC(=O)O[C@H]1CCOC1)OCc1ccncc1

def prods_to_eq(prods):
    #print("prods = ", prods)
    seq = [prods[0].lhs()]
    for prod in prods:
        if str(prod.lhs()) == 'Nothing':
            break
        for ix, s in enumerate(seq):
            if s == prod.lhs():
                seq = seq[:ix] + list(prod.rhs()) + seq[ix + 1:]
                break
    try:
        return seq
    except:
        return ''


class ZincGrammarModel():

    def __init__(self, latent_rep_size=56):
        """ Load the (trained) zinc encoder/decoder, grammar model. """
        self._grammar = zinc_grammar
        #self.MAX_LEN = self._model.MAX_LEN
        self._productions = self._grammar.GCFG.productions()
        self._prod_map = {}
        for ix, prod in enumerate(self._productions):
            self._prod_map[prod] = ix
        self._parser = nltk.ChartParser(self._grammar.GCFG)
        self._tokenize = get_zinc_tokenizer(self._grammar.GCFG)
        self._n_chars = len(self._productions)
        self._lhs_map = {}
        for ix, lhs in enumerate(self._grammar.lhs_list):
            self._lhs_map[lhs] = ix

    def decode(self, unmasked):
        # Convert from one-hot to sequence of production rules
        prod_seq = [[self._productions[unmasked[index, t].argmax()]
                     for t in range(unmasked.shape[1])]
                    for index in range(unmasked.shape[0])]
        return [prods_to_eq(prods) for prods in prod_seq]

h5f = h5py.File('data_prepro/onehot_atom.h5', 'r')
data = h5f['data'][:]
h5f.close()

#data = data[145:149]

grammar_model = ZincGrammarModel()

prepro = []
for oh in data:
    oh2 = oh.reshape(1, oh.shape[0], oh.shape[1])
    for mol in grammar_model.decode(oh2):
        #print(np.array(mol))
        prepro.append(str(mol)[1:-1])

np_prepro = np.array(prepro)
#print(np_prepro)
#print(np_prepro.shape)

import pandas as pd
df = pd.DataFrame(np_prepro)
print(df)
#print(df.shape)
df.to_csv("data_prepro/data_atom_char.csv", index=False)