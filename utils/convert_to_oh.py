import numpy as np

SMILES_CHARS = [' ', '#', '(', '(=N)', '(=O)', '(=S)', '(Br)', '(C#N)', '(C)',
                '(C1)', '(C2)', '(C3)', '(C4)', '(C5)', '(C6)', '(C=O)',
                '(CBr)', '(CC)', '(CC2)', '(CC3)', '(CCC)', '(CCCl)',
                '(CCN)', '(CCO)', '(CCl)', '(CF)', '(CN)', '(CO)', '(COC)',
                '(CS)', '(CSC)', '(Cl)', '(F)', '(I)', '(N)', '(N=O)',
                '(NC)', '(O)', '(OC)', '(OCC)', '(SC)', '(c1)', '(c13)',
                '(c1Cl)', '(c2)', '(c21)', '(c23)', '(c3)', '(c32)',
                '(c35)', '(c4)', '(c5)', '(cc2)', '(cn1)', '(cn2)',
                '(cn3)', '(n1)', '(n2)', '(o2)', '(s1)', '(s2)', ')',
                '-', '/', '1', '2', '3', '4', '5', '6', '7', '=',
                'B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S',
                '[C+]', '[C@@H]', '[C@@]', '[C@H]', '[C@]', '[N+]', '[N-]',
                '[N@@+]', '[NH2+]', '[NH3+]', '[O+]', '[O-]', '[OH+]',
                '[Se]', '[c-]', '[n+]', '[n-]', '[nH+]', '[nH]', '[o+]',
                '\\\\', 'c', 'n', 'o', 's', 'Br', 'C', 'Cl', 'F', 'I', 'N',
                'O', 'S', '[C-]', '[N-]', '[NH3+]', '[O-]', 'c']

smi2index = dict((c, i) for i, c in enumerate(SMILES_CHARS))
index2smi = dict((i, c) for i, c in enumerate(SMILES_CHARS))


# maxlen=100 based on bindingDB dataset
def smiles_encoder(smiles, maxlen=100):
    # smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    X = np.zeros((maxlen, len(SMILES_CHARS)), dtype=np.uint16)
    for i, c in enumerate(smiles):
        X[i, smi2index[str(c).replace("'", "").strip()]] = 1
    return X


def smiles_decoder(X):
    smi = ''
    X = X.argmax(axis=-1)
    for i in X:
        smi += index2smi[i]
    return smi
