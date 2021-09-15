# step one
# canonical form and reduce number of data
# the output data_prepro/data_smiles.csv

import pandas as pd
import warnings
from rdkit import Chem

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def load_data():
    ### BindingDB_curation_20200627 data is exluded by github since the size is large
    df = pd.read_csv('data/BindingDB_curation_20200627.csv')
    df = df.rename(columns={'Ligand SMILES': 'compound', 'BindingDB Target Chain  Sequence': 'target'})
    df = df[0:15000]
    return df

def canonical_smiles(smiles, sanitize=True, throw_warning=False):
    new_smiles = []
    for sm in smiles:
        try:
            mol = Chem.MolFromSmiles(sm, sanitize=sanitize)
            new_smiles.append(Chem.MolToSmiles(mol))
        except:
            if throw_warning:
                warnings.warn(sm + ' can not be canonized: invalid '
                                   'SMILES string!', UserWarning)
    return new_smiles

if __name__ == '__main__':
    df = load_data()
    # remove if target has a numberic value
    for tgt in df['target']:
        if (hasNumbers(tgt) == True):
            df = df[df['target'] != tgt]

    # target to all uppercase
    df['target'] = df['target'].str.upper()
    df = df.reset_index(drop=True)
    print(df)

    sanitized = canonical_smiles(df['compound'].values, sanitize=False, throw_warning=False)
    df2 = pd.DataFrame(sanitized, columns=['canon_smiles'])
    df2 = df2.reset_index(drop=True)

    df_total = pd.concat([df, df2], axis=1)
    df_total.to_csv("data_prepro/data_smiles.csv",index=False)
