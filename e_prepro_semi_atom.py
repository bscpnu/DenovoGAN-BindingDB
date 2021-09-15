#step five

import pandas as pd
import numpy as np

def load_data():
    df1 = pd.read_csv('data_prepro/list_obracket.csv')
    df2 = pd.read_csv('data_prepro/list_cbracket.csv')
    return df1, df2

if __name__ == '__main__':
    df1, df2 = load_data()

    #df = df.replace(to_replace=r'\\\\', value='\\\\')

    s1 = pd.Series(df1["0"])
    s1 = s1.str.split(",", expand=True)


    dfs = pd.DataFrame(s1)
    #dfs.to_csv('bkurung_df.csv', index=False)

    b0 = dfs[0].astype(int)
    b1 = dfs[1].astype(int)
    b2 = dfs[2].astype(int)
    b3 = dfs[3].astype(int)

    bkur = pd.concat([b0, b1, b2, b3], axis=1)
    d_bkur = pd.DataFrame(bkur)
    d_bkur.to_csv("data_prepro/open_bracket.csv", index=False)

    s2 = pd.Series(df2["0"])
    s2 = s2.str.split(",", expand=True)

    dfs2 = pd.DataFrame(s2)
    #dfs2.to_csv('tkurung_df.csv', index=False)

    t0 = dfs2[0].astype(int)
    t1 = dfs2[1].astype(int)
    t2 = dfs2[2].astype(int)
    t3 = dfs2[3].astype(int)

    tkur = pd.concat([t0, t1, t2, t3], axis=1)
    t_bkur = pd.DataFrame(tkur)
    t_bkur.to_csv("data_prepro/closed_bracket.csv", index=False)
