#step four

import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv('data_prepro/data_atom_char.csv')
    return df


def parse_semi_atom(seq):
    tot_seq = pd.Series(seq)
    list_br = []
    list_ro = []
    for i in range(tot_seq.size):
        if str(tot_seq.iloc[i]).strip() == "'" + '[' + "'":
            br_char = []
            br_char.append(str(tot_seq.iloc[i]).replace("'", "").strip())
            for j, value in tot_seq[i + 1:].items():
                if str(value).strip() == "'" + ']' + "'":
                    br_char.append(str(value).replace("'", "").strip())
                    list_br.append(''.join(br_char))
                    break
                else:
                    br_char.append(str(value).replace("'", "").strip())
        if str(tot_seq.iloc[i]).strip() == "'" + '(' + "'":
            sum = 0
            ro_char = []
            ro_char.append(str(tot_seq.iloc[i]).replace("'", "").strip())
            for j, value in tot_seq[i + 1:].items():
                if str(value).strip() == "'" + '(' + "'":
                    # new_seq.append(''.join(str(value).replace("'", "").strip()))
                    break
                elif str(value).strip() == "'" + ')' + "'" and sum <= 3:
                    ro_char.append(str(value).replace("'", "").strip())
                    list_ro.append(''.join(ro_char))
                    break
                elif sum > 3:
                    break
                else:
                    ro_char.append(str(value).replace("'", "").strip())
                    sum = sum + 1
        if None == tot_seq.iloc[i]:
            break

    return list_br, list_ro


def prase_semi_atom(seq):
    tot_seq = pd.Series(seq)
    new_seq = []
    i = 0
    foundIt = False
    while i < tot_seq.size:
        if str(tot_seq.iloc[i]).strip() == "'" + '[' + "'":
            br_char = []
            br_char.append(str(tot_seq.iloc[i]).replace("'", "").strip())
            for value in tot_seq[i + 1:]:
                if str(value).strip() == "'" + ']' + "'":
                    # print("masuk -1 = ", str(value))
                    br_char.append(str(value).replace("'", "").strip())
                    new_seq.append(''.join(br_char))
                    i = i + len(br_char)
                    foundIt = True
                    break
                else:
                    # print("masuk -2 = ", str(value))
                    br_char.append(str(value).replace("'", "").strip())
                    # idx = idx + 1
        elif str(tot_seq.iloc[i]).strip() == "'" + '(' + "'":
            sum = 0
            ro_char = []
            # print("masuk1 = ", (tot_seq.iloc[i]).strip())
            ro_char.append(str(tot_seq.iloc[i]).replace("'", "").strip())
            # idx = idx + 1
            for value in tot_seq[i + 1:]:
                if str(value).strip() == "'" + ')' + "'" and sum <= 3:
                    ro_char.append(str(value).replace("'", "").strip())
                    new_seq.append(''.join(ro_char))
                    i = i + len(ro_char)
                    foundIt = True
                    break
                elif str(value).strip() == "'" + '(' + "'":
                    new_seq.append(str(value).replace("'", "").strip())
                    foundIt = True
                    i = i + 1
                    break
                elif sum > 3:
                    new_seq.append(str("'" + '(' + "'").replace("'", "").strip())
                    foundIt = True
                    i = i + 1
                    break
                else:
                    ro_char.append(str(value).replace("'", "").strip())
                    sum = sum + 1
        elif None == tot_seq.iloc[i]:
            break
        else:
            new_seq.append(str(tot_seq.iloc[i]).replace("'", "").strip())
            i += 1

    return new_seq


def unique(list1):
    x = np.array(list1)
    return np.unique(x)

def paranthesis_check(seq):
    tot_seq = pd.Series(seq)
    bkurung = []
    tkurung = []
    for i in range(tot_seq.size):
        if str(tot_seq.iloc[i]).strip() == "'" + '(' + "'":
            bkurung.append(i)
        if str(tot_seq.iloc[i]).strip() == "'" + ')' + "'":
            tkurung.append(i)
    if len(bkurung) == 0:
        bkurung.append(0)
        tkurung.append(0)
    for x in bkurung:
        if len(bkurung) < 4:
            bkurung.append(0)
        else:
            continue
    for y in tkurung:
        if len(tkurung) < 4:
            tkurung.append(0)
        else:
            continue
    converted_bkurung2 = ','.join(str(s) for s in bkurung)
    converted_tkurung2 = ','.join(str(s) for s in tkurung)

    return np.asarray(converted_bkurung2), np.asarray(converted_tkurung2)


if __name__ == '__main__':
    df = load_data()
    print(df)
    s = pd.Series(df["0"])
    s = s.str.split(",", expand=True)
    print(s)
    df2 = pd.DataFrame(s)
    semi_atom = []
    list_bracket = []
    list_chain = []

    list_bkurung = []
    list_tkurung = []


    for i in range(len(df2)):
        bracket, chain = parse_semi_atom(df2.iloc[i].values)
        print(df2.iloc[i].values)
        semi_atom.append(str(prase_semi_atom(df2.iloc[i].values))[1:-1].strip('"').replace(r'\\\\','\\\\'))
        list_bracket.append(bracket)
        list_chain.append(chain)

    np_semiatom = np.array(semi_atom)

    import pandas as pd

    df_atom = pd.DataFrame(np_semiatom)
    #print(df)
    # print(df.shape)
    df_atom.to_csv("semi_atom2.csv", index=False)
    print("df_atom")
    print(df_atom)
    serie = pd.Series(np_semiatom)
    print("serie")
    print(serie)
    serie = serie.str.split(",", expand=True)
    df_serie = pd.DataFrame(serie)

    for i in range(len(df_serie)):
        bkurung, tkurung = paranthesis_check(df_serie.iloc[i].values)
        list_bkurung.append(str(bkurung))
        list_tkurung.append(str(tkurung))

    k_bkurung = np.array(list_bkurung)
    df_bkurung = pd.DataFrame(np.array(list_bkurung))
    df_bkurung.to_csv("data_prepro/open_bracket.csv", index=False)

    df_tkurung = pd.DataFrame(np.array(list_tkurung))
    df_tkurung.to_csv("data_prepro/closed_bracket.csv", index=False)

    list_bracket2 = [x for x in list_bracket if x != []]
    list_chain2 = [x for x in list_chain if x != []]
    # print(df2.iloc[i])

    list_bracket3 = [val for sublist in list_bracket2 for val in sublist]
    list_chain3 = [val for sublist in list_chain2 for val in sublist]

    print("bracket = ", unique(list_bracket3))
    print("bracket = ", unique(list_chain3))

    bracket_atom = np.array(unique(list_bracket3))

    df_bracket_atom = pd.DataFrame(bracket_atom)
    df_bracket_atom.to_csv("data_prepro/bracket_atom.csv", index=False)
    rounded_atom = np.array(unique(list_chain3))
    df_rounded_atom = pd.DataFrame(rounded_atom)
    df_rounded_atom.to_csv("data_prepro/rounded_atom.csv", index=False)
