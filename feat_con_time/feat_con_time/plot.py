import pandas as pd
import pandas.io.formats.style as style
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(palette='pastel',color_codes = True)

def create_latex_table(sum_data:dict):
    """Creates a single latex table for one KG/dataset

    Args:
        sum_data (dict): _description_

    Returns:
        _type_: _description_
    """
    sorted_keys = sorted([x for x in sum_data.keys()], key=sum_data.get, reverse=True)
    df = pd.DataFrame.from_dict({'Stat':sorted_keys, 'Time (s)': [sum_data[x] for x in sorted_keys]})
    df_style = style.Styler(df, precision=2, caption = '', thousands=',', escape='latex')
    df_style = df_style.hide()
    txt = df_style.to_latex(hrules=True, label='fig:cu_feat_time')
    return txt


def create_latex_table_cum_data(sum_dct:dict):
    """Creates a single latex table for multiple KG/dataset. The input should be the output of get_time_data from util.py

    Args:
        sum_data (dict): _description_

    Returns:
        _type_: _description_
    """
    key_func = lambda x: max([sum_dct[i][x] for i in sum_dct.keys()])
    unsorted_keys = set()
    for k in sum_dct.keys():
        for s in sum_dct[k].keys():
            unsorted_keys.add(s)
        break
    sorted_keys = sorted([x for x in unsorted_keys], key=key_func, reverse=True)
    data = {'Stat':sorted_keys}
    for k in sum_dct.keys():
        data['%s Time (s)'%k] = []
        for s in sorted_keys:
            data['%s Time (s)'%k].append(sum_dct[k][s])
    
    df = pd.DataFrame.from_dict(data)
    df_style = style.Styler(df, precision=2, caption = '', thousands=',', escape='latex')
    df_style = df_style.hide()
    txt = df_style.to_latex(hrules=True, label='fig:cu_feat_time')
    return txt

def plot_pie(perc_data):
    """plot the percentage running time into a pie chart

    Args:
        perc_data (_type_): _description_
    """
    fig, ax = plt.subplots(figsize=(16,9))
    patches, text, text2 = ax.pie([x for x in perc_data[0].values()], autopct='%1.1f%%')
    patches
    ax.legend(patches, [x for x in perc_data[0].keys()], loc="lower left")
    plt.show()