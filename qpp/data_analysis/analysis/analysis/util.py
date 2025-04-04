import matplotlib.pyplot as plt
import pandas as pd


def hello():
    return 1


def plot_operator_freq(df, showPlot=True, plotname="operator_freq.png", x_label=""):
    skippeable_columns_freq = [
        "queryID",
        "timestamp",
        "queryString",
        "tripleCount",
        "duration",
        "resultSize",
        "latency",
        "triple",
        "bgp",
        "treesize",
        "projectVariables",
        "joinVertexCount",
    ]  #'slice',#, 'treesize','projectVariables', 'leftjoin', 'union'
    skippeable_columns_freq.append("resultCount")
    freq = get_operator_freq(df, skipable_cols=skippeable_columns_freq)
    print(freq)

    ax = freq.plot(kind="bar", x="operators", y="freq")
    ax.bar_label(ax.containers[0], label_type="edge")
    ax.set_ylabel("Total Operators in dataset")
    ax.set_xlabel(x_label)
    plt.yscale("log")
    plt.subplots_adjust(
        left=0.104, bottom=0.218, right=0.986, top=1, hspace=0.2, wspace=0.2
    )
    if showPlot:
        plt.show()
    else:
        plt.savefig(plotname)
        plt.close()


def plot_operator_presence(
    df, showPlot=True, plotname="operator_presence.png", x_label=""
):
    skippeable_columns_freq = [
        "queryID",
        "timestamp",
        "queryString",
        "tripleCount",
        "duration",
        "resultSize",
        "latency",
        "triple",
        "bgp",
        "treesize",
        "projectVariables",
        "joinVertexCount",
    ]  #'leftjoin', 'union'
    skippeable_columns_freq.extend(["resultCount"])
    freq = get_operator_presence(df, skipable_cols=skippeable_columns_freq)
    print(freq)
    ax = freq.plot(kind="bar", x="operators", y="freq")
    ax.set_ylabel("Queries")
    ax.set_xlabel(x_label)
    ax.bar_label(ax.containers[0], label_type="edge")
    plt.yscale("log")
    plt.subplots_adjust(
        left=0.104, bottom=0.218, right=0.986, top=1, hspace=0.2, wspace=0.2
    )
    # plt.subplots_adjust(left=)
    if showPlot:
        plt.show()
    else:
        plt.savefig(plotname)
        plt.close()


def get_operator_freq(train: pd.DataFrame, skipable_cols=[]):
    freq = {"operators": [], "freq": []}
    for col in train.columns:
        if col in skipable_cols:
            continue
        freq["operators"].append(col)
        freq["freq"].append(train[col].sum())
    df = pd.DataFrame.from_dict(freq)
    df = df.sort_values(by=["freq"], ascending=False)
    df = df.drop(df.loc[df["freq"] == 0].index)
    return df


def get_operator_presence(train: pd.DataFrame, skipable_cols=[]):
    freq = {"operators": [], "freq": []}
    for col in train.columns:
        if col in skipable_cols:
            continue
        freq["operators"].append(col)
        temp_df = train.loc[(train[col] > 0)]
        freq["freq"].append(len(temp_df))
    df = pd.DataFrame.from_dict(freq)
    df = df.sort_values(by=["freq"], ascending=False)
    df = df.drop(df.loc[df["freq"] == 0].index)
    return df


def latency_bxplot(latency, title, showPlot=True, plotname="boxplot.png"):
    plt.clf()
    fig1, ax1 = plt.subplots()

    print(ax1)
    # plt.yscale('log')
    ax1.set_title(title)
    ax1.set_ylabel("Query Run Tim in MS")
    ax1.boxplot(latency)
    if showPlot:
        plt.show()
    else:
        plt.savefig(plotname)
        plt.close()
