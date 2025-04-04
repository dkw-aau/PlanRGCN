import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches

def extract_qs(path, verbose=False):
    qs = []
    timed_out_n_slow = []
    for p in os.listdir(path):
        if p == "main.json":
            continue
        if verbose:
            print(f"{path}/{p}")
        try:
            data = json.load(open(f"{path}/{p}", "r"))
            for q in data:
                q_data = json.loads(q["query"])
                for k in q_data.keys():
                    q[k] = q_data[k]

                if (not "slow" in p) and q["response"] == "timed out":
                    timed_out_n_slow.append(q)
                q["ex_time"] = q["query_execution_end"] - q["query_execution_start"]
                q["latency"] = q["query_execution_end"] - q["arrival_time"]
                q["queue_wait_time"] = (
                    q["query_execution_start"] - q["queue_arrival_time"]
                )
                q["arrival_time"] -= q["start_time"]
                q["queue_arrival_time"] -= q["start_time"]
                q["query_execution_start"] -= q["start_time"]
                q["query_execution_end"] -= q["start_time"]
                q["start_time"] = 0
                q["process"] = p
                qs.append(q)
        except Exception as e:
            if verbose:
                print(f"Did not work {path}/{p}: {e}")
    return qs, timed_out_n_slow


def plot_box_latency(dct, figsize=(4, 6)):
    fig, ax = plt.subplots(layout="constrained", figsize=figsize)
    for k in dct.keys():
        ax = sns.boxplot(y=[q["latency"] for q in dct[k]], x=[k for q in dct[k]], ax=ax)
    # ax = sns.boxplot(y=[q['latency'] for q in qpp_lb],x=['PlanRGCN Load Balancer' for q in qpp_lb], ax=ax)
    ax.set_ylabel("Query Latency (s)")
    ax.set_title("Query Latency Plots")
    ax.tick_params(axis="x", rotation=15)
    plt.show()


def get_overview_table(path_data):
    d_p = {"Good Queries": {}, "Time out": {}}
    for k in path_data:
        data, timeouts = extract_qs(path_data[k])
        d_p["Time out"][k] = np.sum([1 for x in data if x["ex_time"] >= 900])
        d_p["Good Queries"][k] = np.sum([1 for x in data if x["response"] == "ok"])
    df = pd.DataFrame.from_dict(d_p)
    return df


def get_time_outs(data):
    print("Time outs")
    dct = {}
    for k in data.keys():
        val = np.sum([1 for x in data[k] if x["ex_time"] >= 900])
        print(f"{k} Timeouts : {val}")
        dct[k] = val
    return dct


def get_good_qs(data):
    print("Good Queries")
    dct = {}
    for k in data.keys():
        val = np.sum([1 for x in data[k] if x["response"] == "ok"])
        print(f"{k} Good Queries : {val}")
        dct[k] = val
    return dct


def calculate_total_latency(qs):
    sum = 0
    for q in qs:
        sum += q["latency"]
    return sum


def calculate_avg_latency(data):
    dct = {}
    for k in data.keys():
        val = calculate_total_latency(data[k]) / len(data[k])
        print(f"{k}:  {val}")
        dct[k] = val
    return dct


def plot_box_ex_time(dct, figsize=(4, 4)):
    fig, ax = plt.subplots(layout="constrained", figsize=figsize)
    for k in dct.keys():
        ax = sns.boxplot(y=[q["ex_time"] for q in dct[k]], x=[k for q in dct[k]], ax=ax)
    ax.tick_params(axis="x", rotation=15)
    plt.show()


def plot_box_queu_wait_time(dct, figsize=(4, 6)):
    fig, ax = plt.subplots(layout="constrained", figsize=figsize)
    for k in dct.keys():
        ax = sns.boxplot(
            y=[q["queue_wait_time"] for q in dct[k]], x=[k for q in dct[k]], ax=ax
        )
    ax.set_ylabel("Queue Wait Time (s)")
    ax.set_title("Queue Wait Time Plots (Wikidata Full)")
    ax.tick_params(axis="x", rotation=10)
    plt.show()


def plot_box_queu_wait_time_int(
    dct,
    figsize=(4, 6),
    title="Queue Wait Time by Time Intervals",
    time_intervals=["fast", "medium", "slow"],
):
    load_balancer = []
    que_w_time = []
    runtime_interval = []
    for k in dct.keys():
        for q in dct[k]:
            load_balancer.append(k)
            que_w_time.append(q["queue_wait_time"])
            match q["true_interval"]:
                case "0":
                    runtime_interval.append(time_intervals[0])
                case "1":
                    runtime_interval.append(time_intervals[1])
                case "2":
                    runtime_interval.append(time_intervals[2])

    df = pd.DataFrame.from_dict(
        {
            "Approach": load_balancer,
            "Runtime Interval": runtime_interval,
            "queue wait time": que_w_time,
        }
    )
    fig, ax = plt.subplots(layout="constrained", figsize=figsize)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax = sns.boxplot(
        x="Runtime Interval",
        y="queue wait time",
        data=df,
        hue="Approach",
        ax=ax,
        order=time_intervals,
    )
    ax.set_xlabel("")
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Queue Wait Time (s)", fontsize=14)
    ax.legend(loc=2)

import ast
"""def get_latency_bxp_data(dct, time_intervals):
    load_balancer = []
    latencies = []
    runtime_interval = []
    for k in dct.keys():
        for q in dct[k]:
            load_balancer.append(k)
            latencies.append(q["latency"])
            print('hello')
            
            match q["true_interval"]:
                case "0":
                    runtime_interval.append(time_intervals[0])
                case "1":
                    runtime_interval.append(time_intervals[1])
                case "2":
                    runtime_interval.append(time_intervals[2])
    return load_balancer, runtime_interval, latencies"""


def plot_box_latency_int(
    dct,
    figsize=(4, 6),
    title="Queue Latency Time by Time Intervals",
    time_intervals=["fast", "medium", "slow"],
):
    load_balancer, runtime_interval, latencies = get_latency_bxp_data(
        dct, time_intervals
    )
    df = pd.DataFrame.from_dict(
        {
            "Approach": load_balancer,
            "Runtime Interval": runtime_interval,
            "queue wait time": latencies,
        }
    )
    fig, ax = plt.subplots(layout="constrained", figsize=figsize)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax = sns.boxplot(
        x="Runtime Interval",
        y="queue wait time",
        data=df,
        hue="Approach",
        ax=ax,
        order=time_intervals,
    )
    ax.set_xlabel("")
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Query Latency (s)", fontsize=14)
    ax.legend(loc=2)

def get_latency_bxp_data(dct, time_intervals):
    load_balancer = []
    latencies = []
    runtime_interval = []
    for k in dct.keys():
        for q in dct[k]:
            load_balancer.append(k)
            latencies.append(q["latency"])
            if isinstance(q["true_interval"], str):
                q["true_interval"] = int(ast.literal_eval(q["true_interval"]))
            if isinstance(q["true_interval"], float):
                q["true_interval"] = int(q["true_interval"])
            match q["true_interval"]:
                case 0:
                    runtime_interval.append(time_intervals[0])
                case 1:
                    runtime_interval.append(time_intervals[1])
                case 2:
                    runtime_interval.append(time_intervals[2])
                case _:
                    print(q)
                    raise Exception("Should not happen")
    return load_balancer, runtime_interval, latencies

def get_query_per_interval(dct, time_intervals):
    load_balancer = []
    latencies = []
    runtime_interval = []
    for k in dct.keys():
        for q in dct[k]:
            load_balancer.append(k)
            latencies.append(q["latency"])
            if isinstance(q["true_interval"], str):
                q["true_interval"] = int(ast.literal_eval(q["true_interval"]))
            if isinstance(q["true_interval"], float):
                q["true_interval"] = int(q["true_interval"])
            match q["true_interval"]:
                case 0:
                    runtime_interval.append(time_intervals[0])
                case 1:
                    runtime_interval.append(time_intervals[1])
                case 2:
                    runtime_interval.append(time_intervals[2])
                case _:
                    print(q)
                    raise Exception("Should not happen")
    return load_balancer, runtime_interval, latencies

def plot_box_lat_int_comb(
    *dcts,
    figsize=(4, 6),
    title="Queue Latency Time by Time Intervals",
    time_intervals=["fast", "medium", "slow"],
    legend_dict={"bbox_to_anchor": (1.1, 1.05)},
    int_to_col = { 'FIFO (μ=44)':'#a1c9f4','PlanRGCN (μ=44)':'#ffb482',  'Oracle (μ=44)':'#8de5a1'},
    hatch_dict = None,
    x_label_size=10,
    x_rotation = 0,
    y_label_size=10,
    save_path = None,
    dpi = 100,
    nrows = 1
):
    fig, axes = plt.subplots(
        nrows=nrows, ncols=len(dcts), layout="constrained", figsize=figsize,sharey=True
    )
    for i, (dct, log_name) in enumerate(dcts):
        load_balancer, runtime_interval, latencies = get_latency_bxp_data(
            dct, time_intervals
        )
        df = pd.DataFrame.from_dict(
            {
                "Approach": load_balancer,
                "Runtime Interval": runtime_interval,
                "queue wait time": latencies,
            }
        )
        if len(dcts)==1:
            axes.tick_params(axis="x", rotation=x_rotation)
            axes.tick_params(axis="x", labelsize=x_label_size)
            axes.tick_params(axis="y", labelsize=y_label_size)
            axes = sns.boxplot(
                x="Runtime Interval",
                y="queue wait time",
                data=df,
                hue="Approach",
                palette=int_to_col,
                ax=axes,
                order=time_intervals,
               medianprops={'linewidth': 2}, whiskerprops={'linewidth': 2}, capprops={'linewidth': 2},
                
            )
            axes.set_title(title, fontsize=14)
            if i == 0:
                axes.set_ylabel("Query Latency (s)", fontsize=14, weight='bold')
                axes.legend("", frameon=False)
            else:
                axes.set_ylabel("")
                axes.legend("", frameon=False)
            axes.set_xlabel(log_name)
        else:
            axes[i].tick_params(axis="x", rotation=x_rotation)
            axes[i].tick_params(axis="x", labelsize=x_label_size)
            axes[i].tick_params(axis="y", labelsize=y_label_size)
            axes[i] = sns.boxplot(
                x="Runtime Interval",
                y="queue wait time",
                data=df,
                hue="Approach",
                palette=int_to_col,
                ax=axes[i],
                order=time_intervals,
               medianprops={'linewidth': 2}, whiskerprops={'linewidth': 2}, capprops={'linewidth': 2},
                
            )
            axes[i].set_title(title, fontsize=14)
            #print(axes[i].)
            if i == 0:
                axes[i].set_ylabel("Query Latency (s)", fontsize=14, weight='bold')
                axes[i].legend("", frameon=False)
            else:
                axes[i].set_ylabel("")
                axes[i].legend("", frameon=False)
            axes[i].set_xlabel(log_name)
            # Add hatch patterns if hatch_dict is provided
    if hatch_dict != None:
        print(hatch_dict)
        patches = [ mpatches.Patch(facecolor=int_to_col[k], label=k, hatch=hatch_dict[k]) for k in int_to_col.keys()]
    else:
        patches = [ mpatches.Patch(color=int_to_col[k], label=k, ) for k in int_to_col.keys()]
    lgd = fig.legend(handles=patches,loc=2, **legend_dict)
    if save_path != None:
        fig.savefig(save_path, dpi=dpi,bbox_extra_artists=(lgd,), bbox_inches='tight')
    #plt.legend(loc=2, **legend_dict)

def check_interval_change(
    *dcts,
    figsize=(4, 6),
    time_intervals=["fast", "medium", "slow"],
    int_to_col = { 'FIFO (μ=44)':'#a1c9f4','PlanRGCN (μ=44)':'#ffb482',  'Oracle (μ=44)':'#8de5a1'},
):
    for i, (dct, log_name) in enumerate(dcts):
        load_balancer, runtime_interval, latencies = get_latency_bxp_data(
            dct, time_intervals
        )
        df = pd.DataFrame.from_dict(
            {
                "Approach": load_balancer,
                "Runtime Interval": runtime_interval,
                "queue wait time": latencies,
            }
        )
