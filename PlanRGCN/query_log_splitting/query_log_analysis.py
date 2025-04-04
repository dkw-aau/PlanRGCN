import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter


class QueryLogAnalyzer:
    def __init__(self, query_df:pd.DataFrame):
        self.query_df = query_df

    def print_statistics(self):
        print(self.query_df['mean_latency'].describe(percentiles=[0.25,0.50,0.75,0.9,0.95,0.99]))
    def hist(self, single_plt=True, x_tick_plot=2, title=''):
        sns.set_style('white')
        sns.set_context("paper",font_scale=1.5)
        n_bins= 50
        tick_labs = []
        if single_plt:
            fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
            # We can set the number of bins with the *bins* keyword argument.
            _, b,_ = axs.hist(self.query_df['mean_latency'], bins=n_bins)
            print(b)
            for n_t, t in enumerate(b):
                if n_t % x_tick_plot == 0:
                    tick_labs.append(t)
            print(tick_labs)
            plt.xticks(tick_labs,rotation=45)
            #axs.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

        plt.yscale('log')

        #plt.xscale('log')
        plt.ylabel('# Queries')
        plt.xlabel('Runtime (s)')
        plt.title(title)
        sns.despine()

        plt.show()

    def bxp(self,single_plt=True, x_tick_plot=2, title=''):
            sns.set_style('white')
            sns.set_context("paper", font_scale=1.5)
            n_bins = 50
            tick_labs = []
            if single_plt:
                fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)

                VP = axs.boxplot(self.query_df['mean_latency'], patch_artist=True, showmeans=True, showfliers=True )

                # axs.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

            plt.yscale('log')

            # plt.xscale('log')
            plt.ylabel('# Queries')
            plt.xlabel('Runtime (s)')
            plt.title(title)
            sns.despine()

            plt.show()