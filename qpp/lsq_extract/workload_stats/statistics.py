import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

class Stat:
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(path, sep='§', engine='python')
        print(len(self.df))
        cols = self.df.columns[8:35]
        total_count = self.df[cols].sum()
        #print(self.get_no_operators())
        #plt_df = pd.DataFrame({"op":list(total_count.index),"Total":list(total_count)})
        #print(plt_df)
        #sns.swarmplot(x="op", y="Total", data=plt_df)
        #plt.show()

        #self.freq = self.sort('sum')
        #print(self.freq)
    def get_no_operators(self):
        cols = self.df.columns[8:35]
        total_count = self.df[cols].sum()
        plt_df = pd.DataFrame({"op":list(total_count.index),"Total":list(total_count)})
        return list(plt_df.loc[plt_df['Total'] == 0]['op'])