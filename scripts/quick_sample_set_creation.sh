cp DBpedia_3_class_full/val_sampled.tsv test_set/
cp DBpedia_3_class_full/train_sampled.tsv test_set/
cp DBpedia_3_class_full/test_sampled.tsv test_set/
cd test_set
source /virt_env_develop/bin/activate
python3 -c """
import panda as pd

def sample_df(path):
  df = pd.read_csv(path, sep='\t')
  fast = df[df.mean_latency<1].sample(10)
  med = df[(1<df.mean_latency)&(df.mean_latency<10)].sample(10)
  slow = df[(10<df.mean_latency)&(df.mean_latency<900)].sample(10)
  df = pd.concat([fast,med,slow])
  df.to_csv(path, sep='\t', index=False)

sample_df('test_sampled.tsv')

sample_df('train_sampled.tsv')

sample_df('val_sampled.tsv')
"""