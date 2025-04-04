import sys
import pandas as pd
import pathlib
import inductive_query.pp_qextr as pp_qextr
PPQueryExtractor = pp_qextr.PPQueryExtractor

#Arg1 : eg /data/wikidatav3_path_PPrepV2
ext = PPQueryExtractor(sys.argv[1])
unseen_pred_queryID =[pathlib.Path(x).name for x in ext.get_val_PP_files()]
df = pd.read_csv(f'{sys.argv[1]}/val_sampled.tsv',sep='\t')
unseen_pred_queryID = [f'http://lsq.aksw.org/{x}' for x in unseen_pred_queryID]
t_df = df[df['id'].isin(unseen_pred_queryID)]
t_df.to_csv(f'{sys.argv[1]}/PP_val_sampled.tsv',sep='\t',index=False)