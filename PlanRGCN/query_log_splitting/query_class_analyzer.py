import pickle
from collections import Counter

import jpype
import numpy as np
import pandas as pd

from graph_construction.jar_utils import get_query_graph, get_ent_rel


class QueryClassAnalyzer:
    def __init__(self, train_file, val_file, test_file, data_split_path=None, objective_file=None):
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file

        self.train_df = pd.read_csv(train_file, sep='\t')
        self.val_df = pd.read_csv(val_file, sep='\t')
        self.test_df = pd.read_csv(test_file, sep='\t')

        if objective_file == None:
            self.is_obj = False
            with open(data_split_path, 'rb') as f:
                self.data_splitter = pickle.load(f)
            self.intervals = []
            for i in range(len(self.data_splitter.interval_extractor.thresholds) - 1):
                self.intervals.append(str((self.data_splitter.interval_extractor.thresholds[i],
                                           self.data_splitter.interval_extractor.thresholds[i + 1])))
        else:
            self.is_obj = True
            global thresholds
            thresholds = None
            exec(open(objective_file).read(),globals())
            assert thresholds != None
            self.intervals = []
            for i in range(len(thresholds) - 1):
                if thresholds[i] == 0:
                    self.intervals.append(f'({thresholds[i]};{thresholds[i + 1]:.4f}]')
                elif thresholds[i + 1] == 900:
                    self.intervals.append(f'({thresholds[i]:.4f};{thresholds[i + 1]}]')
                else:
                    self.intervals.append(f'({thresholds[i]:.4f};{thresholds[i + 1]:.4f}]')

        entries = []
        entries.extend(self.get_query_class(self.train_df, 'Train'))
        entries.extend(self.get_query_class(self.val_df, 'val'))
        entries.extend(self.get_query_class(self.test_df, 'Test'))
        t_df = pd.DataFrame(entries, columns=['Split', 'interval', 'query type', '# queries'])
        t_df = t_df.set_index(['query type']).pivot(columns=['Split', 'interval'], values='# queries')
        print(t_df.to_latex())

    def latex_print(self, table):
        s = table.style.highlight_max(subset='0.2', props='textbf:--rwrap:')
        s = s.format(precision=2, escape='latex')
        s = s.set_table_styles([{'selector': 'midrule', 'props': ':hline'}])
        print(s.to_latex(clines='skip-last;data', multicol_align='c', multirow_align='c',
                         column_format='l' * 3 + '|' + 'r' * len(table.columns)))
    def get_assign_interval_df(self, df: pd.DataFrame):
        if self.is_obj:
            global cls_func
            int_func = lambda x: str(self.intervals[np.argmax(cls_func(x))])
            df['interval'] = df['mean_latency'].apply(
                lambda x: str(int_func(x)))
        else:
            df['interval'] = df['mean_latency'].apply(
                lambda x: str(self.data_splitter.interval_extractor.find_query_intervals(x)))
        return df

    def get_query_class(self, df: pd.DataFrame, splt_type: str, ):
        entries = []
        opt_ids = []
        pp_ids = []
        remaining = []
        filter_ids = []
        complex_ids = []
        all_ids = []
        for idx, row in df.iterrows():
            try:
                qg = get_query_graph(row['queryString'])
            except jpype.JException:
                continue
            all_ids.append(idx)
            isPP = False
            isOptional = False
            isFilter = False
            isRegular = False
            isComplex = False
            if np.sum([1 for x in qg['nodes'] if x['nodeType'] == 'PP']) > 0:
                pp_ids.append(idx)
                isPP = True
            if np.sum([1 for x in qg['edges'] if x[2] == 'Optional']) > 0:  # for optional
                opt_ids.append(idx)
                isOptional = True
            if np.sum([1 for x in qg['nodes'] if x['nodeType'] == 'FILTER']) > 0:
                filter_ids.append(idx)
                isFilter = True
            if (isPP == False) and (isOptional == False) and (isFilter == False):
                remaining.append(idx)
                isRegular = True
            if (isOptional and isFilter) or (isOptional and isPP) or (isPP and isFilter):
                isComplex = True
                complex_ids.append(idx)

        for query_class, idx in zip(['Optional', 'PP', 'Filter', 'Complex', 'Simple', 'Total'],
                                    [opt_ids, pp_ids, filter_ids, complex_ids, remaining, all_ids]):
            df_cls = df.loc[idx].copy()
            df_cls = self.get_assign_interval_df(df_cls)
            interval_dst = dict(Counter(df_cls['interval']))
            for k in self.intervals:
                query_number = interval_dst[k]
                entries.append((splt_type, k, query_class, query_number))

        #For unseen test query statistics
        if splt_type == 'Test':
            df_cls = self.get_unseen_query_distibution()
            df_cls = self.get_assign_interval_df(df_cls)
            interval_dst = dict(Counter(df_cls['interval']))
            for k in self.intervals:
                query_number = interval_dst[k]
                entries.append((splt_type, k, 'unseen', query_number))
        else:
            #they don't exist for validation and training set
            for k in self.intervals:
                entries.append((splt_type, k, 'unseen', '---'))
        return entries

    def get_unseen_query_distibution(self):
        non_test_terms = set()
        for idx, row in self.train_df.iterrows():
            try:
                ents, rels = get_ent_rel(row['queryString'])
                non_test_terms.update(ents)
                non_test_terms.update(rels)

            except jpype.JException:
                continue


        for idx, row in self.val_df.iterrows():
            try:
                ents, rels = get_ent_rel(row['queryString'])
                non_test_terms.update(ents)
                non_test_terms.update(rels)
            except jpype.JException:
                continue

        complete_unseen_idx = []
        for idx, row in self.test_df.iterrows():
            try:
                ents, rels = get_ent_rel(row['queryString'])
                is_completely_unseen = True
                for e in ents:
                    if e in non_test_terms:
                        is_completely_unseen = False
                for r in rels:
                    if r in non_test_terms:
                        is_completely_unseen = False
                if is_completely_unseen:
                    complete_unseen_idx.append(row['id'])
            except jpype.JException:
                continue
        return self.test_df[self.test_df['id'].isin(complete_unseen_idx)].copy()




class FineGrainedQueryClassAnalyzer(QueryClassAnalyzer):

    def __init__(self,train_file, val_file, test_file, data_split_path=None, objective_file=None):
        super().__init__(train_file, val_file, test_file, data_split_path=data_split_path, objective_file=objective_file)



    def get_op_count(self, qp:dict, q:str):
        tps = 0
        pps = 0
        filters = 0

        for n in qp['nodes']:
            if n['nodeType'] == 'TP':
                tps += 1
            elif n['nodeType'] == 'PP':
                pps += 1
            elif n['nodeType'] == 'FILTER':
                filters += 1

        optional = q.lower().count('optional')
        return tps, pps, optional, filters

    def get_query_class(self, df: pd.DataFrame, splt_type: str, print_PPs=True ):
        stat = {
        'tp_pp_fil_opt' : [],
        'tp_pp_fil' : [],
        'tp_opt_pp' : [],
        'tp_opt_fil' : [],
        'tp_fil' : [],
        'tp_pp' : [],
        'tp_opt' : [],
        'tp_1' : [],
        'tp_2' : [],
        'tp_3' : [],
        'tp_more' : [],
        'pp' : [],
        'pp_fil' : [],
        'pp_opt' : [],
        'pp_opt_fil' : [],
        'Total': df.index.tolist()
        }

        unbinned = []
        pp_qs = 0
        for idx, row in df.iterrows():
            try:
                qg = get_query_graph(row['queryString'])
                tps, pps, optionals, filters = self.get_op_count(qg, row['queryString'])
                if pps > 0:
                    pp_qs += 1
                if tps > 0:
                    # TP related info
                    if optionals > 0 and filters > 0 and pps > 0:
                        stat['tp_pp_fil_opt'].append(idx)
                    elif pps > 0 and filters > 0:
                        stat['tp_pp_fil'].append(idx)
                    elif optionals > 0 and pps > 0:
                        stat['tp_opt_pp'].append(idx)
                    elif optionals > 0 and filters > 0:
                        stat['tp_opt_fil'].append(idx)
                    elif filters > 0:
                        stat['tp_fil'].append(idx)
                    elif pps > 0:
                        stat['tp_pp'].append(idx)
                    elif optionals> 0:
                        stat['tp_opt'].append(idx)
                    elif tps == 1:
                        stat['tp_1'].append(idx)
                    elif tps == 2:
                        stat['tp_2'].append(idx)
                    elif tps == 3:
                        stat['tp_3'].append(idx)
                    elif tps > 3:
                        stat['tp_more'].append(idx)
                    else:
                        unbinned.append(idx)
                else:
                    assert pps > 0
                    if optionals > 0 and filters > 0:
                        stat['pp_opt_fil'].append(idx)
                    elif optionals > 0:
                        stat['pp_opt'].append(idx)
                    elif filters > 0:
                        stat['pp_fil'].append(idx)
                    else:
                        stat['pp'].append(idx)
            except jpype.JException:
                unbinned.append(idx)

        if print_PPs:
            print(f"Total PP queries: {pp_qs}")
            print(f"Total unbinned querires: {len(unbinned)}")

        #Class remaming:
        n = {}

        n['TP 1'] = stat['tp_1']
        n['TP 2'] = stat['tp_2']
        n['TP 3'] = stat['tp_3']
        n['$>$TP 3'] = stat['tp_more']
        n['TP F'] = stat['tp_fil']
        n['TP OPT'] = stat['tp_opt']
        n['TP OPT F'] = stat['tp_opt_fil']
        n['TP PP F'] = stat['tp_pp_fil']
        n['PP'] = stat['pp']
        n['TP PP'] = stat['tp_pp']
        n['TP OPT PP'] = stat['tp_opt_pp']
        n['PP F'] = stat['pp_fil']
        n['PP OPT'] = stat['pp_opt']
        n['PP OPT F'] = stat['pp_opt_fil']
        n['TP PP F OPT'] = stat['tp_pp_fil_opt']
        n['Total'] = stat['Total']
        workload_a = {'simple': n['TP 1']+n['TP 2']+n['TP 3']+n['$>$TP 3'],
                      'C2RPQ': n['TP PP'],
                      'complex BGP': n['TP F'] +n['TP OPT'] +n['TP OPT F'],
                      'complex C2RPQ': n['TP OPT PP'] + n['TP PP F OPT'] + n['TP PP F']}
        print('Workload a')
        workload_b = {}
        for k in workload_a.keys():
            workload_b[k] = len(workload_a[k])

        print(workload_b)
        stat = n
        entries = []
        for query_class in ['TP 1','TP 2','TP 3','$>$TP 3','TP F' ,'TP OPT', 'TP OPT F', 'TP PP F', 'PP', 'TP PP', 'TP OPT PP', 'PP F', 'PP OPT', 'PP OPT F', 'TP PP F OPT','Total' ]:
            df_cls = df.loc[stat[query_class]].copy()
            df_cls = self.get_assign_interval_df(df_cls)
            interval_dst = dict(Counter(df_cls['interval']))
            for k in self.intervals:
                try:
                    query_number = interval_dst[k]
                except Exception:
                    query_number = 0
                entries.append((splt_type, k, query_class, query_number))

        #For unseen test query statistics
        if splt_type == 'Test':
            df_cls = self.get_unseen_query_distibution()
            df_cls = self.get_assign_interval_df(df_cls)
            interval_dst = dict(Counter(df_cls['interval']))
            for k in self.intervals:
                query_number = interval_dst[k]
                entries.append((splt_type, k, 'unseen', query_number))
        else:
            #they don't exist for validation and training set
            for k in self.intervals:
                entries.append((splt_type, k, 'unseen', '---'))
        return entries

class SemiFineGrainedQueryClassAnalyzer(FineGrainedQueryClassAnalyzer):
    def __init__(self, train_file, val_file, test_file, data_split_path=None, objective_file=None):
        super().__init__(train_file, val_file, test_file, data_split_path=data_split_path, objective_file=objective_file)

    def get_query_class(self, df: pd.DataFrame, splt_type: str, print_PPs=True ):
        tp1 = 'With 1 TP'
        tp2 = 'With 2 TP'
        tpmore = 'With more than 2 TP'
        optional = 'With OPTIONAL'
        filter_s = 'With FILTER'
        pp_str = 'With PP'
        stat = {
            tp1: [],
            tp2: [],
            tpmore: [],
            optional: [],
            filter_s: [],
            pp_str: [],
            'Total': df.index.tolist()
        }

        unbinned = []
        pp_qs = 0
        for idx, row in df.iterrows():
            try:
                qg = get_query_graph(row['queryString'])
                tps, pps, optionals, filters = self.get_op_count(qg, row['queryString'])
                if pps > 0:
                    stat[pp_str].append(idx)
                if tps > 0:
                    # TP related info
                    if optionals > 0:
                        stat[optional].append(idx)
                    if filters > 0:
                        stat[filter_s].append(idx)
                    if pps > 0:
                        stat[pp_str].append(idx)
                    if tps == 1:
                        stat[tp1].append(idx)
                    elif tps == 2:
                        stat[tp2].append(idx)
                    else:
                        stat[tpmore].append(idx)
            except jpype.JException:
                unbinned.append(idx)

        if print_PPs:
            print(f"Total PP queries: {pp_qs}")
            print(f"Total unbinned querires: {len(unbinned)}")

        entries = []
        for query_class in [tp1, tp2, tpmore, optional, filter_s, pp_str,'Total' ]:
            df_cls = df.loc[stat[query_class]].copy()
            df_cls = self.get_assign_interval_df(df_cls)
            interval_dst = dict(Counter(df_cls['interval']))
            for k in self.intervals:
                try:
                    query_number = interval_dst[k]
                except Exception:
                    query_number = 0
                entries.append((splt_type, k, query_class, query_number))

        #For unseen test query statistics
        if splt_type == 'Test':
            df_cls = self.get_unseen_query_distibution()
            df_cls = self.get_assign_interval_df(df_cls)
            interval_dst = dict(Counter(df_cls['interval']))
            for k in self.intervals:
                query_number = interval_dst[k]
                entries.append((splt_type, k, 'unseen', query_number))
        else:
            #they don't exist for validation and training set
            for k in self.intervals:
                entries.append((splt_type, k, 'unseen', '---'))
        return entries
