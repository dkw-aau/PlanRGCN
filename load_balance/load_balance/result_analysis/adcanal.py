import json
import os
from collections import Counter
from decimal import Decimal
import numpy as np
import pandas as pd


class ADMCTRLAnalyser:
    def __init__(self, test_file, objective_file=None):
        self.stat_label = ['Precision', 'Recall', 'Specificity']
        self.entries = []
        self.acronym_map = {
            'Correct Rejection Rate': 'CRR',
            'Correct Acceptance Rate': 'CAR',
            'Evaluated qs': '\# queries'
        }

        if not (objective_file == None or objective_file == "None"):
            exec(open(objective_file).read(), globals())
            self.label_map = {}
            self.label_index = []
            if 'thresholds' in globals():
                global thresholds
                global cls_func
                for i in range(len(thresholds) - 1):
                    self.label_map[i] = f"({thresholds[i]:.4f};{thresholds[i + 1]:.4f}]"
                    self.label_index.append(f"({thresholds[i]:.4f};{thresholds[i + 1]:.4f}]")

        test_df = pd.read_csv(test_file, sep='\t')
        self.test_df = test_df.set_index('id')
        self.num_slow_qs = dict(Counter(self.test_df['mean_latency'].apply(lambda x: np.argmax(cls_func(x)))))[2]

    def process_results(self):
        new_entries = []
        for e in self.entries:
            dat_set, meth, crr, car, eq, wkl_pct, fast, med, slow, evaluated_non_slow_qs,act_time_outs,qtp, f_rej,false_rejects_fast,false_rejects_med, mean_lat = e
            #new_entries.append((dat_set, meth, self.acronym_map['Correct Rejection Rate'], crr))

            #new_entries.append((dat_set, meth, self.acronym_map['Evaluated qs'], eq))

            new_entries.append((dat_set, meth, 'Throughput', qtp))
            new_entries.append((dat_set, meth, 'NonTimeOut Qs', evaluated_non_slow_qs))
            new_entries.append((dat_set, meth, 'TimeOut Qs', act_time_outs))
            new_entries.append((dat_set, meth, 'False Rejects', f_rej))
            new_entries.append((dat_set, meth, 'False Rejects (fast)', false_rejects_fast))
            new_entries.append((dat_set, meth, 'False Rejects (med)', false_rejects_med))
            new_entries.append((dat_set, meth, 'Workload Latency (s)', wkl_pct))
            new_entries.append((dat_set, meth, 'Mean Latency (s)', mean_lat))

            #new_entries.append((dat_set, meth, self.acronym_map['Correct Acceptance Rate'], car))
            #new_entries.append((dat_set, meth, 'fast', fast))
            #new_entries.append((dat_set, meth, 'med', med))
            #new_entries.append((dat_set, meth, 'slow', slow))

        res = pd.DataFrame(new_entries, columns=['Dataset', 'Method', 'Metric', 'value']).pivot(index='Method',
                                                                                                columns=['Dataset',
                                                                                                         'Metric'],
                                                                                                values='value')
        try:
            print(res.to_latex(column_format='lrrrrrrrrrrrrrrrr', multicolumn_format='c'))
        except:
            print('could not print to latex')
        return res

    def evaluate_dataset(self, path_to_result, method_name, dataset=None):
        self.path_to_res = path_to_result
        rejc = pd.read_csv(os.path.join(self.path_to_res, 'rejected.csv'))
        dct = dict(Counter(rejc['true_cls']))

        false_rejects = (dct[0] if 0 in dct.keys() else 0) + (dct[1] if 1 in dct.keys() else 0)
        false_rejects_fast = (dct[0] if 0 in dct.keys() else 0)
        false_rejects_med =  (dct[1] if 1 in dct.keys() else 0)
        dict(Counter(self.test_df['mean_latency'].apply(lambda x: np.argmax(cls_func(x)))))[2]

        stat = {'Correct Acceptance Rate': None,
                'Correct Rejection Rate': f"{((dct[2] if 2 in dct.keys() else 0) / len(rejc)) * 100:.2f}\%"
                }

        stat['evaluated_qs'] = 0
        tot_lat = 0
        tot_qs = 0
        worker_files = [x for x in os.listdir(self.path_to_res) if x.startswith('w_')]
        actd_fast_qs = 0  # Accepted slow queries
        responses_n_ok = []
        gt_intervals = []
        act_time_outs = 0
        for w in worker_files:
            w_f = os.path.join(self.path_to_res, w)
            data = json.load(open(w_f, 'r'))

            for d in data:
                query = json.loads(d['query'])
                query['true_interval'] = int(Decimal(query['true_interval']))
                gt_intervals.append(query['true_interval'])
                if d['response'] == 'ok':
                    stat['evaluated_qs'] += 1
                    tot_lat += (d['query_execution_end'] - d['queue_arrival_time'])
                else:
                    responses_n_ok.append(d['response'])
                if d['execution_time'] <= 10:
                    actd_fast_qs += 1
                else:
                    act_time_outs += 1
                tot_qs += 1
        resp_n_ok = dict(Counter(responses_n_ok))  # for debugging/run verification process
        gt_dct = dict(Counter(gt_intervals))
        stat['Mean Latency'] = f"{tot_lat / stat['evaluated_qs']:.2f}"


        stat['Correct Acceptance Rate'] = f"{(actd_fast_qs / tot_qs) * 100:.2f}\%"
        """stat['True Positive Rejection Rate'] = f"{stat['True Positive Rejection Rate'] * 100:.2f}\%"
        stat['False Positive 0'] = f"{stat['False Positive 0'] * 100:.2f}\%"
        stat['False Positive 1'] = f"{stat['False Positive 1'] * 100:.2f}\%"
        stat['False Positive all'] = f"{stat['False Positive all'] * 100:.2f}\%"

        stat['Mean Latency'] = stat['Mean Latency']
        del stat['Mean Latency']
        del stat['False Positive all']
        del stat['False Positive 1']
        del stat['False Positive 0']"""

        stat['Evaluated qs'] = stat['evaluated_qs']
        # stat['Timed out qs (gt)'] = gt_timeout_qs
        fast = gt_dct[0] if 0 in gt_dct.keys() else 0
        med = gt_dct[1] if 1 in gt_dct.keys() else 0
        slow = gt_dct[2] if 2 in gt_dct.keys() else 0

        wkl_prc_time = None
        wkl_prc_time_raw = None
        if os.path.exists(os.path.join(self.path_to_res, 'elapsed_time.txt')):
            try:
                txt = open(os.path.join(self.path_to_res, 'elapsed_time.txt'), 'r').read()
                wkl_prc_time = float(Decimal(txt.split(':')[1]))
                wkl_prc_time_raw = wkl_prc_time
                wkl_prc_time = f'{wkl_prc_time:.2f}'
            except:
                wkl_prc_time_raw = 7200
                wkl_prc_time = f'{wkl_prc_time_raw:.2f}'
        wkl_prc_time_raw = 7200 if wkl_prc_time_raw == None else wkl_prc_time_raw
        qtp = f"{(stat['evaluated_qs']/wkl_prc_time_raw)*100:.2f}"
        self.entries.append((dataset, method_name, stat['Correct Rejection Rate'], stat['Correct Acceptance Rate'],
                             stat['Evaluated qs'], wkl_prc_time, fast, med, slow, actd_fast_qs, act_time_outs, qtp, false_rejects,false_rejects_fast,false_rejects_med,stat['Mean Latency']))


class SPARQLquery:
    def __init__(self, ID, queryString, true_interval, execution_time, response_status):
        self.ID = ID
        self.queryString =queryString
        self.true_interval = true_interval
        self.execution_time =execution_time
        self.response_status =response_status


class ADMCTRLAnalyserV2(ADMCTRLAnalyser):

    def __init__(self, test_file, objective_file=None):
        super().__init__(test_file, objective_file=objective_file)


    def process_results(self):
        new_entries = []
        for e in self.entries:
            dat_set, meth, qtp, false_rejections = e
            new_entries.append((dat_set, meth, 'Throughput (query/sec)', qtp))
            new_entries.append((dat_set, meth, 'Falsely rejected queries', false_rejections))

        res = pd.DataFrame(new_entries, columns=['Dataset', 'Method', 'Metric', 'value']).pivot(index='Method',
                                                                                                columns=['Dataset',
                                                                                                         'Metric'],
                                                                                                values='value')
        try:
            print(res.to_latex(column_format='lrrr', multicolumn_format='c'))
        except:
            print('could not print to latex')
        return res

    def evaluate_dataset(self, path_to_result, method_name, dataset=None):
        self.path_to_res = path_to_result
        rejc = pd.read_csv(os.path.join(self.path_to_res, 'rejected.csv'))
        dct = dict(Counter(rejc['true_cls']))

        false_rejects = (dct[0] if 0 in dct.keys() else 0) + (dct[1] if 1 in dct.keys() else 0)
        false_rejects_fast = (dct[0] if 0 in dct.keys() else 0)
        false_rejects_med =  (dct[1] if 1 in dct.keys() else 0)
        dict(Counter(self.test_df['mean_latency'].apply(lambda x: np.argmax(cls_func(x)))))[2]

        stat = {'Correct Acceptance Rate': None,
                'Correct Rejection Rate': f"{((dct[2] if 2 in dct.keys() else 0) / len(rejc)) * 100:.2f}\%"
                }

        stat['evaluated_qs'] = 0
        tot_lat = 0
        tot_qs = 0
        worker_files = [x for x in os.listdir(self.path_to_res) if x.startswith('w_')]
        actd_fast_qs = 0  # Accepted slow queries
        responses_n_ok = []
        gt_intervals = []
        act_time_outs = 0
        qs :[SPARQLquery] = []
        for w in worker_files:
            w_f = os.path.join(self.path_to_res, w)
            data = json.load(open(w_f, 'r'))

            for d in data:
                query = json.loads(d['query'])


                query['true_interval'] = int(Decimal(query['true_interval']))
                qs.append(
                    SPARQLquery(query['ID'], query['text'], query['true_interval'], d['execution_time'], d['response']))
                gt_intervals.append(query['true_interval'])
                if d['response'] == 'ok':
                    stat['evaluated_qs'] += 1
                    tot_lat += (d['query_execution_end'] - d['queue_arrival_time'])
                else:
                    responses_n_ok.append(d['response'])
                if d['execution_time'] <= 10:
                    actd_fast_qs += 1
                else:
                    act_time_outs += 1
                tot_qs += 1
        false_rejections = 0
        for x in qs:
            if x.true_interval in [0, 1] and x.response_status == 'timed out':
                false_rejections += 1


        gt_dct = dict(Counter(gt_intervals))

        stat['Evaluated qs'] = stat['evaluated_qs']

        fast = gt_dct[0] if 0 in gt_dct.keys() else 0
        med = gt_dct[1] if 1 in gt_dct.keys() else 0
        slow = gt_dct[2] if 2 in gt_dct.keys() else 0

        wkl_prc_time = None
        wkl_prc_time_raw = None
        if os.path.exists(os.path.join(self.path_to_res, 'elapsed_time.txt')):
            try:
                txt = open(os.path.join(self.path_to_res, 'elapsed_time.txt'), 'r').read()
                wkl_prc_time = float(Decimal(txt.split(':')[1]))
                wkl_prc_time_raw = wkl_prc_time
                wkl_prc_time = f'{wkl_prc_time:.2f}'
            except:
                wkl_prc_time_raw = 7200
                wkl_prc_time = f'{wkl_prc_time_raw:.2f}'
        wkl_prc_time_raw = 7200 if wkl_prc_time_raw == None else wkl_prc_time_raw
        qtp = f"{(stat['evaluated_qs']/wkl_prc_time_raw)*100:.2f}"
        self.entries.append((dataset, method_name,  qtp,false_rejections))