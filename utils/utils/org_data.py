import pandas as pd
def reorder_df_to_pred_format(df: pd.DataFrame):
    """
    Reorders the df to be on the same format as test_sampled.tsv such that inference functionality can be used
    @param df:
    @return: df
    """
    cols = ['id', 'queryString', 'query_string_0', 'latency_0', 'resultset_0',
            'query_string_1', 'latency_1', 'resultset_1', 'query_string_2',
            'latency_2', 'resultset_2', 'mean_latency', 'min_latency',
            'max_latency', 'time_outs', 'path', 'triple_count', 'subject_predicate',
            'predicate_object', 'subject_object', 'fully_concrete', 'join_count',
            'filter_count', 'left_join_count', 'union_count', 'order_count',
            'group_count', 'slice_count', 'zeroOrOne', 'ZeroOrMore', 'OneOrMore',
            'NotOneOf', 'Alternative', 'ComplexPath', 'MoreThanOnePredicate',
            'queryID', 'Queries with 1 TP', 'Queries with 2 TP',
            'Queries with more TP', 'S-P Concrete', 'P-O Concrete', 'S-O Concrete']
    for x in cols:
        if x not in df.columns:
            df[x] = None
    return df[cols]