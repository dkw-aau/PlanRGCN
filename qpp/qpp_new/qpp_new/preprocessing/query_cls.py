import pandas as pd
import functools


class QueryCls:
    def __init__(self):
        pass

    def get_features_filter(df, query_type="simple"):
        df["queryString"] = df["queryString"].astype(str)
        df_filt_idx = df["queryString"].str.contains("FILTER")
        df_filt_not_exists_idx = df["queryString"].str.contains("FILTER NOT EXISTS")
        df_filt_not_exists_idx = df_filt_not_exists_idx.astype(bool)
        df_filt_idx = df_filt_idx & (~df_filt_not_exists_idx)
        df_filt_exists_idx = df["queryString"].str.contains("FILTER EXISTS")
        df_filt_exists_idx = df_filt_exists_idx.astype(bool)
        df_filt_idx = df_filt_idx & ~df_filt_exists_idx
        df_union_idx = df["queryString"].str.contains("UNION")
        df_optionals_idx = df["queryString"].str.contains("OPTIONAL")
        df_values_idx = df["queryString"].str.contains("VALUES")
        df_minus_idx = df["queryString"].str.contains("MINUS")
        df_having_idx = df["queryString"].str.contains("HAVING")
        df_groupby_idx = df["queryString"].str.contains("GROUP BY")
        df_orderby_idx = df["queryString"].str.contains("ORDER BY")

        if query_type == "opt":
            return df_optionals_idx

        if query_type == "filt":
            return df_filt_idx

        subq_lambda = lambda x: True if len(x.split("SELECT")) > 2 else False
        df_subq_idx = pd.Series(df["queryString"].apply(subq_lambda))
        df_limit = df["queryString"].str.contains("LIMIT")

        if "path" in df.columns:
            df_path = df["path"] > 0
        else:
            df_path = df["path*"] > 0
            df_path = df_path | df["pathN*"] > 0
            df_path = df_path | df["path+"] > 0
            df_path = df_path | df["pathN+"] > 0
            df_path = df_path | df["path?"] > 0
            df_path = df_path | df["notoneof"] > 0

        simple_query_filt = [
            df_filt_idx,
            df_filt_not_exists_idx,
            df_filt_exists_idx,
            df_union_idx,
            df_optionals_idx,
            df_values_idx,
            df_minus_idx,
            df_having_idx,
            df_groupby_idx,
            df_path,
            df_orderby_idx,
            df_subq_idx,
            df_limit,
        ]
        simple_query_filt = functools.reduce(lambda x, y: x | y, simple_query_filt)
        simple_query_filt = ~simple_query_filt

        simple_query_simpl_filt_filt = [
            df_filt_not_exists_idx,
            df_filt_exists_idx,
            df_union_idx,
            df_optionals_idx,
            df_values_idx,
            df_minus_idx,
            df_having_idx,
            df_groupby_idx,
            df_path,
            df_orderby_idx,
            df_subq_idx,
            df_limit,
        ]
        simple_query_simpl_filt_filt = functools.reduce(
            lambda x, y: x | y, simple_query_simpl_filt_filt
        )
        simple_query_simpl_filt_filt = ~simple_query_simpl_filt_filt

        simple__opt_query_filt = [
            df_filt_idx,
            df_filt_not_exists_idx,
            df_filt_exists_idx,
            df_union_idx,
            df_values_idx,
            df_minus_idx,
            df_having_idx,
            df_groupby_idx,
            df_path,
            df_orderby_idx,
            df_subq_idx,
            df_limit,
        ]
        simple__opt_query_filt = functools.reduce(
            lambda x, y: x | y, simple__opt_query_filt
        )
        simple__opt_query_filt = ~simple__opt_query_filt

        simple_simple_filt_opt_query_filt = [
            df_filt_not_exists_idx,
            df_filt_exists_idx,
            df_union_idx,
            df_values_idx,
            df_minus_idx,
            df_having_idx,
            df_groupby_idx,
            df_path,
            df_orderby_idx,
            df_subq_idx,
            df_limit,
        ]
        simple_simple_filt_opt_query_filt = functools.reduce(
            lambda x, y: x | y, simple_simple_filt_opt_query_filt
        )
        simple_simple_filt_opt_query_filt = ~simple_simple_filt_opt_query_filt

        simple_query_filt_union = [
            df_filt_idx,
            df_filt_not_exists_idx,
            df_filt_exists_idx,
            df_optionals_idx,
            df_values_idx,
            df_minus_idx,
            df_having_idx,
            df_groupby_idx,
            df_path,
            df_orderby_idx,
            df_subq_idx,
            df_limit,
        ]
        simple_query_filt_union = functools.reduce(
            lambda x, y: x | y, simple_query_filt_union
        )
        simple_query_filt_union = ~simple_query_filt_union

        match query_type:
            case "single":
                return (df["triple"] == 1) & simple_query_filt
            case "multiple":
                return (df["triple"] > 1) & simple_query_filt
            case "simple":
                return simple_query_filt
            case "simple_filt":
                return simple_query_simpl_filt_filt
            case "simple_opt":
                return simple__opt_query_filt
            case "simple_opt_filt":
                return simple_simple_filt_opt_query_filt
            case "simple_union":
                return simple_query_filt_union
