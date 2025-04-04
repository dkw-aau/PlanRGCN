def tablify(df_latex_str):
    if "caption" in df_latex_str:
        return (
            df_latex_str.replace("\\caption{", "\\caption*{")
            .replace(">", "$>$")
            .replace("\\begin{table}", "\\begin{table}\n\centering")
        )
