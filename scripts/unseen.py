import importlib
import inductive_query.res_proc_helper as help
importlib.reload(help)
import inductive_query.result_processor as res_proc
importlib.reload(res_proc)
import inductive_query.utils as ih
importlib.reload(ih)
CompletelyUnseenQueryExtractor = ih.CompletelyUnseenQueryExtractor
ResultProcessor = res_proc.ResultProcessor
get_completely_unseen_r_processor = help.get_completely_unseen_r_processor
import importlib
import inductive_query.res_proc_helper as help
importlib.reload(help)
import inductive_query.result_processor as res_proc
importlib.reload(res_proc)
import inductive_query.utils as ih
importlib.reload(ih)

parse = argparse.ArgumentParser(prog="PredictionProcessor", description="Post Processing of results for data. Formatted from the Notebooks")
parse.add_argument('-s','--split_dir', help='Folder name in path to where the test_sampled.tsv file is located')
parse.add_argument('-t','--time_intervals', type=int, help='the amount of time intervals. Choices are 3, 5!')
parse.add_argument('-f','--pred', help='The prediction files')
parse.add_argument('-a', '--approach', help='the name of the results')
parser.add_argument('-c','--pred_col')
parse.add_argument('-o', '--outputfolder', default=None, help='the path where the result folder should be created')


args = parse.parse_args()

def time_ints(t):
    match t:
        case 3:
            ResultProcessor.gt_labels = [0,1,2]
            return {
                0: '0s to 1s',
                1: '1s to 10s',
                2: '$>$ 10s',
            }, snap_lat2onehotv2
        case 5:
            ResultProcessor.gt_labels = [0,1,2,3,4]
            return {
                0: '(0s; 0.004]',
                1: '(0.004s; 1]',
                2: '(1s; 10]',
                3: '(10; timeout]',
                4: 'timeout',
            }, snap5cls

if args.outputfolder == None:
    output_fold = pathlib.Path(args.pred).parent
    output_fold = os.path.join(output_fold, "results")
    os.makedirs(output_fold, exist_ok=True)
else:
    output_fold = args.outputfolder

CompletelyUnseenQueryExtractor = ih.CompletelyUnseenQueryExtractor
ResultProcessor = res_proc.ResultProcessor
get_completely_unseen_r_processor = help.get_completely_unseen_r_processor
c = CompletelyUnseenQueryExtractor(args.split_dir)
q_files = c.run()

dbpedia_base = get_completely_unseen_r_processor(args.split_dir, args.pred, os.path.join(args.split_dir, "test_sampled.tsv"), name_dict, "PlanRGCN Completely unseen", q_files,apply_cls_func=None, pred_col=args.pred_col)

with open(os.path.join(output_fold,'confusion_matrix_all_row_wise.txt'),'w') as f:
    c, t = dbpedia_base.confusion_matrix_to_latex_row_wise(name_dict=name_dict, return_sums=True, add_sums=True)
    f.write(c)
    f.write('\n')
    f.write(str(t))
dbpedia_base_df_row = dbpedia_base.confusion_matrix_to_latex_row_wise(name_dict=name_dict, return_sums=True, add_sums=False,to_latex =False)
dbpedia_base_df_row.to_csv(os.path.join(output_fold,'confusion_matrix_all_row_wise.csv'))

with open(os.path.join(output_fold,'confusion_matrix_all.txt'),'w') as f:
    f.write(dbpedia_base.confusion_matrix_to_latex(row_percentage=False,name_dict=name_dict))
dbpedia_base.confusion_matrix_to_latex(row_percentage=False,name_dict=name_dict,to_latex =False).to_csv(os.path.join(output_fold,'confusion_matrix_all.csv'))

DBpedia_PP = get_PP_result_processor(path, pred_path, split_path, name_dict, approach_name, split_type='test')
with open(os.path.join(output_fold,'confusion_matrix_PP_row_wise.txt'),'w') as f:
    c,t = DBpedia_PP.confusion_matrix_to_latex_row_wise(name_dict=name_dict, return_sums=True, add_sums=True)
    f.write(c)
    f.write('\n')
    f.write(str(t))
DBpedia_PP.confusion_matrix_to_latex_row_wise(name_dict=name_dict, return_sums=False, add_sums=False,to_latex =False).to_csv(os.path.join(output_fold,'confusion_matrix_PP_row_wise.csv'))


with open(os.path.join(output_fold,'confusion_matrix_PP.txt'),'w') as f:    
    f.write(DBpedia_PP.confusion_matrix_to_latex(row_percentage=False,name_dict=name_dict))
DBpedia_PP.confusion_matrix_to_latex(row_percentage=False,name_dict=name_dict,to_latex =False).to_csv(os.path.join(output_fold,'confusion_matrix_PP.csv'))
