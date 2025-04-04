import pathlib
from inductive_query.result_processor import ResultProcessor

from inductive_query.utils import UnseenQueryExtractor

def get_unseen_result_processor(dataset_path, pred_path, split_path, unseen_type, name_dict, approach_name,apply_cls_func=None,remove_prefix=0):
    if unseen_type != "default":
        ext = UnseenQueryExtractor(dataset_path)
        ext.set_train_pred_ents()
        ext.set_train_pred_queryIDs()
        ext.set_train_ent_queryIDs()
        ext.set_test_pred_ents()
        ext.set_test_pred_queryIDs()
        ext.set_test_ent_queryIDs()
    match unseen_type:
        case "pred":
            unseen_pred_queryID =[pathlib.Path(x).name for x in ext.get_unseen_pred_queryIds()] 
            p = ResultProcessor(pred_path, approach_name=approach_name,apply_cls_func=apply_cls_func)
            p.retain_path(split_path, remove_prefix=remove_prefix)
            p.retain_ids(unseen_pred_queryID)
            print("unseen predicate")
            print(p.confusion_matrix_to_latex_row_wise(name_dict=name_dict))
            print(p.confusion_matrix_to_latex(row_percentage=False,name_dict=name_dict))
            return p
        case "entity":
            print("unseen entity")
            unseen_ent_queryID = [pathlib.Path(x).name for x in ext.get_unseen_ent_queryIds()]
            p = ResultProcessor(pred_path, approach_name=approach_name,apply_cls_func=apply_cls_func)
            p.retain_path(split_path, remove_prefix=remove_prefix)
            p.retain_ids(unseen_ent_queryID)
            print(p.confusion_matrix_to_latex_row_wise(name_dict=name_dict))
            print(p.confusion_matrix_to_latex(row_percentage=False,name_dict=name_dict))
            return p
        case "default":
            print("No unseen")
            p = ResultProcessor(pred_path, approach_name=approach_name,apply_cls_func=apply_cls_func)
            p.retain_path(split_path, remove_prefix=remove_prefix)
            #p.retain_path(split_path, remove_prefix=20)
            print(p.confusion_matrix_to_latex_row_wise(name_dict=name_dict))
            print(p.confusion_matrix_to_latex(row_percentage=False,name_dict=name_dict))
            return p
        case _:
            print("Unseen All")
            unseen_queryID =[pathlib.Path(x).name for x in ext.get_unseen_pred_queryIds()]
            unseen_queryID.extend([pathlib.Path(x).name for x in ext.get_unseen_ent_queryIds()])
            p = ResultProcessor(pred_path, approach_name=approach_name,apply_cls_func=apply_cls_func)
            print(p.df)
            p.retain_path(split_path, remove_prefix=remove_prefix)
            print(p.df)
            p.retain_ids(unseen_queryID)
            print(p.df)
            print(p.confusion_matrix_to_latex_row_wise(name_dict=name_dict))
            print(p.confusion_matrix_to_latex(row_percentage=False,name_dict=name_dict))
            return p
            
def get_PP_result_processor(dataset_path, pred_path, split_path, unseen_type, name_dict, approach_name,apply_cls_func=None):
    ext = PPQueryExtractor(dataset_path)
    ext.set_test_pp()
        
    unseen_pred_queryID =[pathlib.Path(x).name for x in ext.get_test_PP_files()] 
    p = ResultProcessor(pred_path, approach_name=approach_name,apply_cls_func=apply_cls_func)
    p.retain_path(split_path)
    p.retain_ids(unseen_pred_queryID)
    print("unseen predicate")
    print(p.confusion_matrix_to_latex_row_wise(name_dict=name_dict))
    print(p.confusion_matrix_to_latex(row_percentage=False,name_dict=name_dict))
    return p

def get_completely_unseen_r_processor(dataset_path, pred_path, split_path, name_dict, approach_name, filt_files,apply_cls_func=None, pred_col='planrgcn_prediction', prefix=0):
    unseen_pred_queryID =[pathlib.Path(x).name for x in filt_files]
    p = ResultProcessor(pred_path, approach_name=approach_name,apply_cls_func=apply_cls_func, pred_col=pred_col)
    p.retain_path(split_path, remove_prefix=prefix)
    p.retain_ids(unseen_pred_queryID)
    print(p.confusion_matrix_to_latex_row_wise(name_dict=name_dict))
    print(p.confusion_matrix_to_latex(row_percentage=False,name_dict=name_dict))
    return p
    

def get_result_processor(prediction_path, split_path, name_dict, approach_name, apply_cls_func=None,remove_prefix=0):
    return get_unseen_result_processor(None, prediction_path, split_path, "default", name_dict, approach_name,apply_cls_func=apply_cls_func, remove_prefix=remove_prefix)

class UnseenResultProcessor:
    def __init__(self, dataset_path,prediction_path, split_path, name_dict, approach_name, apply_cls_func=None,remove_prefix=0):
        self.dataset_path = dataset_path
        self.prediction_path = prediction_path
        self.split_path = split_path
        self.name_dict = name_dict
        self.approach_name = approach_name
        self.apply_cls_func = apply_cls_func
        self.remove_prefix = remove_prefix
    
    def get_partially_unseen_all(self):
        p = get_unseen_result_processor(self.dataset_path, self.prediction_path, self.split_path, "All", self.name_dict, self.approach_name,apply_cls_func=self.apply_cls_func, remove_prefix=self.remove_prefix)
        p:ResultProcessor
        print(p.confusion_matrix_to_latex_row_wise(name_dict=self.name_dict))
        print(p.confusion_matrix_to_latex(row_percentage=False,name_dict=self.name_dict))
        print(self.approach_name)
        return p
        
    def get_partially_unseen_predicate(self):
        p = get_unseen_result_processor(self.dataset_path, self.prediction_path, self.split_path, "pred", self.name_dict, self.approach_name,apply_cls_func=self.apply_cls_func, remove_prefix=self.remove_prefix)
        p:ResultProcessor
        print(p.confusion_matrix_to_latex_row_wise(name_dict=self.name_dict))
        print(p.confusion_matrix_to_latex(row_percentage=False,name_dict=self.name_dict))
        print(self.approach_name)
        return p
        
    def get_partially_unseen_entity(self):
        p = get_unseen_result_processor(self.dataset_path, self.prediction_path, self.split_path, "entity", self.name_dict, self.approach_name,apply_cls_func=self.apply_cls_func, remove_prefix=self.remove_prefix)
        p:ResultProcessor
        print(p.confusion_matrix_to_latex_row_wise(name_dict=self.name_dict))
        print(p.confusion_matrix_to_latex(row_percentage=False,name_dict=self.name_dict))
        print(self.approach_name)
        return p
