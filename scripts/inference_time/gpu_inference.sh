pip3 install virtualenv
python3 -m venv test
source test/bin/activate

pip3 install -r /PlanRGCN/requirements.txt
pip3 uninstall dgl -y
pip3 install ray[tune]
pip3 install jpype1
pip3 install -e /PlanRGCN/utils/
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html


old_inference () {

python3 -m trainer.inference.py \
  --prep_path "$prep_path" \
  --model_path "$model_path" \
  --config_path "$config_path" \
  --output_path "$output_path" \
  --query_path "$query_path"
}
#
# wikidata Run - GPU
prep_path="/data/wikidata_3_class_full/planRGCN_no_pred_co/prepper.pcl"
model_path="/data/wikidata_3_class_full/planRGCN_no_pred_co/best_model.pt"
config_path="/data/wikidata_3_class_full/planRGCN_no_pred_co/model_config.json"
output_path="/data/wikidata_3_class_full/test_inf/plan_GPU_inference.csv"
query_path="/data/wikidata_3_class_full/test_sampled.tsv"
old_inference

