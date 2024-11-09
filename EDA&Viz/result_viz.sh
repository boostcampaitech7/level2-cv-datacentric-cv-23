data_dir="/data/ephemeral/home/code/data" 
gt_file_name="val_random.json"
inference_file="/data/ephemeral/home/code/predictions/output.csv" 

streamlit run result_viz.py -- --data_dir "$data_dir" --gt_file_name "$gt_file_name" --inference_file "$inference_file"