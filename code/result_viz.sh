data_dir="/data/ephemeral/home/code/data" 
inference_file="/data/ephemeral/home/code/predictions/output.csv" 

streamlit run result_viz.py -- --data_dir "$data_dir" --inference_file "$inference_file"