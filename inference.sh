data_dir='/data/ephemeral/home/data'
model_dir='/data/ephemeral/home/cv-23/code/trained_models'
output_dir='/data/ephemeral/home/cv-23/code/outputs'
json_name='val_random' # 추론하고 싶은 json 파일의 이름을 적어주세요
img_dir='train' # 실제 이미지가 있는 img 디렉토리 하위 폴더 (test, train 등)을 적어주세요

run_id='x1il0w3g' # 추론하고 싶은 wandb run id를 적어주세요. wandb UI에서 overview 탭에서 확인할 수 있어요.

python inference.py --data_dir "$data_dir" --model_dir "$model_dir" \
                    --json_name "$json_name" --img_dir "$img_dir" \
                    --run_id "$run_id"