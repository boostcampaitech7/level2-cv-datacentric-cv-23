import json
import random
import os

def split_data(json_file, train_output_file, val_output_file, train_ratio=0.8):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    images = data['images']
    image_keys = list(images.keys())
    random.shuffle(image_keys)

    split_index = int(len(images) * train_ratio)
    train_keys = image_keys[:split_index]
    val_keys = image_keys[split_index:]
    
    train_images = {key: images[key] for key in train_keys}
    val_images = {key: images[key] for key in val_keys}

    train_data = {'images': train_images}
    val_data = {'images': val_images}

    with open(train_output_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)

    with open(val_output_file, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=4)


# 각 폴더의 경로
folders = ['chinese_receipt', 'japanese_receipt', 'thai_receipt', 'vietnamese_receipt']
base_dir = '/data/ephemeral/home/code/data/'

for folder in folders:
    json_file_path = os.path.join(base_dir, folder, 'ufo/train.json')
    train_output_file = os.path.join(base_dir, folder, 'ufo/train_random.json')
    val_output_file = os.path.join(base_dir, folder, 'ufo/val_random.json')

    # 파일이 존재하는지 확인 후 split_data 함수 호출
    if os.path.exists(json_file_path):
        split_data(json_file_path, train_output_file, val_output_file)
    else:
        print(f"File not found: {json_file_path}")