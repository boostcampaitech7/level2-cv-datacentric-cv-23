'''baseline 모델에 사용되는 train과 validation 데이터셋에 언어 태그를 추가하는 스크립트'''

import json
import os
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default=os.environ.get('SM_CHANNEL_EVAL', 'data'))
    args = parser.parse_args()
    return args

def add_language_tags(json_path):
    # 언어 태그 매핑
    lang_tags = {
        'zh': 'china',
        'ja': 'japan',
        'th': 'thailand', 
        'vi': 'vietnam'
    }
    
    # JSON 파일 읽기
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 각 이미지에 대해 언어 태그 추가
    for img_name, img_info in data['images'].items():
        # 파일명에서 언어 코드 추출
        if 'zh' in img_name:
            lang_code = 'zh'
        elif 'ja' in img_name:
            lang_code = 'ja'
        elif 'th' in img_name:
            lang_code = 'th'
        elif 'vi' in img_name:
            lang_code = 'vi'
        else:
            continue
            
        # 해당 언어 태그가 있으면 추가
        if lang_code in lang_tags:
            if 'tags' not in img_info:
                img_info['tags'] = []
            img_info['tags'].append(lang_tags[lang_code])
    
    # 수정된 JSON 파일 저장
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def main():
    # args 파싱
    args = parse_args()
    # train과 validation 데이터셋 경로 
    base_path = args.data_dir
    train_paths = [
        os.path.join(base_path, 'chinese_receipt/ufo/train_random.json'),
        os.path.join(base_path, 'japanese_receipt/ufo/train_random.json'),
        os.path.join(base_path, 'thai_receipt/ufo/train_random.json'),
        os.path.join(base_path, 'vietnamese_receipt/ufo/train_random.json')
    ]
    
    val_paths = [
        os.path.join(base_path, 'chinese_receipt/ufo/val_random.json'),
        os.path.join(base_path, 'japanese_receipt/ufo/val_random.json'), 
        os.path.join(base_path, 'thai_receipt/ufo/val_random.json'),
        os.path.join(base_path, 'vietnamese_receipt/ufo/val_random.json')
    ]
    
    # 각 데이터셋에 태그 추가
    for path in train_paths + val_paths:
        if os.path.exists(path):
            add_language_tags(path)
    
    print('언어 태그 추가가 완료되었습니다.')

if __name__ == '__main__':
    main()
