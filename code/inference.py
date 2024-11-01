import os
import os.path as osp
import json
from argparse import ArgumentParser
from glob import glob

import torch
import cv2
from torch import cuda
from model import EAST
from tqdm import tqdm

from detect import detect
from deteval import calc_deteval_metrics
from TIoUeval import calc_tioueval_metrics

import pandas as pd


CHECKPOINT_EXTENSIONS = ['.pth', '.ckpt']
LANGUAGE_LIST = ['chinese', 'japanese', 'thai', 'vietnamese']

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', default=os.environ.get('SM_CHANNEL_EVAL', 'data'))
    parser.add_argument('--model_dir', default=os.environ.get('SM_CHANNEL_MODEL', 'trained_models'))
    parser.add_argument('--output_dir', default=os.environ.get('SM_OUTPUT_DATA_DIR', 'predictions'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--input_size', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=5)

    parser.add_argument('--json_name', type=str, default='val_random')
    parser.add_argument('--img_dir', type=str, default='train')

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

# json 파일에 있는 jpg에 따라 inference
def load_images_from_json(data_dir):
    image_list = []
    gt_bboxes_dict = {}
    for lang in LANGUAGE_LIST:
        json_path = osp.join(data_dir, f"{lang}_receipt/ufo/{args.json_name}.json")
        with open(json_path, 'r') as f:
            data = json.load(f)
        image_list.extend(list(data["images"].keys()))
        gt_bboxes_dict.update(extract_bboxes_dict(data))
    return image_list, gt_bboxes_dict

def do_inference(model, ckpt_fpath, data_dir, input_size, batch_size, split='test'):
    model.load_state_dict(torch.load(ckpt_fpath, map_location='cpu'))
    model.eval()

    image_fnames, by_sample_bboxes = [], []
    images = []

    image_list, gt_bboxes_dict = load_images_from_json(data_dir)

    for image_fname in tqdm(image_list):
        for lang in LANGUAGE_LIST:
            image_fpath = osp.join(data_dir, f'{lang}_receipt/img/{split}/{image_fname}')
            if osp.exists(image_fpath):
                break

        image_fnames.append(image_fname)
        images.append(cv2.imread(image_fpath)[:, :, ::-1])
        if len(images) == batch_size:
            by_sample_bboxes.extend(detect(model, images, input_size))
            images = []

    if len(images):
        by_sample_bboxes.extend(detect(model, images, input_size))

    ufo_result = dict(images=dict())
    for image_fname, bboxes in zip(image_fnames, by_sample_bboxes):
        words_info = {idx: dict(points=bbox.tolist()) for idx, bbox in enumerate(bboxes)}
        ufo_result['images'][image_fname] = dict(words=words_info)

    return ufo_result, gt_bboxes_dict

def extract_bboxes_dict(json_data):
    bboxes_dict = {}
    for image_id, image_info in json_data["images"].items():
        bboxes = []
        for word_info in image_info["words"].values():
            points = word_info["points"]
            bboxes.append(points)
        bboxes_dict[image_id] = bboxes
    return bboxes_dict

def get_language_json_path(image_name):
    """이미지 이름에 따른 언어별 JSON 파일 경로 반환"""
    base_path = args.data_dir
    if 'zh' in image_name:
        return os.path.join(base_path, 'chinese_receipt/ufo/val_random.json')
    elif 'ja' in image_name:
        return os.path.join(base_path, 'japanese_receipt/ufo/val_random.json')
    elif 'th' in image_name:
        return os.path.join(base_path, 'thai_receipt/ufo/val_random.json')
    elif 'vi' in image_name:
        return os.path.join(base_path, 'vietnamese_receipt/ufo/val_random.json')
    return None

def process_data(output_path):
    """output.csv와 train.json 파일들을 처리하여 태그별 평가 결과 반환"""
    with open(output_path, 'r', encoding='utf-8') as f:
        output_data = json.load(f)
    
    results = []
    for image_name, image_data in output_data['images'].items():
        # 언어 식별
        lang_code = None
        if 'zh' in image_name:
            lang_code = 'chinese'
        elif 'ja' in image_name:
            lang_code = 'japanese'
        elif 'th' in image_name:
            lang_code = 'thai'
        elif 'vi' in image_name:
            lang_code = 'vietnamese'
        
        if lang_code:
            # GT JSON 파일 읽기
            gt_json_path = get_language_json_path(image_name)
            if gt_json_path and os.path.exists(gt_json_path):
                with open(gt_json_path, 'r', encoding='utf-8') as f:
                    gt_data = json.load(f)
                
                # 이미지별 태그 기준으로 bbox 분류
                gt_tags_bbox = {}
                if image_name in gt_data['images']:
                    image_tags = gt_data['images'][image_name].get('tags', [])
                    if image_tags:
                        main_tag = image_tags[0]  # 첫 번째 태그 사용
                        
                        # 해당 이미지의 모든 bbox를 메인 태그로 분류
                        gt_boxes = []
                        for word in gt_data['images'][image_name].get('words', {}).values():
                            points = word.get('points', [])
                            if points:
                                gt_boxes.append(points)
                        
                        if gt_boxes:
                            gt_tags_bbox[main_tag] = gt_boxes
                
                # 예측된 bbox
                pred_bboxes = []
                for word in image_data.get('words', {}).values():
                    points = word.get('points', [])
                    if points:
                        pred_bboxes.append(points)
                
                # 태그별로 평가 수행
                for tag, gt_boxes in gt_tags_bbox.items():
                    # DetEval 메트릭 계산
                    eval_result = calc_deteval_metrics(
                        {image_name: pred_bboxes},
                        {image_name: gt_boxes}
                    )
                    
                    results.append({
                        'language': lang_code,
                        'tag': tag,
                        'precision': eval_result['total']['precision'],
                        'recall': eval_result['total']['recall'],
                        'hmean': eval_result['total']['hmean']
                    })
    
    return pd.DataFrame(results)

def calculate_tag_scores(df):
    """태그별 평균 점수 계산"""
    metrics = ['precision', 'recall', 'hmean']
    results = {}
    
    # 모든 태그에 대해 평균 계산
    for tag in df['tag'].unique():
        tag_mask = (df['tag'] == tag)
        tag_scores = {}
        for metric in metrics:
            score = df[tag_mask][metric].mean()
            tag_scores[metric] = round(float(score), 4) if not pd.isna(score) else 0.0
        results[tag] = tag_scores
            
    return results

def main(args):
    # Initialize model
    model = EAST(pretrained=False).to(args.device)

    # Get paths to checkpoint files
    ckpt_fpath = osp.join(args.model_dir, 'latest.pth')

    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print('Inference in progress')

    ufo_result = dict(images=dict())
    split_result, gt_bboxes_dict = do_inference(model, ckpt_fpath, args.data_dir, args.input_size,
                                args.batch_size, split='train')
    ufo_result['images'].update(split_result['images'])

    # 예측 바운딩 박스 딕셔너리 생성
    pred_bboxes_dict = extract_bboxes_dict(ufo_result)

    output_fname = 'output.csv'
    with open(osp.join(args.output_dir, output_fname), 'w') as f:
        json.dump(ufo_result, f, indent=4)

    # Det eval 출력
    results = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict)
    print('--------------------DetEval------------------------')
    print("Overall Precision:", results['total']['precision'])
    print("Overall Recall:", results['total']['recall'])
    print("Overall Hmean:", results['total']['hmean'])


    # 태그별 평가 결과 계산
    output_df = process_data(osp.join(args.output_dir, output_fname))
    scores = calculate_tag_scores(output_df)

    # 결과 출력
    for tag, metrics in scores.items():
        print(f"\n{tag} 평가 결과:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-score: {metrics['hmean']:.4f}")


    results_tiou = calc_tioueval_metrics(pred_bboxes_dict, gt_bboxes_dict)
    print('---------------------TIoU-----------------------')
    print("Overall Precision:", results_tiou['total']['precision'])
    print("Overall Recall:", results_tiou['total']['recall'])
    print("Overall Hmean:", results_tiou['total']['hmean'])


if __name__ == '__main__':
    args = parse_args()
    main(args)
