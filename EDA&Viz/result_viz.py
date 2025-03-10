import os
import argparse
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from PIL import Image, ImageOps
from pathlib import Path
from deteval import calc_deteval_metrics
from io import BytesIO

def parse_args():
    parser = argparse.ArgumentParser(description='Streamlit')
    parser.add_argument('--data_dir', type=str, default='/data/ephemeral/home/code/data')
    parser.add_argument('--gt_file_name', type=str, default='val_random.json')
    parser.add_argument('--inference_file', type=str, default='/data/ephemeral/home/code/predictions/output.csv')
    args = parser.parse_args()
    return args

language_dict = {
    'zh': 'chinese',
    'ja': 'japanese',
    'th': 'thai',
    'vi': 'vietnamese',
}

def read_json(filename: str):
    with Path(filename).open(encoding='utf8') as handle:
        annotation = json.load(handle)
    return annotation

def get_bboxes_dict(data, name):
    bboxes_dict = dict()
    info = data['images'][name]
    bboxes_dict[name] = [bbox['points'] for bbox in info['words'].values()]
    return bboxes_dict

def get_matched_id(pairs):
    
    gt_matched_id = []
    pred_matched_id = []
    for pair in pairs:
        if isinstance(pair['gt'], list):
            for gt_num in pair['gt']:
                gt_matched_id.append(gt_num)
        else: gt_matched_id.append(pair['gt'])

        if isinstance(pair['det'], list):
            for pred_num in pair['det']:
                pred_matched_id.append(pred_num)
        else: pred_matched_id.append(pair['det'])
    
    return gt_matched_id, pred_matched_id

def get_buf_of_fig(img_path, gt_bboxes, pred_bboxes):

    img = Image.open(img_path)
    img = ImageOps.exif_transpose(img).convert("RGB")

    fig, ax = plt.subplots(dpi=300)
    ax.imshow(img)
    ax.axis('off')

    # Bounding box 시각화
    for points in np.array(gt_bboxes):
        ax.plot(points[:,0], points[:,1], 'b', linewidth=0.4)
        ax.plot([points[-1][0], points[0][0]], [points[-1][1], points[0][1]], 'b', linewidth=0.4)
    
    for points in np.array(pred_bboxes):
        ax.plot(points[:,0], points[:,1], 'r', linewidth=0.3)
        ax.plot([points[-1][0], points[0][0]], [points[-1][1], points[0][1]], 'r', linewidth=0.3)
    
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)

    return buf

def get_result(opt, pred_data, name: 'str'):

    lang = language_dict[name.split('.')[1]]
    gt_data = read_json(f'{opt.data_dir}/{lang}_receipt/ufo/train.json')

    # Pred와 GT bounding box 추출
    gt_bboxes_dict = get_bboxes_dict(gt_data, name)
    pred_bboxes_dict = get_bboxes_dict(pred_data, name)

    # deteval 계산
    resdict = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict)

    # resdict에서 필요한 정보 추출
    resdict_sample = resdict['per_sample'][name]
    score_dict = {score: resdict_sample[score] for score in ['precision', 'recall', 'hmean']}
    pairs = resdict_sample['pairs']

    gt_matched_id, pred_matched_id = get_matched_id(pairs)
    
    # 매칭된 GT와 Pred bounding box
    gt_matched = [gt_bboxes_dict[name][id] for id in gt_matched_id]
    pred_matched = [pred_bboxes_dict[name][id] for id in pred_matched_id]
    matched_len = [len(gt_matched), len(pred_matched)]


    # 매칭되지 않은 GT와 Pred bounding box
    gt_unmatched = [gt_bboxes_dict[name][i] for i in range(len(gt_bboxes_dict[name])) if i not in gt_matched_id]
    pred_unmatched = [pred_bboxes_dict[name][i] for i in range(len(pred_bboxes_dict[name])) if i not in pred_matched_id]
    unmatched_len = [len(gt_unmatched), len(pred_unmatched)]

    # 이미지 열기 및 시각화
    img_path = os.path.join(opt.data_dir, f'{lang}_receipt', "img/train", name)

    matched_buf = get_buf_of_fig(img_path, gt_matched, pred_matched)
    unmatched_buf = get_buf_of_fig(img_path, gt_unmatched, pred_unmatched)

    return (matched_buf, matched_len), (unmatched_buf, unmatched_len), score_dict

def main(opt):
    st.set_page_config(layout="wide")
    
    st.title('결과 시각화😎')
    st.header('Matching gt & pred bboxes')

    # inference file 불러오기
    pred_data = read_json(opt.inference_file)
    image_files = [i for i in pred_data['images'].keys()]

    if 'image_index' not in st.session_state:
        st.session_state.image_index = 0

    # 세 개의 열 생성
    col1, col2, col3 = st.columns(3)

    with col2:
        for _ in range(2):
            st.write('\n')
        # 이미지 탐색 버튼
        search = st.columns(5)
        with search[0]:
            if st.button("Previous"):
                if st.session_state.image_index > 0:
                    st.session_state.image_index -= 1
        with search[1]:
            if st.button("Next"):
                if st.session_state.image_index < len(image_files) - 1:
                    st.session_state.image_index += 1
    with col1:
        # 이미지 선택 sidebar
        selected_image = st.selectbox("이미지를 선택하세요:", image_files, index=st.session_state.image_index)
        if image_files.index(selected_image) != st.session_state.image_index:
            st.session_state.image_index = image_files.index(selected_image)
            st.rerun()
        # matched, unmatched box 이미지 출력
        matched, unmatched, score_dict = get_result(opt, pred_data, selected_image)
        st.text('Matched')
        st.image(matched[0], width=450)
    with col2:
        st.text('Unmatched')
        st.image(unmatched[0], width=450)
    with col3:
        st.header('Statistics')
        st.markdown(f"##### Ground truth")
        st.markdown(f"##### Matched : {matched[1][0]}, Unmatched : {unmatched[1][0]}")
        st.text("\n")
        st.markdown(f"##### Prediction")
        st.markdown(f"##### Matched : {matched[1][1]}, Unmatched : {unmatched[1][1]}\n")
        st.text("\n")
        df = pd.DataFrame({"name": score_dict.keys(),"score": score_dict.values()}).set_index('name')
        st.dataframe(df)

if __name__ == '__main__':
    opt = parse_args()
    main(opt)