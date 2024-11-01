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
    parser.add_argument('--inference_file', type=str, default='/data/ephemeral/home/code/predictions/output.csv')
    args = parser.parse_args()
    return args


def read_json(filename: str):
    with Path(filename).open(encoding='utf8') as handle:
        ann = json.load(handle)
    return ann

nation_dict = {
    'vi': 'vietnamese_receipt',
    'th': 'thai_receipt',
    'zh': 'chinese_receipt',
    'ja': 'japanese_receipt',
}

def get_matched(pairs):
    
    gt_matched = []
    det_matched = []
    for pair in pairs:
        if isinstance(pair['gt'], list):
            for gt_num in pair['gt']:
                gt_matched.append(gt_num)
        else: gt_matched.append(pair['gt'])

        if isinstance(pair['det'], list):
            for det_num in pair['det']:
                det_matched.append(det_num)
        else: det_matched.append(pair['det'])
    
    return gt_matched, det_matched

def unmatched_show(opt, det_data, selected_image: 'str'):

    # 선택한 이미지 데이터 처리
    name = selected_image
    det_info = det_data['images'][name]

    lang = nation_dict[name.split('.')[1]]

    # Pred와 GT bounding box 추출
    pred_bboxes_dict = {name: [det_point_v['points'] for det_point_v in det_info['words'].values()]}
    gt_bboxes_dict = {name: [gt_point_v['points'] for gt_point_v in read_json(f'{opt.data_dir}/{lang}/ufo/train.json')['images'][name]['words'].values()]}

    # deteval 계산
    resdict = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict)

    score_dict = {score: resdict['per_sample'][name][score] for score in ['precision', 'recall', 'hmean']}

    pairs = resdict['per_sample'][name]['pairs']
    
    gt_matched, det_matched = get_matched(pairs)
    
    # 매칭되지 않은 GT와 Pred bounding box 추출
    gt_unmatched = [gt_bboxes_dict[name][i] for i in range(len(gt_bboxes_dict[name])) if i not in gt_matched]
    det_unmatched = [pred_bboxes_dict[name][i] for i in range(len(pred_bboxes_dict[name])) if i not in det_matched]
    unmatched_len = [len(gt_unmatched), len(det_unmatched)]

    # 이미지 열기 및 시각화
    img_path = os.path.join(opt.data_dir, lang, "img/train", name)
    img = Image.open(img_path)
    img = ImageOps.exif_transpose(img).convert("RGB")

    fig, ax = plt.subplots(dpi=500)
    ax.imshow(img)
    ax.axis('off')

    # Bounding box 시각화
    for points in np.array(gt_unmatched):
        ax.plot(points[:,0], points[:,1], 'b', linewidth=0.4)
        ax.plot([points[-1][0], points[0][0]], [points[-1][1], points[0][1]], 'b', linewidth=0.4)
    
    for points in np.array(det_unmatched):
        ax.plot(points[:,0], points[:,1], 'r', linewidth=0.3)
        ax.plot([points[-1][0], points[0][0]], [points[-1][1], points[0][1]], 'r', linewidth=0.3)
    
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    
    return buf, score_dict, unmatched_len

def main(opt):
    st.set_page_config(layout="wide")
    
    st.title('결과 시각화😎')
    st.header('Unmatched gt & det bboxes')
    # inference file 불러오기
    det_data = read_json(opt.inference_file)
    image_files = [i for i in det_data['images'].keys()]

    if 'image_index' not in st.session_state:
        st.session_state.image_index = 0

    # 두 개의 열 생성
    col1, col2 = st.columns(2)

    with col2:
        for _ in range(2):
            st.write('\n')
        # 이미지 탐색 버튼
        search = st.columns(8)
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
        # unmatched box 이미지 출력
        buf, score_dict, unmatched_len = unmatched_show(opt, det_data, selected_image)
        st.image(buf, width=450)
    with col2:
        st.header('Statistics')
        st.markdown(f"##### Unmatched gt : {unmatched_len[0]}, Unmatched det : {unmatched_len[1]}")
        df = pd.DataFrame({"name": score_dict.keys(),"score": score_dict.values()}).set_index('name')
        st.dataframe(df)

if __name__ == '__main__':
    opt = parse_args()
    main(opt)