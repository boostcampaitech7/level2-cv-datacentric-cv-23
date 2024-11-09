
# ♻️ Multilingual Receipt OCR
<p align="center">
    </picture>
    <div align="center">
        <img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue">
        <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
        <img src="https://img.shields.io/badge/W&B-FFBE00.svg?style=for-the-badge&logo=weightsandbiases&logoColor=white">
        <img src="https://img.shields.io/badge/streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
        <img src="https://img.shields.io/badge/tmux-1BB91F?style=for-the-badge&logo=tmux&logoColor=white">
    </div>
    </picture>
    <div align="center">
        <img src="https://github.com/user-attachments/assets/7c6a4a88-9183-47f0-aa37-b57012021701" width="600"/>
    </div>
</p>

<br />

## ✏️ Introduction
OCR ­(Optical Character Recognition)은 문서 등의 이미지에서 글자를 인식하는 Task 입니다. OCR의 모듈로는 글자 영역을 판단하는 Text Detector, 영역에 포함된 글자를 인식하는 Text Recognizer, 자연어를 유의미한 순서로 정렬하는 Serialiser, 기정의된 key들에 대한 value 추출하는 Text Parser가 있습니다. 해당 대회는 영수증 이미지 데이터에서 글자를 검출하는 ocr 대회로 다음 두 가지 규칙이 있습니다.

1. 영수증 이미지에서 영역을 탐지하는 Text Detection만을 수행합니다.
2. 모델은 EAST로 고정하고, Data만을 수정하여 성능을 높여야 합니다.


<br />

## 📅 Schedule
프로젝트 전체 일정

- 2024.10.30 ~ 2024.11.07

프로젝트 세부일정

- 2024.10.30 ~ 2024.10.31 : 데이터 EDA 및 시각화, 평가 지표 설정 
- 2024.10.30 ~ 2024.11.03 : Wandb 연동 및 로그 추가
- 2024.11.01 ~ 2024.11.03 : 외부 데이터 조사 및 Annotation 가이드라인 설정
- 2024.11.02 ~ 2024.11.05 : 외부 데이터 Annotation 
- 2024.11.02 ~ 2024.11.07 : 모델 실험 및 평가
- 2024.11.07 ~ 2024.11.08 : 모델 앙상블 실험
- 2024.11.08 ~ 2024.11.08 : 최종 모델 평가

## 🕵️ 프로젝트 파이프라인 
밑은 나중에 변경

<img src="https://github.com/user-attachments/assets/5300dad3-8e0f-4927-ade9-241b01771e6d" width="500"/>

각 파이프라인에 대한 상세한 내용은 아래 링크를 통해 확인할 수 있습니다.

- [MLFlow 및 Wandb 연동](https://shadowed-fact-f9b.notion.site/Wandb-with-mmdection-train-8854fc9596a743ebb7ecdbb894dbd807?pvs=4)
- [데이터 EDA 및 Streamlit 시각화](https://shadowed-fact-f9b.notion.site/EDA-Streamlit-bd10bb80c7704431b27c05929899bc4e?pvs=4)
- [Validation 전략 구축](https://shadowed-fact-f9b.notion.site/Validation-d56cc4f852334249905ef1c99b05133d?pvs=4)
- [모델 실험 및 평가](https://shadowed-fact-f9b.notion.site/4287a4ea70f145739bf45738ae35051d?pvs=4)
- [모델 앙상블 실험](https://shadowed-fact-f9b.notion.site/ensemble-ca0522e34a544108a8f2b1ff66ca7ed3?pvs=4)

<br />

## 🥈 Result
Private 리더보드에서 최종적으로 아래와 같은 결과를 얻었습니다.
<img align="center" src="https://github.com/user-attachments/assets/10e71c5f-8f3c-4d66-ac4a-7469fe3198c5" width="600" height="50">

<br />

## 🗃️ Dataset Structure
```
data/
├── chinese_receipt
│   ├── img
│   │   ├── test
│   │   │   ├── extractor.zh.in_house...jpg
│   │   │   ├── extractor.zh.in_house...jpg
│   │   │   └── ...
│   │   └── train
│   │       ├── extractor.zh.in_house...jpg
│   │       ├── extractor.zh.in_house...jpg
│   │       └── ...
│   └── ufo
│       ├── test.json
│       └── train.json
├── japanese_receipt
│   ├── img
│   │   ├── test
│   │   │   ├── extractor.ja.in_house...jpg
│   │   │   ├── extractor.ja.in_house...jpg
│   │   │   └── ...
│   │   └── train
│   │       ├── extractor.ja.in_house...jpg
│   │       ├── extractor.ja.in_house...jpg
│   │       └── ...
│   └── ufo
│       ├── test.json
│       └── train.json
├── thai_receipt
│   ├── img
│   │   ├── test
│   │   │   ├── extractor.th.in_house...jpg
│   │   │   ├── extractor.th.in_house...jpg
│   │   │   ├── extractor.th.in_house...jpg
│   │   │   └── ...
│   │   └── train
│   │       ├── extractor.th.in_house...jpg
│   │       ├── extractor.th.in_house...jpg
│   │       └── ...
│   └── ufo
│       ├── test.json
│       └── train.json
└── vietnamese_receipt
    ├── img
    │   ├── test
    │   │   ├── extractor.vi.in_house...jpg
    │   │   ├── extractor.vi.in_house...jpg
    │   │   └── ...
    │   └── train
    │       ├── extractor.vi.in_house...jpg
    │       ├── extractor.vi.in_house...jpg
    │       └── ...
    └── ufo
        ├── test.json
        └── train.json

```
- 데이터셋은 다국어(중국어, 일본어, 태국어, 베트남어) 영수증 이미지로 핸드폰 카메라 등으로 찍은 영수증 사진과 스캔된 영수증 사진으로 이뤄지며, train 400개(언어별 100개), test 120개(언어별 30개)로 구성되어 있습니다.

### Train & Test json

Train json 파일은 coco format을 따르며 Info, licenses, images, categories, annotations로 구성되어 있습니다.
- Images
  ```json
    "images": [
    {
      "width": 1024,
      "height": 1024,
      "file_name": "train/0000.jpg",
      "license": 0,
      "flickr_url": null,
      "coco_url": null,
      "date_captured": "2020-12-26 14:44:23",
      "id": 0
    },
    ...
  ```
- Annotation
  ```json
      "annotations": [
    {
      "image_id": 0,
      "category_id": 0,
      "area": 257301.66,
      "bbox": [
        197.6,
        193.7,
        547.8,
        469.7
      ],
      "iscrowd": 0,
      "id": 0
    },
    ...
  ```
- Test JSON 파일은 Train JSON 파일과 동일한 구조를 가지며, 단 Annotation 정보만 빠져 있습니다.
<br />

## ⚙️ Requirements

### env.
이 프로젝트는 Ubuntu 20.04.6 LTS, CUDA Version: 12.2, Tesla v100 32GB의 환경에서 훈련 및 테스트되었습니다.

### Installment
또한, 이 프로젝트에는 다앙한 라이브러리가 필요합니다. 다음 단계를 따라 필요한 모든 라이브러리를 설치할 수 있습니다.
``` bash
  git clone https://github.com/boostcampaitech7/level2-objectdetection-cv-23.git
  cd level2-objectdetection-cv-23
  pip install -r requirements.txt
```

<br />

## 🎉 Project

### 1. Structure
  ```bash
project
├── Detectron2
│   ├── detectron2_inference.py
│   └── detectron2_train.py
├── EDA
│   ├── confusion_matrix_trash.py
│   └── Stramlit
│       ├── arial.ttf
│       ├── EDA_Streamlit.py
│       ├── EDA_Streamlit.sh
│       ├── inference_json
│       │   └── val_split_rand411_pred_latest.json
│       └── validation_json
│           └── val_split_random411.json
├── mmdetection2
│   ├── mmdetection2_inference.py
│   ├── mmdetection2_train.py
│   └── mmdetection2_val.py
├── mmdetection3
│   ├── mmdetectionV3_inference.py
│   ├── mmdetectionV3_train.py
│   └── mmdetectionV3_val.py
├── README.md
├── requirements.txt
└── src
    ├── ensemble.py
    └── make_val_dataset.ipynb
```
### 2. EDA
#### 2-1. Streamlit
Train data 및 inference 결과의 EDA을 위해 Streamlit을 활용했습니다. Streamlit을 통해 EDA를 진행하기 위해 다음을 실행하세요.
```bash
bash EDA_Streamlit.sh
```
실행을 위해 다음의 인자가 필요합니다.

  - **dataset_path** : dataset 경로
  - **font_path** : bbox의 시각화를 위한 font 경로 (우리의 Repository에 있는 arial.ttf을 이용하세요)
  - **inference_path** : inference json 파일 경로
  - **validation_path** : validation json 파일 경로
  
데모 실행을 위해 validation_json, inference_json directory에 데모 json 파일이 있습니다.

#### 2-2. confusion_matrix
Confusion matrix를 시각화하기 위해 confusion_matrix_trash.py 코드를 추가하였습니다.

해당 코드는 validation inference 시 confusion matrix도 함께 출력하기 위한 코드로 직접 실행하지 않고 val.py에서 import해 사용합니다. mmdetectionv2_val.py에서 confusion matrix를 출력하는 코드를 확인하실 수 있습니다.

mmdetectionv2_val.py를 실행하면 추론 결과를 담은 json 파일, confusion_matrix를 위한 pickel파일, confusion_matrix png파일이 함께 저장됩니다.
        
### 3. Train and inference
프로젝트를 위해 mmdetection V2 및 V3, Detectron2를 사용했습니다. 각 라이브러리에 해당하는 directory에 train과 inference를 위한 코드가 있습니다.

해당 코드들을 사용하기 위해 mmdetection 및 Detectron2 라이브러리에 포함된 config 파일이 필요합니다. 밑의 링크들을 통해 config 파일과 그에 필요한 구성 요소들을 clone할 수 있습니다.
  
- [mmdetection](https://github.com/open-mmlab/mmdetection) 
- [Detectron2](https://github.com/facebookresearch/detectron2)

[라이브러리명]_val.py 파일은 Streamlit 시각화를 위해 validation inference 결과에 대한 json 파일을 추출하는 코드입니다. Detectron2의 경우 detectron2_inference.py를 통해 json 파일을 추출할 수 있습니다. 
<br />

### 4. ensemble
앙상블을 사용하기 위해 다음을 실행하세요.
```bash
python ./src/ensemble.py
```

아래 변수 값을 수정하여 csv 파일 및 json 저장경로를 지정할 수 있습니다.
```python
root = ['*.csv',] # 앙상블을 진행할 csv 파일을 지정합니다.
submission_dir = '../../submission/' # csv 파일이 저장된 경로 및 앙상블 후 저장할 경로를 지정합니다.
annotation = '../../dataset/test.json' # 앙상블에 사용하기 위해 file의 image 정보가 포함된 json 파일 경로를 지정합니다.
```

아래 변수 값을 수정하여 앙상블 기법 및 수치를 지정할 수 있습니다.
```python
ensemble_type = '' #[nms, wbf, nmw, soft-nms] 중 사용할 앙상블 기법을 선택합니다. 
iou_thr = 0.5 #iou threshold 값을 설정합니다.

# WBF 기법 설정 값
wbf_conf_type='avg' # ['avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg'] # WBF 기법 수행 시 신뢰도 계산 방법을 설정 값입니다.
wbf_allows_overflow = False # {True: 가중치 합 > 1, False: 가중치 합 1로 고정} # 가중치 합을 1을 초과하거나 1로 고정 하는 설정 값입니다.
wbf_skip_box_thr = 0.0 # 값에 해당하는 정확도가 넘지 않으면 제외하는 설정 값입니다.

# Soft-NMS 기법 설정 값
method = 2 # 1 - linear soft-NMS, 2 - gaussian soft-NMS, 3 - standard NMS 기본값: 2  # Soft-NMS의 방식을 선택하는 설정 값입니다.
sn_sigma = 0.5 # Gaussian soft-NMS 방식 사용 시 분산을 설정하는 값입니다. 
sn_thresh = 0.001 # 값에 해당하는 신뢰도 미만의 Box를 제거하는 설정 값입니다.


weights = [1] * len(submission_df) # 각 모델의 동일한 가중치 1을 고정하는 설정 값입니다. None으로 설정 시 각 모델에 적용된 가중치로 진행됩니다. 

```

해당 코드들은 Weighted-Boxes-Fusion GitHub 내 ensemble_boxes 라이브러리가 포함되어 있습니다.
- [Weighted-Boxes-Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)  

## 🧑‍🤝‍🧑 Contributors
<div align="center">
<table>
  <tr>
    <td align="center"><a href="https://github.com/Yeon-ksy"><img src="https://avatars.githubusercontent.com/u/124290227?v=4" width="100px;" alt=""/><br /><sub><b>김세연</b></sub><br />
    </td>
        <td align="center"><a href="https://github.com/jihyun-0611"><img src="https://avatars.githubusercontent.com/u/78160653?v=4" width="100px;" alt=""/><br /><sub><b>안지현</b></sub><br />
    </td>
        <td align="center"><a href="https://github.com/dhfpswlqkd"><img src="https://avatars.githubusercontent.com/u/123869205?v=4" width="100px;" alt=""/><br /><sub><b>김상유</b></sub><br />
    </td>
        <td align="center"><a href="https://github.com/K-ple"><img src="https://avatars.githubusercontent.com/u/140207345?v=4" width="100px;" alt=""/><br /><sub><b>김태욱</b></sub><br />
    </td>
        <td align="center"><a href="https://github.com/myooooon"><img src="https://avatars.githubusercontent.com/u/168439685?v=4" width="100px;" alt=""/><br /><sub><b>김윤서</b></sub><br />
    </td>
  </tr>
</table>
</div>

## ⚡️ Detail   
프로젝트에 대한 자세한 내용은 [Wrap-Up Report](https://github.com/boostcampaitech7/level2-objectdetection-cv-23/blob/main/docs/CV_23_WrapUp_Report_detection.pdf) 에서 확인할 수 있습니다.
