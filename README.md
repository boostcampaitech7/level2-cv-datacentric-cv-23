
# 📜 Multilingual Receipt OCR
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
        <img src="https://github.com/user-attachments/assets/eaeefa98-f5b7-4be0-bd7c-723e22380b6f" width="600"/>
        (이미지 출처 : https://www.ncloud.com/product/aiService/ocr)
    </div>
</p>

<br />

## ✏️ Introduction
OCR ­(Optical Character Recognition)은 문서 등의 이미지에서 글자를 인식하는 Task 입니다. OCR의 모듈로는 글자 영역을 판단하는 Text Detector, 영역에 포함된 글자를 인식하는 Text Recognizer, 자연어를 유의미한 순서로 정렬하는 Serialiser, 기정의된 key들에 대한 value 추출하는 Text Parser가 있습니다. 해당 대회는 영수증 이미지 데이터에서 글자를 검출하는 OCR 대회로 다음 두 가지 규칙이 있습니다.

1. 영수증 이미지에서 영역을 탐지하는 Text Detection만을 수행합니다.
2. 모델은 EAST로 고정하고, Data만을 수정하여 성능을 높여야 합니다.

대회는 영역 탐지 성능을 평가하는 DetEval을 통해 평가됩니다.

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

Train json 파일은 UFO format을 따르며 paragraphs, words, characters, image width & height, image tag, annotation log, license tag 등으로 구성되어 있습니다.

```json
{
    "images": {
        "extractor.ja.in_house.appen_000911_page0001.jpg": {
            "paragraphs": {},
            "words": {
                "0001": {
                    "transcription": "LAWSON",
                    "points": [
                        [
                            728.553329490908,
                            235.436074339687
                        ],
                        [
                            735.9530472990916,
                            428.81846937705836
                        ],
                        [
                            216.57843990309,
                            340.653026763541
                        ],
                        [
                            220.926247854489,
                            246.740375013324
                        ]
                    ]
                },
                "chars": {},
                "img_w": 960,
                "img_h": 1280,
                "num_patches": null,
                "tags": [],
                "relations": {},
                "annotation_log": {
                    "worker": "worker",
                    "timestamp": "2024-06-07",
                    "tool_version": "",
                    "source": null
                },
                "license_tag": {
                    "usability": true,
                    "public": false,
                    "commercial": true,
                    "type": null,
                    "holder": "Upstage"
                }
            }
        },
        ...
    }
```
- point는 각 라벨의 위치 좌표이며, 글자를 읽는 방향의 왼쪽 위에서부터 시계 방향으로 x,y 좌표로 총 4개의 (x,y) 좌표로 구성되어 있습니다.


Test JSON 파일은 Train JSON 파일과 동일한 구조를 가지며, 단 point 정보만 빠져 있습니다.

<br />

## 🎉 Project

### 1. Structure
```bash
project
├── EDA&Viz
│   ├── eda.ipynb
│   ├── result_viz.py
│   └── result_viz.sh
├── inference.py
├── inference.sh
├── preprocessing
│   ├── COCO2UFO.py
│   ├── CORD2UFO.ipynb
│   ├── SROIE2UFO.ipynb
│   └── UFO2COCO.py
├── README.md
├── requirements.txt
├── src
│   ├── dataset_CV2.py
│   ├── dataset.py
│   ├── deteval.py
│   ├── __init__.py
│   └── TIoUeval.py
├── train.py
└── utils
    ├── artifacts_download.py
    ├── bbox_check.py
    ├── create_train_val_tag.py
    └── create_val_data.py
```
EDA&Viz : 데이터 분석 및 시각화를 위한 디렉토리입니다.
- **eda.ipynb** : 이미지 크기 분포, 단어 개수 분포, Bounding box 크기 분포, Aspect Ratio 분포, 예시 이미지 등을 확인할 수 있습니다.
- **result_viz.py**: 최종 모델 평가 결과를 시각화하는 코드입니다. 해당 코드를 위해 다음을 실행해야 합니다.
  ```bash
  bash result_viz.sh
  ```
preprocessing : 데이터 전처리를 위한 디렉토리입니다.
- **COCO2UFO.py** : COCO format의 데이터를 UFO format의 데이터로 변환하는 코드입니다.
- **CORD2UFO.ipynb** : CORD format의 데이터를 UFO format의 데이터로 변환하는 코드입니다.
- **SROIE2UFO.ipynb** : SROIE format의 데이터를 UFO format의 데이터로 변환하는 코드입니다.
- **UFO2COCO.py** : UFO format의 데이터를 COCO format의 데이터로 변환하는 코드입니다.
src : 모델 학습 및 추론을 위한 디렉토리입니다.
- **dataset.py** : 데이터 로더를 정의하는 코드입니다. (PIL)
- **dataset_CV2.py** : 데이터 로더를 정의하는 코드입니다. (CV2)
- **deteval.py** : DetEval을 계산하는 코드입니다.
- **TIoUeval.py** : TIoU를 계산하는 코드입니다.

utils : 유틸리티 코드를 정의하는 디렉토리입니다.
- **artifacts_download.py** : Wandb에 저장된 아티팩트를 다운로드하는 코드입니다.
- **bbox_check.py** : Bounding box 체크를 위한 코드입니다.
- **create_train_val_tag.py** : Train & Validation json 파일에 태그를 생성하는 코드입니다.
- **create_val_data.py** : Validation 데이터를 생성하는 코드입니다. (8 : 2 비율)

<br />

## ⚙️ Requirements

### env.
이 프로젝트는 Ubuntu 20.04.6 LTS, CUDA Version: 12.2, Tesla v100 32GB의 환경에서 훈련 및 테스트되었습니다.

### Installment
또한, 이 프로젝트에는 다앙한 라이브러리가 필요합니다. 다음 단계를 따라 필요한 모든 라이브러리를 설치할 수 있습니다.
``` bash
  git clone https://github.com/boostcampaitech7/level2-cv-datacentric-cv-23.git
  cd level2-datacentric-cv-23
  pip install -r requirements.txt
```

<br />

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
