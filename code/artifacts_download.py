import os.path as osp
from typing import Tuple
import wandb

def download_artifacts(
    project: str,
    entity: str,
    run_id: str,
    model_filename: str = 'epoch_30.pth'
) -> Tuple[str, str]:
    """
    WandB에서 모델과 데이터셋 아티팩트 다운로드
    
    model_filename: 다운로드 하고 싶은 모델 파일 이름

    Returns:
        model_path: 다운로드된 모델 파일 경로
        anno_dir: 다운로드된 데이터셋 디렉토리 경로
    """
    model_artifact_name=f"{entity}/{project}/model-{run_id}:latest"
    data_artifact_name=f"{entity}/{project}/dataset_annotation:latest"
    
    # wandb 초기화
    run = wandb.init(
        project=project,
        entity=entity,
        id=run_id,
        resume="must"
    )
    
    try:
        # 모델 아티팩트 다운로드
        model_artifact = wandb.use_artifact(model_artifact_name, type='model')
        model_dir = model_artifact.download()
        model_path = osp.join(model_dir, model_filename)
        
        # 데이터셋 아티팩트 다운로드
        data_artifact = wandb.use_artifact(data_artifact_name, type='dataset')
        anno_dir = data_artifact.download()
        
        return model_path, anno_dir
        
    finally:
        # wandb 종료
        wandb.finish()
        
model_path, anno_dir = download_artifacts(
    project="lv2-OCR",
    entity="cv23-lv2-ocr",
    run_id="bcm6tz5j", # 다운로드 하고 싶은 run_id 작성해주셔야 합니다. wandb UI에서 overview 탭에서 확인할 수 있어요. 
    model_filename='epoch_1.pth', # 다운로드 하고 싶은 모델 파일 이름도 작성해주셔야 합니다.
)

# 다운로드된 경로 출력
print(f"모델 경로: {model_path}")
print(f"데이터셋 경로: {anno_dir}")