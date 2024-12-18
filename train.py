import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from src import SceneTextDataset
from model import EAST

import wandb


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', 'data'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=1)
    
    # 사전학습된 모델 파일을 사용할 경우 : --pretrained 를 True로 설정해주세요.
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--pretrained_path', type=str, default='trained_models/epoch_30.pth')
    
    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, 
                pretrained, pretrained_path):

    # wandb config 
    wandb_config = {
        'learning_rate': learning_rate,
        'max_epoch' : max_epoch,
        'batch_size' : batch_size,
        'image_size' : image_size,
    }
    
    # wandb run : name, notes, tags -> 실험 시 수정 
    run = wandb.init(
        project="lv2-OCR",
        entity="cv23-lv2-ocr",
        name="expriment_wandb_artifact2",
        notes="experiment using wandb artifact",
        tags=["wandb", "artifact"],
        config=wandb_config,
    )
    
    # wandb : dataset json 파일 artifact에 추가 
    dataset_artifact = wandb.Artifact(
        name="dataset_annotation",
        type="dataset",
        description="Dataset annotation for training"
    )
    
    annotation_files = [
        'train_random.json',
        'val_random.json'
    ]
    
    # 각 언어 폴더에서 annotation 파일 추가
    languages = ['chinese', 'japanese', 'thai', 'vietnamese']
    
    for lang in languages:
        ufo_dir = os.path.join(data_dir, f'{lang}_receipt', 'ufo')
        if os.path.exists(ufo_dir):
            for annotation_file in annotation_files:
                file_path = os.path.join(ufo_dir, annotation_file)
                if os.path.exists(file_path):
                    dataset_artifact.add_file(file_path, name=f"{lang}_receipt/ufo/{annotation_file}")
    
    # artifact 로깅
    wandb.log_artifact(dataset_artifact)
    
    train_dataset = SceneTextDataset(
        data_dir,
        lang_list=['chinese', 'japanese', 'thai', 'vietnamese', 'cord', 'sroie'],
        split='train_random', # 여기서 json 파일 이름 넣어주시면 됩니다
        image_size=image_size,
        crop_size=input_size,
    )
    train_dataset = EASTDataset(train_dataset)
    num_batches = math.ceil(len(train_dataset) / batch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # Validation dataset
    val_dataset = SceneTextDataset(
        data_dir,
        lang_list=['chinese', 'japanese', 'thai', 'vietnamese'],
        split='val_random',
        image_size=image_size,
        crop_size=input_size,
    )
    val_dataset = EASTDataset(val_dataset)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    
    if pretrained:
        if osp.exists(pretrained_path):
            pretrained_weights = torch.load(pretrained_path, map_location=device)
            model.load_state_dict(pretrained_weights, strict=False)
            print('사전학습된 가중치를 성공적으로 불러왔습니다.')
        else:
            print('사전학습된 가중치 파일을 찾을 수 없습니다.')
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    model.train()
    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        cls_loss_sum, angle_loss_sum, iou_loss_sum = 0, 0, 0
        
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val
                

                cls_loss = extra_info.get('cls_loss', 0)
                angle_loss = extra_info.get('angle_loss', 0)
                iou_loss = extra_info.get('iou_loss', 0)
                
                cls_loss_sum += cls_loss if cls_loss is not None else 0
                angle_loss_sum += angle_loss if angle_loss is not None else 0
                iou_loss_sum += iou_loss if iou_loss is not None else 0
                
                # wandb log batch metrics
                wandb.log({
                    'train/batch_loss': loss_val,
                })

                pbar.update(1)
                val_dict = {
                    'Cls loss': cls_loss if cls_loss is not None else 0, 
                    'Angle loss': angle_loss if angle_loss is not None else 0,
                    'IoU loss': iou_loss if iou_loss is not None else 0
                }
                pbar.set_postfix(val_dict)

        scheduler.step()
        
        # wandb log epoch metrics
        mean_epoch_loss = epoch_loss / num_batches
        mean_cls_loss = cls_loss_sum / num_batches
        mean_angle_loss = angle_loss_sum / num_batches
        mean_iou_loss = iou_loss_sum / num_batches
        
        wandb.log({
            'epoch': epoch + 1,
            'train/total_loss': mean_epoch_loss,
            'train/cls_loss': mean_cls_loss,
            'train/angle_loss': mean_angle_loss,
            'train/iou_loss': mean_iou_loss,
            'elapsed_time': time.time() - epoch_start
        })

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            mean_epoch_loss, timedelta(seconds=time.time() - epoch_start)))
        
        if (epoch + 1) % 1 == 0: # 몇 epoch 마다 val 평가
            val_batches = len(val_loader)
            with tqdm(total=val_batches) as pbar:
                pbar.set_description('[Val {}]'.format(epoch + 1))
                model.eval()
                val_loss = [0, 0, 0, 0] # total loss, cls loss, angle loss, iou loss
                with torch.no_grad():
                    for img, gt_score_map, gt_geo_map, roi_mask, in val_loader:
                        loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                        val_loss[0] += loss.item()
                        
                        # None 체크 추가
                        cls_loss = extra_info.get('cls_loss', 0)
                        angle_loss = extra_info.get('angle_loss', 0)
                        iou_loss = extra_info.get('iou_loss', 0)
                        
                        val_loss[1] += cls_loss if cls_loss is not None else 0
                        val_loss[2] += angle_loss if angle_loss is not None else 0
                        val_loss[3] += iou_loss if iou_loss is not None else 0
                        
                        pbar.update(1)
                
                mean_val_loss = [v / val_batches for v in val_loss]
                print(
                    f'Validation Loss after Epoch {epoch + 1}: '
                    f'Total Loss: {mean_val_loss[0]:.4f}, '
                    f'Cls Loss: {mean_val_loss[1]:.4f}, '
                    f'Angle Loss: {mean_val_loss[2]:.4f}, '
                    f'IoU Loss: {mean_val_loss[3]:.4f}'
                )
                
                wandb.log({
                    'epoch': epoch + 1,
                    'val/total_loss': mean_val_loss[0],
                    'val/cls_loss': mean_val_loss[1],
                    'val/angle_loss': mean_val_loss[2],
                    'val/iou_loss': mean_val_loss[3],
                })
                
                model.train()

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, f'epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), ckpt_fpath)
            
            # wandb : log model checkpoint 
            wandb.save(ckpt_fpath)
            # wandb artifact로도 저장
            artifact = wandb.Artifact(
                name=f"model-{run.id}", 
                type="model",
                description=f"Model checkpoint from epoch {epoch + 1}"
            )
            artifact.add_file(ckpt_fpath)
            wandb.log_artifact(artifact)
    wandb.finish()


def main(args):
    do_training(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)