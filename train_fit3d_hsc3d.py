import argparse
import os
import json

import numpy as np
import pkg_resources
import torch
import wandb
import torch.nn as nn
import torch.nn.init as init
from torch import optim
from torch.optim.lr_scheduler import StepLR ,CosineAnnealingLR
from tqdm import tqdm

from loss.pose3d import loss_mpjpe, n_mpjpe, loss_velocity, loss_limb_var, loss_limb_gt, loss_angle, loss_angle_velocity
from loss.pose3d import jpe as calculate_jpe
from loss.pose3d import p_mpjpe as calculate_p_mpjpe
from loss.pose3d import mpjpe as calculate_mpjpe
from loss.pose3d import acc_error as calculate_acc_err
from data.const import H36M_JOINT_TO_LABEL, H36M_UPPER_BODY_JOINTS, H36M_LOWER_BODY_JOINTS, H36M_1_DF, H36M_2_DF, H36M_3_DF
from data.reader.fit3d_hsc3d_dataset import FitHscDataset3D
from utils.data import flip_data
from utils.tools import set_random_seed, get_config, print_args, create_directory_if_not_exists
from torch.utils.data import DataLoader

from utils.learning import load_model, AverageMeter, decay_lr_exponentially
from utils.tools import count_param_numbers
from utils.data import Augmenter2D

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/fit3d_hsc3d/MotionAGFormer-base.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('--new-checkpoint', type=str, metavar='PATH', default='checkpoint', help='new checkpoint directory')
    parser.add_argument('--checkpoint-file', type=str, help="checkpoint file name")
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('--num-cpus', default=16, type=int, help='Number of CPU cores')
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--wandb-name', default=None, type=str)
    parser.add_argument('--wandb-run-id', default=None, type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    opts = parser.parse_args()
    return opts

def save_metrics(epoch, metrics, file_path):
    #성능 지표를 JSON 파일로 저장
    with open(file_path, 'a') as f:
        json.dump({'epoch': epoch, **metrics}, f)
        f.write('\n')

def train_one_epoch(args, model, train_loader, optimizer, device, losses):
    model.train()
    for x, y in tqdm(train_loader):
        batch_size = x.shape[0]
        x, y = x.to(device), y.to(device)

        # 모델의 예측
        pred = model(x)  # (N, T, 17, 3)
        if torch.isnan(pred).any(): continue
        optimizer.zero_grad()

        # 손실 함수 계산
        loss_3d_pos = loss_mpjpe(pred, y)
        loss_3d_scale = n_mpjpe(pred, y)
        loss_3d_velocity = loss_velocity(pred, y)
        loss_lv = loss_limb_var(pred)
        loss_lg = loss_limb_gt(pred, y)
        loss_a = loss_angle(pred, y)
        loss_av = loss_angle_velocity(pred, y)

        loss_total = loss_3d_pos + \
                    args.lambda_scale * loss_3d_scale + \
                    args.lambda_3d_velocity * loss_3d_velocity + \
                    args.lambda_lv * loss_lv + \
                    args.lambda_lg * loss_lg + \
                    args.lambda_a * loss_a + \
                    args.lambda_av * loss_av

        # 손실 업데이트
        losses['3d_pose'].update(loss_3d_pos.item(), batch_size)
        losses['3d_scale'].update(loss_3d_scale.item(), batch_size)
        losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size)
        losses['lv'].update(loss_lv.item(), batch_size)
        losses['lg'].update(loss_lg.item(), batch_size)
        losses['angle'].update(loss_a.item(), batch_size)
        losses['angle_velocity'].update(loss_av.item(), batch_size)
        losses['total'].update(loss_total.item(), batch_size)

        # 기울기 계산 및 최적화
        loss_total.backward()
        optimizer.step()

def evaluate(args, model, test_loader, device):
    print("[INFO] Evaluation")
    model.eval()
    mpjpe_all, p_mpjpe_all = AverageMeter(), AverageMeter()
    with torch.no_grad():
        for x, y, indices in tqdm(test_loader):
            batch_size = x.shape[0]
            x = x.to(device)

            if args.flip:
                batch_input_flip = flip_data(x)
                predicted_3d_pos_1 = model(x)
                predicted_3d_pos_flip = model(batch_input_flip)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)  # Flip back
                predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2
            else:
                predicted_3d_pos = model(x)
            if args.root_rel:
                predicted_3d_pos[:, :, 0, :] = 0  # [N,T,17,3]
            else:
                y[:, 0, 0, 2] = 0

            predicted_3d_pos = predicted_3d_pos.detach().cpu().numpy()
            y = y.cpu().numpy()

            denormalized_predictions = []
            for i, prediction in enumerate(predicted_3d_pos):
                prediction = test_loader.dataset.denormalize(prediction, indices[i].item(), is_3d=True)
                denormalized_predictions.append(prediction[None, ...])
            denormalized_predictions = np.concatenate(denormalized_predictions)

            # Root-relative Errors
            predicted_3d_pos = denormalized_predictions - denormalized_predictions[...,  0:1, :]
            y = y - y[..., 0:1, :]

            mpjpe = calculate_mpjpe(predicted_3d_pos, y)
            # p_mpjpe = calculate_p_mpjpe(predicted_3d_pos, y)
            mpjpe_all.update(mpjpe, batch_size)
            # p_mpjpe_all.update(p_mpjpe, batch_size)
    print(f"Protocol #1 error (MPJPE): {mpjpe_all.avg} mm")
    #print(f"Protocol #2 error (P-MPJPE): {p_mpjpe_all.avg} mm")
    return mpjpe_all.avg#, p_mpjpe_all.avg

def save_checkpoint(checkpoint_path, epoch, lr, optimizer, model, min_mpjpe, wandb_id):
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'min_mpjpe': min_mpjpe,
        'wandb_id': wandb_id,
    }, checkpoint_path)

def train(args, opts):
    print_args(args)
    create_directory_if_not_exists(opts.new_checkpoint)
    
    keypoints_path = 'data/keypoints'
    train_dataset = FitHscDataset3D(keypoints_path, 'train')
    test_dataset = FitHscDataset3D(keypoints_path, 'val')

    common_loader_params = {
        'batch_size': args.batch_size,
        'num_workers': opts.num_cpus - 1,
        'pin_memory': True,
        'prefetch_factor': (opts.num_cpus - 1) // 3,
        'persistent_workers': True
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_params)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_params)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args)
    # if torch.cuda.is_available():
    #     model = torch.nn.DataParallel(model)
    model.to(device)

    n_params = count_param_numbers(model)
    print(f"[INFO] Number of parameters: {n_params:,}")

    lr = args.learning_rate
    # AdmaW ---> Adam ---> SGD
    optimizer = optim.SGD([param for param in model.parameters() if param.requires_grad],
                            lr=lr,
                            weight_decay=args.weight_decay)
    # lr_decay = args.lr_decay

    # scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=args.eta_min)

    epoch_start = 0
    min_mpjpe = float('inf')  # Used for storing the best model
    wandb_id = opts.wandb_run_id if opts.wandb_run_id is not None else wandb.util.generate_id()
    
    if opts.checkpoint:
        checkpoint_path = os.path.join(opts.checkpoint, opts.checkpoint_file if opts.checkpoint_file else "latest_epoch.pth.tr")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            
            # 키 조정: 'module.' 접두사 제거
            new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
            
            model.load_state_dict(new_state_dict, strict=True)

            if opts.resume:
                # 기타 체크포인트 정보 로드
                lr = checkpoint['lr']
                epoch_start = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                min_mpjpe = checkpoint['min_mpjpe']
                if 'wandb_id' in checkpoint and opts.wandb_run_id is None:
                    wandb_id = checkpoint['wandb_id']
        else:
            print("[WARN] Checkpoint path is empty. Starting from the beginning")
            opts.resume = False
    
    # 사전 학습 모델의 앞단 레이어의 가중치 고정
    for name, param in model.named_parameters():
        if not (name.startswith('layers.12') or name.startswith('layers.13') 
                or name.startswith('layers.14') or name.startswith('layers.15')
                or name == 'rep_logit' or name == 'head' 
                or name == 'rep_logit.fc' or name == 'rep_logit.act'):
            param.requires_grad = False
    
    # 가중치 초기화
    # for name, module in model.named_modules():
    # # 업데이트 되는 레이어의 가중치만 초기화
    #     if (name.startswith('layers.12') or name.startswith('layers.13') 
    #         or name.startswith('layers.14') or name.startswith('layers.15')
    #         or name == 'rep_logit' or name == 'head' 
    #         or name == 'rep_logit.fc' or name == 'rep_logit.act'):
    #         if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
    #             init.kaiming_normal_(module.weight, nonlinearity='relu')
    #             if module.bias is not None:
    #                 init.constant_(module.bias, 0)

    if not opts.eval_only:
        if opts.resume:
            if opts.use_wandb:
                wandb.init(id=wandb_id,
                        project='MotionMetaFormer',
                        resume="must",
                        settings=wandb.Settings(start_method='fork'))
        else:
            print(f"Run ID: {wandb_id}")
            if opts.use_wandb:
                wandb.init(id=wandb_id,
                        name=opts.wandb_name,
                        project='MotionMetaFormer',
                        settings=wandb.Settings(start_method='fork'))
                wandb.config.update({"run_id": wandb_id})
                wandb.config.update(args)
                installed_packages = {d.project_name: d.version for d in pkg_resources.working_set}
                wandb.config.update({'installed_packages': installed_packages})

    for epoch in range(epoch_start, args.epochs):
        if opts.eval_only:
            evaluate(args, model, test_loader, device)
            exit()

        print(f"[INFO] epoch {epoch+1}")
        loss_names = ['3d_pose', '3d_scale', '2d_proj', 'lg', 'lv', '3d_velocity', 'angle', 'angle_velocity', 'total']
        losses = {name: AverageMeter() for name in loss_names}

        train_one_epoch(args, model, train_loader, optimizer, device, losses)
        scheduler.step()
        mpjpe = evaluate(args, model, test_loader, device)

        best_epoch_checkpoint_path = os.path.join(opts.new_checkpoint, f'best_epoch_{epoch}.pth.tr')
        epoch_checkpoint_path = os.path.join(opts.new_checkpoint, f'epoch_{epoch}.pth.tr')

        if mpjpe < min_mpjpe:
            min_mpjpe = mpjpe
            save_checkpoint(best_epoch_checkpoint_path, epoch, lr, optimizer, model, min_mpjpe, wandb_id)
        else:
            save_checkpoint(epoch_checkpoint_path, epoch, lr, optimizer, model, min_mpjpe, wandb_id)

        # 성능 지표 계산 및 저장
        metrics_path = os.path.join(opts.new_checkpoint, 'metrics', f'epoch_{epoch+1}_metrics.json')
        metrics = {'train_loss': losses['total'].avg, 'validation_mpjpe': mpjpe}
        print(f"Epoch {epoch+1} : {metrics['train_loss']}")
        save_metrics(epoch, metrics, metrics_path)

        if opts.use_wandb:
            wandb.log({
                'lr': lr,
                'train/loss_3d_pose': losses['3d_pose'].avg,
                'train/loss_3d_scale': losses['3d_scale'].avg,
                'train/loss_3d_velocity': losses['3d_velocity'].avg,
                'train/loss_2d_proj': losses['2d_proj'].avg,
                'train/loss_lg': losses['lg'].avg,
                'train/loss_lv': losses['lv'].avg,
                'train/loss_angle': losses['angle'].avg,
                'train/angle_velocity': losses['angle_velocity'].avg,
                'train/total': losses['total'].avg,
                'eval/mpjpe': mpjpe,
                'eval/min_mpjpe': min_mpjpe
            }, step=epoch + 1)

        # lr = decay_lr_exponentially(lr, lr_decay, optimizer)
        
    if opts.use_wandb:
        # 모든 체크포인트 파일을 wandb에 로깅
        for epoch in range(epoch_start, args.epochs):
            best_epoch_checkpoint_path = os.path.join(opts.new_checkpoint, f'best_epoch_{epoch}.pth.tr')
            epoch_checkpoint_path = os.path.join(opts.new_checkpoint, f'epoch_{epoch}.pth.tr')
            
            artifact = wandb.Artifact(f'model_epoch_{epoch}', type='model')
            # 성능이 개선된 경우와 그렇지 않은 경우에 따라 다른 파일을 로깅
            if os.path.exists(best_epoch_checkpoint_path):
                artifact.add_file(best_epoch_checkpoint_path)
            elif os.path.exists(epoch_checkpoint_path):
                artifact.add_file(epoch_checkpoint_path)
            wandb.log_artifact(artifact)

def main():
    opts = parse_args()
    set_random_seed(opts.seed)
    torch.backends.cudnn.benchmark = False
    args = get_config(opts.config)
    
    train(args, opts)

if __name__ == '__main__':
    main()