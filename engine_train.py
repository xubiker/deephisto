import math
import sys
from typing import Iterable, Optional

import torch
import numpy as np 
import pandas as pd 
import dataframe_image as dfi 
from timm.data import Mixup
from timm.utils import accuracy
from tqdm import tqdm 
import models.misc as misc
import models.lr_sched as lr_sched
from einops import rearrange
                
def _forward_model_patch_encoder(model,patch_encoder,args,samples,pos,test = False ):
    if test == True:
        batch_size = pos.shape[0]
        samples = rearrange(samples,'b n c h w -> (b n) c h w', b=batch_size, c=3,h=args.patch_size, w=args.patch_size)
        pos = rearrange(pos,'b n d -> (b n) d', b=batch_size, n =args.graph_size*args.graph_size , d=2 )
    else:
        batch_size = args.batch_size // args.graph_size // args.graph_size
    
    if args.join_encoder:
        latents = patch_encoder.forward_linear_features(samples)
    elif args.resnet50:
        logits,latents = patch_encoder(samples)
    else:
        latents = patch_encoder(samples)
    
    latents = rearrange(latents,'(b h w) d -> b d h w', b=batch_size, h=args.graph_size, w=args.graph_size)
    # pos = rearrange(pos,'(b h w) d -> b d h w', b=batch_size, h=args.graph_size, w=args.graph_size)
    if args.model == 'graph-hnet-pseudo':
        _, predicted = torch.max(logits, 1)
        outputs = model(latents,pos,predicted)
    else:
        return logits
    return outputs



def train_one_epoch(model: torch.nn.Module, patch_encoder: torch.nn.Module ,criterion: torch.nn.Module,
                    data_loader_train:Iterable, optimizer: torch.optim.Optimizer,optimizer_patch_encoder: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    patch_encoder.train(True)
    torch.cuda.empty_cache()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'TRAIN : Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
            print('log_dir: {}'.format(log_writer.log_dir))
    data_iter_step = 0 
    
    for samples, targets,pos,_ in tqdm(
            # for inputs, labels in tqdm(
            data_loader_train,
            total=args.train_steps,
            desc=f"Epoch {epoch + 1}",
        ):
        
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / args.train_steps + epoch, args)
        
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        pos = pos.to(device,non_blocking = True)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = _forward_model_patch_encoder(model,patch_encoder,args,samples,pos)
            loss = criterion(outputs,targets)
            loss.backward()
            optimizer.step()
            optimizer_patch_encoder.step()

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)


        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
            optimizer_patch_encoder.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / args.train_steps + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)
        data_iter_step+=1 
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(data_loader, patch_encoder,model, device,args,test=True):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    patch_encoder.eval()
    class_accuracy = {}
    labels = ['AT', 'BG','LP', 'MM', 'TUM']
    # 初始化行和列
    # 创建 DataFrame
    df = pd.DataFrame(np.zeros((labels.__len__(),labels.__len__()),int),columns=labels, index=labels)

    # 设置标题
    df.columns.name = 'gt'
    df.index.name = 'pred'
    for data_iter_step, batch in enumerate(data_loader):
        if len(batch) == 3:
            samples, targets,pos = batch 
        else:
            samples, targets,pos,_ = batch 
        # if data_iter_step == args.test_batch_number:
        #     print(f'Test end after {args.test_batch_number} batch \n')
        #     break
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        pos = pos.to(device,non_blocking = True)
        with torch.no_grad():
            outputs = _forward_model_patch_encoder(model,patch_encoder,args,samples,pos,test = test)
            batch_size = pos.shape[0]
            if test == True :
                targets = rearrange(targets,'b n -> (b n)',b=batch_size)
            loss = criterion(outputs,targets)
            

        acc1, acc3 = accuracy(outputs, targets, topk=(1, 3))
        _, predicted = torch.max(outputs, 1)

        # 遍历预测结果和真实标签，计算每个类别的准确率
        for pred, tar in zip(predicted, targets):
            pred = pred.item()
            tar = tar.item()
            df.iloc[pred,tar] += 1
            if tar not in class_accuracy:
                class_accuracy[tar] = {'total': 0, 'correct': 0}

            class_accuracy[tar]['total'] += 1
            if pred == tar:
                class_accuracy[tar]['correct'] += 1

    # 计算每个类别的准确率


        batch_size = samples.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc3'].update(acc3.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    dfi.export(df.style.background_gradient(), f"{args.output_dir}/imgs/matrix_conf_{args.epoch}.png", table_conversion="matplotlib")
    print('* Acc@1 {top1.global_avg:.3f} Acc@3 {top3.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top3=metric_logger.acc3, losses=metric_logger.loss))
    for class_label, acc in class_accuracy.items():
            acc['acc'] = acc['correct'] / acc['total'] * 100
    print(class_accuracy)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()},class_accuracy