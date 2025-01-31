import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import shutil
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, WeightedRandomSampler
import timm
from torchvision import models
from timm.models.layers import trunc_normal_
# Use Later
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim.optim_factory import param_groups_layer_decay
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from pathlib import Path
from utils import get_img_ano_paths,extract_and_save_tests_gnns
from models.models_esr import models_esrc
import models.lr_decay as lrd
import models.misc as misc
from models.misc import NativeScalerWithGradNormCount as NativeScaler
from engine_train import train_one_epoch, evaluate
from wsiDataset import build_dataset
def get_args_parser():
    parser = argparse.ArgumentParser('Graph-GNN fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--model',default='graph-hnet-pseudo',type=str)
    parser.add_argument('--patch_encoder',default='resnet50',type=str)
    # parser.add_argument('--test_batch_number',default=41,type = int )
    # Model parameters

    parser.add_argument('--patch_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--layer', default=2, type=int,
                        help='images input size')
    parser.add_argument('--graph_size',default=8,type=int)
    parser.add_argument('--finetune',default='/home/z.sun/graph-wsi/pretrained_encoder/resnet-50-best-7.pth',
                        help="fine-tuned model on 100k or 7k dataset")
    parser.add_argument('--linear_probe',action="store_true")

    parser.add_argument('--resnet50',action='store_true')
    parser.add_argument('--join_encoder',action='store_true')
    
    parser.add_argument('--densenet121',action='store_true')
    parser.add_argument('--efficientnet_v2_s',action='store_true')
    parser.add_argument('--mobile_netv2',action='store_true')
    parser.add_argument('--drop_out',default=0,type=float)

    # Optimizer parameters
    
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-5, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=3, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--balance_coeff',type=float,default=0)
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.0,
                        help='Label smoothing (default: 0)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')


    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')


    # Dataset parameters
    parser.add_argument('--train_steps',default=48,type=int)
    parser.add_argument('--train_data_path', default="/home/data_repository/PATH-DT-MSU_dev/WSS2_v2_psi", type=str,
                        help='dataset path')
    parser.add_argument('--val_data_path', default="/home/z.sun/graph-wsi/Test_data/NEW-test_split_v1_64_layer2_coeff1.0_intersection0.8", type=str,
                        help='dataset path')
    parser.add_argument('--test_data_path', default="/home/data_repository/PATH-DT-MSU_dev/WSS2_v2_psi", type=str,
                        help='dataset path')
    parser.add_argument('--test_output_path', default="./patches_test", type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=5, type=int,
                        help='number of the classification types')
    parser.add_argument('--batches_per_worker',type=int,default=4)
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default=0,type=int,
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    
    parser.add_argument('--runs_name',default='debug',type=str)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed',action="store_true")
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

def main(args):
    # Currently no need Distributed
    # misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    
    if args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        save_dir = Path(f'{args.output_dir}/imgs')
        save_dir.mkdir(exist_ok=True, parents=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
        with open(os.path.join(args.log_dir, 'args.txt'), 'w') as f:
            f.write(str(args))
    else:
        log_writer = None

    img_anno_paths_test = get_img_ano_paths(
        ds_folder=Path(args.test_data_path), sample="test",version='v2.5'
    )
    
    out_dir = Path(args.test_output_path)

    if not out_dir.exists():
        extract_and_save_tests_gnns(
            img_anno_paths=img_anno_paths_test,
            out_folder=out_dir,
            patch_size=args.patch_size,
            layer=args.layer,
            n=100,
            graph_size=args.graph_size
        )
    train_val_dataset,data_augmentations = build_dataset(True,args)
    
    test_dataset = build_dataset(False,args)
    data_loader_test = DataLoader(test_dataset, batch_size=4, shuffle=True)
    


    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)


    if args.resnet50:
        patch_encoder  = models.resnet50(pretrained=True)
        num_ftrs = patch_encoder.fc.in_features
        patch_encoder.fc = torch.nn.Sequential(
            torch.nn.Dropout(args.drop_out),
            torch.nn.Linear(num_ftrs, args.nb_classes)
        )
        # model.fc = torch.nn.Sequential(
        #     torch.nn.Dropout(args.drop_out),
        #     torch.nn.Linear(num_ftrs, args.nb_classes)
        # )
        # patch_encoder.fc = torch.nn.Identity()
    elif args.densenet121:
        patch_encoder = models.densenet121(pretrained=True)
        num_ftrs = 1024
        patch_encoder.classifier = torch.nn.Identity()
    elif args.efficientnet_v2_s:
        patch_encoder = models.efficientnet_v2_s( pretrained=True)
        num_ftrs = 1280
        patch_encoder.classifier = torch.nn.Identity()
    elif args.join_encoder:
        patch_encoder=models_esrc(num_in_ch=3,num_feat=64, num_block=15, num_grow_ch=32, scale=2,drop_rate=args.drop_out,
                        num_classes=args.nb_classes)
        num_ftrs = 1024
    else:
        patch_encoder  = models.resnet50(pretrained=True)
        num_ftrs = patch_encoder.fc.in_features
        patch_encoder.fc = torch.nn.Sequential(
            torch.nn.Dropout(args.drop_out),
            torch.nn.Linear(num_ftrs, args.nb_classes)
        )


    if args.finetune != "":
        loadnet = torch.load(args.finetune, map_location=torch.device('cpu'))
        keyname = 'model'

        print(patch_encoder.load_state_dict(loadnet[keyname], strict=False))
        # patch_encoder.eval()
    if args.model == 'graph-vnet':
        from models.graph_vnet import Graph_VNet
        model = Graph_VNet(num_ftrs,args.nb_classes)
    elif args.model == 'graph-hnet':
        from models.graph_hnet import Graph_HNet
        model = Graph_HNet(num_ftrs,args.nb_classes)
    elif args.model == 'graph-hnet-pseudo':
        from models.graph_hnet_pseudo import Graph_HNet
        model = Graph_HNet(num_ftrs,args.nb_classes)
    elif args.model == 'simple_gcn': 
        from models.simple_gcn import GCN_model
        model = GCN_model(num_ftrs,args.nb_classes)
    else:
        model = torch.nn.Linear(1,1)



    patch_encoder.to(device)

    model.to(device)

    model_without_ddp = model

    if args.linear_probe:
        for param in patch_encoder.parameters():
            param.requires_grad = False
        for param in patch_encoder.head.parameters():
            param.requires_grad = True
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    # if args.lr is None:  # only base_lr is specified
    #     args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    param_groups = param_groups_layer_decay(patch_encoder,args.weight_decay,layer_decay=args.layer_decay)
    optimizer_patch_encoder = torch.optim.AdamW(param_groups, lr=args.blr)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    #
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)


    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    
    # test_stats,class_accuracy = evaluate(data_loader_val, patch_encoder,model, device,args)
    for epoch in range(args.start_epoch, args.epochs):
        args.epoch = epoch 
        data_loader_train = train_val_dataset.torch_generator(
            batch_size=args.batch_size,
            n_batches=args.train_steps,
            batches_per_worker=args.batches_per_worker,
            transforms=data_augmentations,
            max_workers=args.num_workers,
        )
        train_stats = train_one_epoch(
            model,patch_encoder, criterion, data_loader_train,
            optimizer,optimizer_patch_encoder, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )
        # print('------------------------test on train----------------------------\n')
        
        data_loader_val = train_val_dataset.torch_generator(
            batch_size=args.batch_size,
            n_batches=16,
            batches_per_worker=args.batches_per_worker,
            transforms=data_augmentations,
            max_workers=args.num_workers,
        )
        val_stats,class_accuracy = evaluate(data_loader_val, patch_encoder,model, device,args,test=False)
        if log_writer is not None:
            log_writer.add_scalar('perf/val_acc1', val_stats['acc1'], epoch)
            log_writer.add_scalar('perf/val_acc3', val_stats['acc3'], epoch)
            log_writer.add_scalar('perf/val_loss', val_stats['loss'], epoch)
            for class_label, acc in class_accuracy.items():
                log_writer.add_scalar(f'val_class_accuracy/{class_label}', acc['acc'], epoch)

        # print('------------------------test on train finish----------------------------\n')
        test_stats,class_accuracy = evaluate(data_loader_test, patch_encoder,model, device,args)

        print(f"Accuracy of the network on the {len(test_dataset)} test images: {test_stats['acc1']:.1f}%")
        if max(max_accuracy, test_stats["acc1"]) > max_accuracy :
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            if args.output_dir:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch,model_name = 'best')
                misc.save_model(
                    args=args, model=patch_encoder, model_without_ddp=patch_encoder, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch,model_name = 'patch_encoder_best')
        # else:
        #     if args.output_dir:
        #         misc.save_model(
        #             args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
        #             loss_scaler=loss_scaler, epoch=epoch
        #             )
        #         misc.save_model(
        #             args=args, model=patch_encoder, model_without_ddp=patch_encoder, optimizer=optimizer,
        #             loss_scaler=loss_scaler, epoch=epoch)
        print(f'Max accuracy: {max_accuracy:.2f}%')
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc3', test_stats['acc3'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)
            for class_label, acc in class_accuracy.items():
                log_writer.add_scalar(f'class_accuracy/{class_label}', acc['acc'], epoch)

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.distributed:
        # pass 
        exit(1)
    else :
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        args.device = 0
    if args.output_dir:
        file_path = os.path.abspath(__file__)
        
        args.output_dir = str(Path.joinpath(Path(__file__).parent,args.output_dir,args.runs_name))
        args.log_dir = args.output_dir
        # print(f"Current time: {current_time}")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        source_directory = Path(__file__).parent
        for root, dirs, files in os.walk(source_directory):
            if 'output_dir'  in root:
                continue
            for file in files:
                if file.endswith('.py'):
                    # 构建源文件的完整路径
                    source_file = os.path.join(root, file)
                    
                    # 计算相对路径
                    rel_path = os.path.relpath(root, source_directory)
                    
                    # 构建目标文件的完整路径，保持原有的目录结构
                    dest_file = os.path.join(args.output_dir, rel_path, file)
                    
                    # 确保目标文件的父目录存在
                    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                    
                    # 复制文件
                    shutil.copy2(source_file, dest_file)
                    # print(f"Copied: {source_file} -> {dest_file}")

    main(args)