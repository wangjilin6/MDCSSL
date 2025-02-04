import argparse

import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch import nn

from split_data import get_train_test
from torch.utils.data import DataLoader
import torch
import os
from diffusion import create_diffusion
from model.AS_DiT import DiT_models, DiT_Classifier
from dataset.ECGDataset import ECGDatasetEMD2DNPY
from tqdm import tqdm
import logging
import math
import warnings

warnings.filterwarnings("ignore")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Dim_reduction_net(torch.nn.Module):
    def __init__(self, input_size=24576, vectordim=128) -> None:
        super().__init__()

        self.downLinear = torch.nn.Linear(input_size, vectordim)
        nn.init.constant_(self.downLinear.weight, 0)
        nn.init.constant_(self.downLinear.bias, 0)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        out = self.downLinear(x)
        return F.normalize(out, p=2, dim=1)


def info_nce_loss(out_1, out_2, batch_size, temperature=0.5):
    out = torch.cat([out_1, out_2], dim=0)
    # print(f'out: {out.shape}')
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    # print(f'out: {torch.mm(out, out.t().contiguous())}')
    # print(f'sim_matrix: {sim_matrix.mean()}')
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    # print(f'info_nce_loss: {loss}')
    return loss


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


def AS_DiT_train(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    device = torch.device('cuda')
    experiment_dir = f'{args.result_path}/{args.experiment_name}'
    checkpoint_dir = f'{experiment_dir}/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # writer = SummaryWriter(os.path.join(experiment_dir, 'logs'))
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")

    # x_train, x_test, y_train, y_test = get_train_test()
    data_label = np.load(f'{args.data_path}/data_label_5class.npy')
    data = data_label[:, 0].reshape(-1)
    label = data_label[:, 1].reshape(-1)

    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=args.global_seed,
                                                        shuffle=True,
                                                        stratify=label)
    trainSet = ECGDatasetEMD2DNPY(datas_idx=x_train, labels=y_train, emd_num=4)
    trainLoader = DataLoader(
        trainSet,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=False
    )

    logger.info(f"Train dataset contains {len(trainSet)}")

    dit = DiT_models["DiT-S/8"](
        input_size=args.input_size,
        in_channels=args.in_channels
    ).to(device)

    diffusion = create_diffusion(timestep_respacing=str(args.time_steps))  # default: 1000 steps, linear noise schedule

    # print(ldae_model.time_steps.device)
    dit_opt = torch.optim.AdamW(dit.parameters(), lr=args.dit_lr)

    # ck = torch.load(
    #     '/data/WangJilin/result/MIT-ECG-Diffusion-Self-Supervision-Classifier/DiT-ASSA-ASCA-EMD4-PreTrain-ECG作条件-两种交叉注意力/checkpoints/dit_6000.pt')
    # dit.load_state_dict((ck['dit']))
    # dit_opt.load_state_dict(ck['dit_opt'])
    #
    # checkpoint = {
    #     "dit": dit.state_dict(),
    #     "dit_opt": dit_opt.state_dict(),
    #     "args": args
    # }
    # checkpoint_path = f"{checkpoint_dir}/dit_6000.pt"
    # torch.save(checkpoint, checkpoint_path)

    start_epoch = 0
    ck_list = os.listdir(checkpoint_dir)
    if len(ck_list) == 0:
        ck = torch.load(
            '/data/WangJilin/result/MCDSSL4ECG/AS-DiT-Pretrain/checkpoints/dit_6000.pt')
        dit.load_state_dict((ck['dit']))
    else:
        for ck_name in ck_list:
            number_str = ck_name[4:-3]
            if int(number_str) > start_epoch:
                start_epoch = int(number_str)

        ck = torch.load(
            f'{checkpoint_dir}/dit_{start_epoch}.pt')
        dit.load_state_dict((ck['dit']))
        dit_opt.load_state_dict(ck['dit_opt'])

    reduction_net = Dim_reduction_net(vectordim=256).to(device)
    for param_q in reduction_net.parameters():  #该网络仅做降维用，因此不需要更新参数
        param_q.requires_grad = False

    logger.info(f"Train for {args.epochs} epochs")
    for epoch in range(start_epoch, args.epochs):
        # 首先自监督训练
        logger.info(f"Beginning epoch {epoch}...")
        # adjust_learning_rate(dit_opt, args.dit_lr, epoch, args)
        dit.train()
        dit_losses = []
        train_bar = tqdm(enumerate(trainLoader), total=len(trainLoader), leave=True, ncols=100, position=0)
        for idx, (index, ecg, x, y) in train_bar:
            # y = y.type(torch.int64)
            x = x.to(device)
            ecg = ecg.to(device)
            batch_size = x.shape[0]

            t = torch.randint(0, args.time_steps, (x.shape[0],), device=device)
            t_1 = (t + args.time_steps//2) % 1000

            t = torch.cat([t, t_1], dim=0)

            x = torch.cat([x, x], dim=0)
            ecg = torch.cat([ecg, ecg], dim=0)

            # t = torch.randint(0, args.time_steps, (x.shape[0],), device=device)
            # t = torch.cat([t, (t + 500) % 1000], dim=0)
            model_kwargs = dict(condition=ecg)

            features = []

            def hook_fn(module, input, output):
                features.append(output)

            hook = dit.blocks[args.encoder_layer - 1].register_forward_hook(hook_fn)

            loss_dict = diffusion.training_losses(dit, x, t, model_kwargs)
            # print(len(loss_dict["loss"]))

            features = features[0]
            hook.remove()

            loss = loss_dict["loss"][0:batch_size].mean()
            # 增强分支-实验证明没有带来性能提升，反而下降了
            # loss += ((features[:batch_size] - features[batch_size:]) ** 2).mean()
            features = reduction_net(features)
            loss += info_nce_loss(features[:batch_size].reshape(batch_size, -1),
                                  features[batch_size:].reshape(batch_size, -1), batch_size, temperature=0.5)

            dit_opt.zero_grad()
            loss.backward()
            dit_opt.step()

            dit_losses.append(loss.item())
            avg_loss = sum(dit_losses) / len(dit_losses)
            train_bar.set_description(
                f"当前训练轮数为{epoch + 1},encoder:{args.encoder_layer},loss:{avg_loss:.6f}")

            del features
        # writer.add_scalar('lDAE_loss', sum(ldae_losses)/len(ldae_losses), global_step=epoch+1)
        logger.info(f"DiT loss in epoch {epoch + 1}: {sum(dit_losses) / len(dit_losses) : .6f}")

        if (epoch + 1) % 50 == 0:
            checkpoint = {
                "dit": dit.state_dict(),
                "dit_opt": dit_opt.state_dict(),
                "args": args
            }
            checkpoint_path = f"{checkpoint_dir}/dit_{epoch + 1}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved dit checkpoint to {checkpoint_path} in epoch {epoch}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MCDSSL')
    parser.add_argument('--num_classes', default=5, type=int, help='数据类别数')
    parser.add_argument('--epochs', default=300, type=int, help='迭代轮数')
    parser.add_argument('--dit_lr', default=1e-4, type=float, help='学习率')
    parser.add_argument('--experiment_name', default='AS-DiT-Train-加入对比学习loss-dim256', type=str,
                        help='本次实验名称')
    parser.add_argument('--batch_size', default=1024, type=int, help='一个批次的数据量')
    parser.add_argument('--global_seed', default=3407, type=int, help='随机种子')
    parser.add_argument('--data_path', default='/data/WangJilin/data/MIT-BIH/data_used', type=str, help='数据存放位置')
    parser.add_argument('--result_path', default='/data/WangJilin/result/MCDSSL4ECG', type=str, help='结果存放位置')
    parser.add_argument('--method', default='gram-npy', type=str, help='ECG数据变换方法')
    parser.add_argument('--input_size', default=64, type=int, help='输入数据大小')
    parser.add_argument('--in_channels', default=1, type=int, help='通道数')
    parser.add_argument('--time_steps', default=1000, type=int, help='扩散模型最大加噪步数')
    parser.add_argument('--encoder_layer', default=6, type=int, help='编码器的AS-DiT block层数')
    parser.add_argument('--flag', default=False, type=bool, help='是否微调')
    args = parser.parse_args()
    AS_DiT_train(args)
