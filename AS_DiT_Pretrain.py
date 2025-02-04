import argparse
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


class Args:
    def __init__(self):
        pass


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


def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    device = torch.device('cuda')
    experiment_dir = f'{args.result_path}/{args.experiment_name}'
    checkpoint_dir = f'{experiment_dir}/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # writer = SummaryWriter(os.path.join(experiment_dir, 'logs'))
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")

    x_train, x_test, y_train, y_test = get_train_test()

    trainSet = ECGDatasetEMD2DNPY(datas_idx=x_train, labels=y_train, emd_num=4)
    trainLoader = DataLoader(
        trainSet,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
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

    logger.info(f"Train for {args.epochs} epochs")
    for epoch in range(args.epochs):
        # 首先自监督训练
        logger.info(f"Beginning epoch {epoch}...")
        # adjust_learning_rate(dit_opt, args.dit_lr, epoch, args)
        dit.train()
        dit_losses = []
        train_bar = tqdm(enumerate(trainLoader), total=len(trainLoader), leave=True)
        for idx, (index, ecg, x, y) in train_bar:
            # y = y.type(torch.int64)
            x = x.to(device)
            ecg = ecg.to(device)
            t = torch.randint(0, args.time_steps, (x.shape[0],), device=device)
            model_kwargs = dict(condition=ecg)

            loss_dict = diffusion.training_losses(dit, x, t, model_kwargs)
            # loss_dict = diffusion.training_losses(model, x, t)
            loss = loss_dict["loss"].mean()
            dit_opt.zero_grad()
            loss.backward()
            dit_opt.step()

            dit_losses.append(loss.item())
            avg_loss = sum(dit_losses) / len(dit_losses)
            train_bar.set_description('当前训练轮数为{0},当前平均损失为:{1:.6f}'.format(epoch + 1, avg_loss))

        # writer.add_scalar('lDAE_loss', sum(ldae_losses)/len(ldae_losses), global_step=epoch+1)
        logger.info(f"DiT loss in epoch {epoch + 1}: {sum(dit_losses) / len(dit_losses) : .6f}")

        if (epoch + 1) % 300 == 0:
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
    parser.add_argument('--epochs', default=6000, type=int, help='迭代轮数')
    parser.add_argument('--dit_lr', default=1e-4, type=float, help='学习率')
    parser.add_argument('--experiment_name', default='AS-DiT-Pretrain', type=str, help='本次实验名称')
    parser.add_argument('--batch_size', default=2048, type=int, help='一个批次的数据量')
    parser.add_argument('--global_seed', default=3407, type=int, help='随机种子')
    parser.add_argument('--data_path', default='/data/WangJilin/data/MIT-BIH/data_used', type=str, help='数据存放位置')
    parser.add_argument('--result_path', default='/data/WangJilin/result/MCDSSL4ECG', type=str, help='结果存放位置')
    parser.add_argument('--method', default='gram-npy', type=str, help='ECG数据变换方法')
    parser.add_argument('--input_size', default=64, type=int, help='输入数据大小')
    parser.add_argument('--in_channels', default=1, type=int, help='通道数')
    parser.add_argument('--time_steps', default=1000, type=int, help='扩散模型最大加噪步数')
    args = parser.parse_args()
    main(args)
