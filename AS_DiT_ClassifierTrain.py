import argparse

from torch.utils.data import DataLoader
import torch
import os
# print(os.getcwd())
from diffusion import create_diffusion
from model.AS_DiT import DiT_models, DiT_Classifier
from dataset.ECGDataset import ECGDatasetEMD2DNPY
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import logging
import math

import warnings

warnings.filterwarnings("ignore")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def Calculate_acc_ppv_sen_spec_f1(matrix, class_num):
    results_matrix = np.zeros([class_num, 5])
    # diagonal负责统计对角线元素的和
    diagonal = 0
    for i in range(class_num):
        tp = matrix[i][i]
        diagonal += tp
        fn = np.sum(matrix, axis=1)[i] - tp
        fp = np.sum(matrix, axis=0)[i] - tp
        tn = np.sum(matrix) - tp - fp - fn
        acc = (tp + tn) / (tp + tn + fp + fn)  # 准确率
        ppv = tp / (tp + fp)  # 精确率
        sen = tp / (tp + fn)  # 召回率
        spec = tn / (tn + fp)
        results_matrix[i][0] = acc * 100
        results_matrix[i][1] = ppv * 100
        results_matrix[i][2] = sen * 100
        results_matrix[i][3] = spec * 100
        results_matrix[i][4] = 2 * ppv * sen / (ppv + sen)
    return results_matrix


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


def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    device = torch.device('cuda')
    experiment_dir = f'{args.result_path}/{args.experiment_name}'
    checkpoint_dir = f'{experiment_dir}/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # writer = SummaryWriter(os.path.join(experiment_dir, 'logs'))
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")

    data_label = np.load(f'{args.data_path}/data_label_5class.npy')
    data = data_label[:, 0].reshape(-1)
    label = data_label[:, 1].reshape(-1)

    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=args.global_seed,
                                                        shuffle=True,
                                                        stratify=label)

    trainSet = ECGDatasetEMD2DNPY(datas_idx=x_train, labels=y_train, emd_num=4)
    testSet = ECGDatasetEMD2DNPY(datas_idx=x_test, labels=y_test, emd_num=4)
    trainLoader = DataLoader(
        trainSet,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    testLoader = DataLoader(
        testSet,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    logger.info(f"Train dataset contains {len(trainSet)}")
    logger.info(f"Test dataset contains {len(testSet)}")

    best_zhunquelv = 0.
    best_f1 = 0.

    dit = DiT_models["DiT-S/8"](
        input_size=args.input_size,
        in_channels=args.in_channels
    ).to(device)
    ck = torch.load(
        '/data/WangJilin/result/MCDSSL4ECG/AS-DiT-Train-加入对比学习loss-dim256/checkpoints/dit_300.pt')
    dit.load_state_dict((ck['dit']))

    diffusion = create_diffusion(timestep_respacing=str(args.time_steps))  # default: 1000 steps, linear noise schedule

    # print(ldae_model.time_steps.device)
    dit_opt = torch.optim.AdamW(dit.parameters(), lr=args.dit_lr)
    dit_opt.load_state_dict(ck['dit_opt'])

    classifier = DiT_Classifier(dit, num_classes=args.num_classes, encoder_layer=args.encoder_layer, flag=args.flag).to(
        device)
    classifier_opt = torch.optim.AdamW(classifier.parameters(), lr=args.classifier_lr)
    criteon = torch.nn.CrossEntropyLoss().to(device)

    dit.eval()
    classifier.set_encoder(dit, flag=args.flag)
    logger.info(f"Train for {args.epochs} epochs")
    for epoch in range(args.epochs):
        # 首先自监督训练
        logger.info(f"Beginning epoch {epoch}...")

        classifier.train()

        linear_train_bar = tqdm(enumerate(trainLoader), total=len(trainLoader),ncols=100,position=0, leave=True)

        # 训练线性分类层
        for idx, (index, ecg, x, y) in linear_train_bar:
            y = y.type(torch.int64)
            x, y, ecg = x.to(device), y.to(device), ecg.to(device)
            t = torch.randint(0, args.time_steps, (x.shape[0],), device=device)
            x_t = diffusion.q_sample(x, t)
            logits = classifier(x_t, t, ecg)
            loss = criteon(logits, y)
            classifier_opt.zero_grad()
            loss.backward()
            classifier_opt.step()
            linear_train_bar.set_description('Linear Probe Train')

        linear_test_bar = tqdm(enumerate(testLoader), total=len(testLoader), leave=True,ncols=100,position=0)
        linear_train_bar.set_description('Linear Probe Test')
        classifier.eval()
        with torch.no_grad():
            confusion = np.zeros((args.num_classes, args.num_classes))
            for idx, (index, ecg, x, y) in linear_test_bar:
                y = y.type(torch.int64)
                x, y, ecg = x.to(device), y.to(device), ecg.to(device)
                t = torch.randint(0, args.time_steps, (x.shape[0],), device=device)

                x_t = diffusion.q_sample(x, t)
                logits = classifier(x_t, t, ecg)
                pred = logits.argmax(dim=1)


                confusion += confusion_matrix(y.cpu(), pred.cpu(),
                                              labels=[label for label in range(args.num_classes)])
            indicator = Calculate_acc_ppv_sen_spec_f1(confusion, args.num_classes)
            indicator_mean = indicator.mean(axis=0)
            acc, ppv, sen, spec, f1 = indicator_mean[0], indicator_mean[1], indicator_mean[2], indicator_mean[
                3], indicator_mean[4]
            corrects = confusion.diagonal(offset=0)
            per_kinds = confusion.sum(axis=1)
            zhunquelv = corrects.sum() / per_kinds.sum()
            logger.info(
                f"Epoch {epoch}: 准确率: {zhunquelv: .6f}, Average acc: {acc: .6f}, ppv: {ppv: .6f}, sen:{sen: .6f}, spec: {spec: .6f}, F1: {f1: .6f}")
            if zhunquelv > best_zhunquelv:
                best_zhunquelv = zhunquelv
                checkpoint = {
                    "dit": dit.state_dict(),
                    "dit_opt": dit_opt.state_dict(),
                    "classifier": classifier.state_dict(),
                    "classifier_opt": classifier_opt.state_dict(),
                    "confusion": confusion,
                    "indicator": indicator,
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/best_zhunquelv.pt"
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved best zhunquelv checkpoint to {checkpoint_path} in epoch {epoch}")

            if f1 > best_f1:
                best_f1 = f1
                checkpoint = {
                    "dit": dit.state_dict(),
                    "dit_opt": dit_opt.state_dict(),
                    "classifier": classifier.state_dict(),
                    "classifier_opt": classifier_opt.state_dict(),
                    "confusion": confusion,
                    "indicator": indicator,
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/best_f1.pt"
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved best f1 checkpoint to {checkpoint_path} in epoch {epoch}")
    print(f'best zhunquelv is: {best_zhunquelv}')
    print(f'best f1 is: {best_f1}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MCDSSL')
    parser.add_argument('--num_classes', default=5, type=int, help='数据类别数')
    parser.add_argument('--epochs', default=300, type=int, help='迭代轮数')
    parser.add_argument('--dit_lr', default=1e-4, type=float, help='dit学习率')
    parser.add_argument('--classifier_lr', default=5e-4, type=float, help='分类头学习率')
    parser.add_argument('--experiment_name', default='AS-DiT-加入对比学习loss-dim256-Linear-Probe', type=str, help='本次实验名称')
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
    main(args)
