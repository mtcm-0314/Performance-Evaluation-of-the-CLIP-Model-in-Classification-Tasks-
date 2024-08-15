import argparse
import numpy as np
import os
# import pretty_errors
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from clip_model import ClipModel
from data.data_loader import ImageTextData
from utils import gather_res, get_logger, set_gpu, set_seed
from torch.utils.data import ConcatDataset
import DCA
import Confusion_matrices


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mode', type=str, choices=['zs', 'fe', 'ft'],
                        default='ft')  # zeroshot, feature extraction, fine-tuning
    parser.add_argument('--dataset', type=int, default=19)
    parser.add_argument('--model', type=int, default=5)  # -1 for sweep
    parser.add_argument('--root', type=str, default='./')  # root path of dataset
    parser.add_argument('--log_file', type=str, default='log.txt')
    parser.add_argument('--seed', type=int, default=42)  # random seed
    parser.add_argument('--result', action='store_true')  # if you want to sweep results statistics
    parser.add_argument('--batchsize', type=int, default=32)
    # 设置momentum
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nepoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)  # 初始值为5e-5
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.98)
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.10)  # 初始值为0.2,改为了0.02
    # the following test data and test batchsize are only used for fine-tuning mode
    parser.add_argument('--test_batchsize', type=int, default=32)
    parser.add_argument('--test_data', type=int, default=24)
    # parser.add_argument('--val_data', type=int, default=14)
    args = parser.parse_args()
    return args


def main(args):
    model, dataset = args.model, args.dataset
    model = args.model
    model_name = ClipModel.get_model_name_by_index(model)
    dataset_name = ImageTextData.get_data_name_by_index(dataset)
    args.log_file = os.getcwd() + '/log/{}_{}_{}.txt'.format(args.mode, model_name, dataset_name)
    logger = get_logger(args.log_file, args.log_file)
    logger.info(args)

    clip = ClipModel(model, logger=logger)

    logger.info(f'Clip model {model_name} loaded')
    # print(clip.model)

    itdata = ImageTextData(dataset, root=args.root, preprocess=clip.preprocess)
    labels = itdata.labels
    ## print("labels:",labels)
    file_to_val = int(0.2 * len(itdata))  # 从训练集中划分了百分之二十作为验证集
    # file_to_test = int(0.3 * len(itdata))
    file_to_train = len(itdata) - file_to_val
    trainset, valset = random_split(itdata, [file_to_train, file_to_val],
                                    generator=torch.Generator().manual_seed(42))

    # trainset = ImageTextData(dataset, root=args.root, preprocess=clip.preprocess)
    # valset = ImageTextData(args.val_data, root=args.root, preprocess=clip.preprocess)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batchsize, shuffle=False, num_workers=4)
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=False)
    logger.info(f'Dataset {dataset_name} loaded')

    if args.mode == 'zs':  # zeroshot
        eval_acc, pre, rec, f1, _, _, _ = clip.evaluate(val_loader, labels)
        logger.info('Results: {}'.format(eval_acc))
        logger.info('Accuracy: {:.2f}%'.format(100. * eval_acc))
    elif args.mode == 'fe':  # feature extraction
        res = clip.feature_extraction(train_loader)
        logger.info('Feature extracted!')
        if not os.path.exists('feat'):
            os.makedirs('feat')
        feat_file = 'feat/{}_{}_{}.csv'.format(args.mode, model_name, dataset_name)
        np.savetxt(feat_file, res, fmt='%.4f')
    elif args.mode == 'ft':  # fine-tuning
        test_data = ImageTextData(args.test_data, root=args.root, preprocess=clip.preprocess)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batchsize, shuffle=False,
                                                  drop_last=True, num_workers=4)
        # 使用SGD作为优化器
        optimizer = optim.SGD(clip.model.parameters(), lr=args.lr, momentum=args.momentum,
                             weight_decay=args.weight_decay)
        # optimizer = optim.AdamW(clip.model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps,
        #                        weight_decay=args.weight_decay)

        best_acc = clip.finetune(train_loader, labels, val_loader, test_loader, optimizer, args.nepoch,
                                 lr=args.lr)
        logger.info('Accuracy: {:.2f}%'.format(best_acc * 100))
    else:
        raise NotImplementedError


def sweep_index(model=-1, data=-1):
    if model == -1 and data == -1:
        m_sweep_index = range(len(ClipModel.CLIP_MODELS))
        d_sweep_index = range(len(ImageTextData._DATA_FOLDER))
    elif model == -1 and data != -1:
        m_sweep_index = range(len(ClipModel.CLIP_MODELS))
        d_sweep_index = range(data, data + 1)
    elif data == -1 and model != -1:
        m_sweep_index = range(model, model + 1)
        d_sweep_index = range(len(ImageTextData._DATA_FOLDER))
    else:
        m_sweep_index = range(model, model + 1)
        d_sweep_index = range(data, data + 1)
    return m_sweep_index, d_sweep_index


def sweep(model=-1, data=-1):
    m_sweep_index, d_sweep_index = sweep_index(model, data)
    if args.result:
        model_name_lst = [ClipModel.get_model_name_by_index(i) for i in m_sweep_index]
        data_name_lst = [ImageTextData.get_data_name_by_index(i) for i in d_sweep_index]
        res = gather_res(model_name_lst, data_name_lst)
        for line in res:
            print(line)
    else:
        for model in m_sweep_index:
            for data in d_sweep_index:
                args.model = model
                args.dataset = data
                main(args)


if __name__ == '__main__':
    args = get_args()
    set_gpu(args.gpu)
    set_seed(args.seed)
    # sweep(args.model, args.dataset)

    # weight_decay = [0.05 * idx for idx in range(1, 21)]
    # print(weight_decay)
    # for w_d in weight_decay:
    #     print('This weight_decay is:', w_d)
    #     args.weight_decay = w_d
    #     main(args)

    lr = [10 ** (-i) for i in range(2, 9)]
    print(lr)
    for lr_ in lr:
        print('This lr is:', lr_)
        args.lr = lr_
        main(args)
