import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
import loss
import torch.nn.functional as F
from utils import *


def image_train(resize_size=256, crop_size=224):
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )


def image_test(resize_size=256, crop_size=224):
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(
        dsets["target"],
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )

    dsets["test"] = ImageList(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(
        dsets["test"],
        batch_size=train_bs * 3,
        shuffle=False,
        num_workers=args.worker,
        drop_last=False,
    )

    return dset_loaders


def test_target(args, zz=""):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == "res":
        netF = network.ResBase(res_name=args.net).cuda()

    netB = network.feat_bottleneck(
        type=args.classifier,
        feature_dim=netF.in_features,
        bottleneck_dim=args.bottleneck,
    ).cuda()
    netC = network.feat_classifier(
        type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck
    ).cuda()

    args.modelpath = args.output_dir_src + "/source_F_" + str(zz) + ".pt"
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + "/source_B_" + str(zz) + ".pt"
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + "/source_C_" + str(zz) + ".pt"
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()

    acc, acc_list = cal_acc(
        dset_loaders["test"], netF, netB, netC, args.dset == "visda17"
    )
    log_str = (
        "\nZz: {}, Task: {}, Accuracy = {:.2f}%".format(zz, args.name, acc)
        + "\n"
        + str(acc_list)
    )
    args.out_file.write(log_str + "\n")
    args.out_file.flush()
    print(log_str + "\n")


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def train_target_bait(args, zz=""):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == "res":
        netF = network.ResBase(res_name=args.net).cuda()

    netB = network.feat_bottleneck(
        type=args.classifier,
        feature_dim=netF.in_features,
        bottleneck_dim=args.bottleneck,
    ).cuda()
    netC = network.feat_classifier(
        type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck
    ).cuda()
    oldC = network.feat_classifier(
        type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck
    ).cuda()

    args.modelpath = args.output_dir_src + "/source_F_" + str(zz) + ".pt"
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + "/source_B_" + str(zz) + ".pt"
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + "/source_C_" + str(zz) + ".pt"
    netC.load_state_dict(torch.load(args.modelpath))
    oldC.load_state_dict(torch.load(args.modelpath))
    oldC.eval()
    netC.train()
    for k, v in oldC.named_parameters():
        v.requires_grad = False

    param_group = []
    param_group_c = []
    for k, v in netF.named_parameters():
        param_group += [{"params": v, "lr": args.lr * 0.01}]  # 0.1
    for k, v in netB.named_parameters():
        param_group += [{"params": v, "lr": args.lr * 0.1}]  # 1
    for k, v in netC.named_parameters():
        param_group_c += [{"params": v, "lr": args.lr * 0.1}]  # 1
    optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=5e-4, nesterov=True)
    optimizer_c = optim.SGD(
        param_group_c, momentum=0.9, weight_decay=5e-4, nesterov=True
    )

    netF.train()
    netB.train()

    iter_num = 0
    iter_target = iter(dset_loaders["target"])
    while iter_num < (args.max_epoch) * len(dset_loaders["target"]):
        try:
            inputs_test, _, tar_idx = iter_target.next()
        except:
            iter_target = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_target.next()
        if inputs_test.size(0) == 1:
            continue
        iter_num += 1
        inputs_test = inputs_test.cuda()
        batch_size = inputs_test.shape[0]

        if True:
            total_loss = 0
            features_test = netB(netF(inputs_test))
            outputs_test = netC(features_test)
            outputs_test_old = oldC(features_test)

            softmax_out = nn.Softmax(dim=1)(outputs_test)
            softmax_out_old = nn.Softmax(dim=1)(outputs_test_old)

            loss_cast = loss.SKL(softmax_out, softmax_out_old).sum(dim=1)

            entropy_old = Entropy(softmax_out_old)
            indx = entropy_old.topk(int(batch_size * 0.5), largest=True)[-1]
            ones_mask = torch.ones(batch_size).cuda() * -1
            ones_mask[indx] = 1
            loss_cast = loss_cast * ones_mask
            total_loss -= torch.mean(loss_cast) * 10

            optimizer_c.zero_grad()
            total_loss.backward()
            optimizer_c.step()

        for _ in range(1):
            total_loss = 0
            features_test = netB(netF(inputs_test))
            outputs_test = netC(features_test)

            softmax_out = nn.Softmax(dim=1)(outputs_test)

            outputs_test_old = oldC(features_test)
            softmax_out_old = nn.Softmax(dim=1)(outputs_test_old)

            msoftmax = softmax_out_old.mean(dim=0)
            cb_loss = torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
            total_loss += cb_loss

            msoftmax = softmax_out.mean(dim=0)
            cb_loss = torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
            total_loss += cb_loss

            loss_bite = (-softmax_out_old * torch.log(softmax_out + 1e-5)).sum(1) - (
                softmax_out * torch.log(softmax_out_old + 1e-5)
            ).sum(1)
            total_loss += torch.mean(loss_bite)  # *0.8

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if iter_num % int(args.interval * len(dset_loaders["target"])) == 0:
            netF.eval()
            netB.eval()
            netC.eval()
            acc, acc_list = cal_acc(
                dset_loaders["test"], netF, netB, oldC, args.dset == "visda17"
            )
            log_str = (
                "Task: {}, Iter:{}/{}; Accuracy = {:.2f}%".format(
                    args.name,
                    iter_num,
                    args.max_epoch * len(dset_loaders["target"]),
                    acc,
                )
                + "\n"
                + str(acc_list)
            )

            args.out_file.write(log_str + "\n")
            args.out_file.flush()
            print(log_str + "\n")

            netF.train()
            netB.train()
            netC.train()
    return netF, netB, netC


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BAIT on VisDA")
    parser.add_argument(
        "--gpu_id", type=str, nargs="?", default="1", help="device id to run"
    )
    parser.add_argument("--s", type=int, default=0, help="source")
    parser.add_argument("--t", type=int, default=1, help="target")
    parser.add_argument("--batch_size", type=int, default=128, help="batch_size")
    parser.add_argument("--worker", type=int, default=4, help="number of workers")
    parser.add_argument(
        "--dset", type=str, default="visda-2017", choices=["visda-2017"]
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--net", type=str, default="resnet101", help="resnet50, resnet101"
    )
    parser.add_argument("--seed", type=int, default=2020, help="random seed")

    parser.add_argument("--max_epoch", type=int, default=15, help="max iterations")
    parser.add_argument("--interval", type=float, default=0.2, help="max iterations")

    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--layer", type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument("--classifier", type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument("--smooth", type=float, default=0.1)
    parser.add_argument("--output", type=str, default="bait")
    parser.add_argument("--da", type=str, default="uda")
    parser.add_argument("--trte", type=str, default="val", choices=["full", "val"])
    parser.add_argument(
        "--zz",
        type=str,
        default="val",
        choices=["5", "10", "15", "20", "25", "30", "val"],
    )
    parser.add_argument("--savename", type=str, default="bait")
    args = parser.parse_args()

    args.interval = args.max_epoch / 10

    names = ["train", "validation"]
    args.class_num = 12

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    folder = "./data/"
    args.s_dset_path = folder + args.dset + "/" + names[args.s] + "_list.txt"
    args.t_dset_path = folder + args.dset + "/" + names[args.t] + "_list.txt"
    args.test_dset_path = args.t_dset_path

    current_folder = "./ckps/"
    args.output_dir_src = osp.join(
        current_folder, args.da, args.output, args.dset, names[args.s][0].upper()
    )
    args.name_src = names[args.s][0].upper()
    if not osp.exists(args.output_dir_src):
        os.system("mkdir -p " + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)

    args.output_dir = osp.join(
        current_folder,
        args.da,
        args.output,
        args.dset,
        names[args.s][0].upper() + names[args.t][0].upper(),
    )
    args.name = names[args.s][0].upper() + names[args.t][0].upper()
    if not osp.exists(args.output_dir):
        os.system("mkdir -p " + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.out_file = open(
        osp.join(args.output_dir, "log_" + str(args.zz) + "_" + args.savename + ".txt"),
        "w",
    )
    args.out_file.write(print_args(args) + "\n")
    args.out_file.flush()
    train_target_bait(args, "val")
