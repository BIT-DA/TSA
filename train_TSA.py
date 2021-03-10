# -*- coding: utf-8 -*

import random
import time
import warnings
import sys
import argparse
import copy

import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F
import os.path as osp
import gc

from network import ImageClassifier
import backbone as BackboneNetwork
from utils import ContinuousDataloader
from transforms import ResizeImage
from lr_scheduler import LrScheduler
from data_list_index import ImageList
from Loss import *

def get_current_time():
    time_stamp = time.time()
    local_time = time.localtime(time_stamp)
    str_time = time.strftime('%Y-%m-%d_%H-%M-%S', local_time)
    return str_time

def main(args: argparse.Namespace, config):
    torch.multiprocessing.set_sharing_strategy('file_system')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    cudnn.benchmark = True

    # load data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if args.dset == "visda":
        train_transform = transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        train_transform = transforms.Compose([
            ResizeImage(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
            
    val_tranform = transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

    train_source_dataset = ImageList(open(args.s_dset_path).readlines(), transform=train_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    if args.dset == "visda":
        memory_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                          shuffle=False, num_workers=64, drop_last=False)
    else:
        memory_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                          shuffle=False, num_workers=args.workers, drop_last=False)

    train_target_dataset = ImageList(open(args.t_dset_path).readlines(), transform=train_transform)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)

    val_dataset = ImageList(open(args.t_dset_path).readlines(), transform=val_tranform)
    if args.dset == "visda":
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=64)
    else:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = val_loader

    train_source_iter = ContinuousDataloader(train_source_loader)
    train_target_iter = ContinuousDataloader(train_target_loader)

    s_len = train_source_dataset.__len__()
    t_len = val_dataset.__len__()

    # load model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = BackboneNetwork.__dict__[args.arch](pretrained=True)
    if args.dset == "office":
        num_classes = 31
    elif args.dset == "office-home":
        num_classes = 65
    elif args.dset == "visda":
        num_classes = 12
    classifier = ImageClassifier(backbone, num_classes).cuda()
    classifier_feature_dim = classifier.features_dim


    # define optimizer and lr scheduler
    all_parameters = classifier.get_parameters()
    optimizer = SGD(all_parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_sheduler = LrScheduler(optimizer, init_lr=args.lr, gamma=0.001, decay_rate=0.75)

    # initialize the memory module
    memory_target_features = torch.zeros(t_len, classifier_feature_dim).cuda()
    memory_target_labels = torch.zeros(t_len).long().cuda()
    flag = False
    for _, (images, label, index) in enumerate(val_loader):
        del _
        images = images.cuda()
        if images.size(0) == 1:
            temp_iter_val = iter(val_loader)
            images_a, _, _ = temp_iter_val.next()
            images_a = images_a.cuda()
            images = torch.cat((images, images_a), dim=0)
            flag = True
            del temp_iter_val
            del _
        with torch.no_grad():
            predictions, features = classifier(images)
            pseudo_labels = predictions.argmax(1)
            if flag:
                memory_target_features[index] = features[0].unsqueeze(0)
                memory_target_labels[index] = pseudo_labels[0].unsqueeze(0)
                flag = False
            else:
                memory_target_features[index] = features
                memory_target_labels[index] = pseudo_labels
        gc.collect()

    memory_source_features = torch.zeros(s_len, classifier_feature_dim).cuda()
    memory_source_labels = torch.zeros(s_len).long().cuda()
    flag = False
    for _, (images, label, index) in enumerate(memory_source_loader):
        del _
        images = images.cuda()
        label = label.cuda()
        if images.size(0) == 1:
            temp_iter = iter(memory_source_loader)
            images_a, _, _ = temp_iter.next()
            images_a = images_a.cuda()
            images = torch.cat((images, images_a), dim=0)
            flag = True
            del temp_iter
            del _
        with torch.no_grad():
            _, features = classifier(images)
            del _
            if flag:
                memory_source_features[index] = features[0].unsqueeze(0)
                memory_source_labels[index] = label
                flag = False
            else:
                memory_source_features[index] = features
                memory_source_labels[index] = label
        gc.collect()
    del memory_source_loader
    print("memory module initialization has finished!")

    # start training
    best_acc1 = 0.
    cls_criterion = Cls_Loss(num_classes).cuda()
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, optimizer,
              lr_sheduler, epoch, args, cls_criterion, memory_source_features, memory_source_labels,
              memory_target_features, memory_target_labels)

        # evaluate on validation set
        if args.dset == "visda":
            acc1 = validate_visda(val_loader, classifier, epoch, config)
        else:
            acc1 = validate(val_loader, classifier, args)

        # remember the best top1 accuracy and checkpoint
        if acc1 > best_acc1:
            best_model = copy.deepcopy(classifier.state_dict())
        best_acc1 = max(acc1, best_acc1)
        print("epoch = {:02d},  acc1={:.3f}, best_acc1 = {:.3f}".format(epoch, acc1, best_acc1))
        config["out_file"].write("epoch = {:02d},  best_acc1 = {:.3f}, best_acc1 = {:.3f}".format(epoch, acc1, best_acc1) + '\n')
        config["out_file"].flush()

    print("best_acc1 = {:.3f}".format(best_acc1))
    config["out_file"].write("best_acc1 = {:.3f}".format(best_acc1) + '\n')
    config["out_file"].flush()

    # evaluate on test set
    classifier.load_state_dict(best_model)
    if args.dset == "visda":
        acc1 = validate_visda(test_loader, classifier, epoch, config)
    else:
        acc1 = validate(test_loader, classifier, args)
    print("test_acc1 = {:.3f}".format(acc1))
    config["out_file"].write("test_acc1 = {:.3f}".format(acc1) + '\n')
    config["out_file"].flush()

def train(train_source_iter: ContinuousDataloader, train_target_iter: ContinuousDataloader, model: ImageClassifier,
        optimizer: SGD, lr_sheduler: LrScheduler, epoch: int, args: argparse.Namespace, cls_criterion, memory_source_features,
          memory_source_labels, memory_target_features, memory_target_labels):
    # switch to train mode
    model.train()
    max_iters = args.iters_per_epoch * args.epochs
    for i in range(args.iters_per_epoch):
        current_iter = i + args.iters_per_epoch * epoch
        Lambda = args.lambda0 * (float(current_iter) / float(max_iters))

        lr_sheduler.step()

        x_s, labels_s, idx_source = next(train_source_iter)
        x_t, _ , idx_target = next(train_target_iter)

        x_s = x_s.cuda()
        x_t = x_t.cuda()
        labels_s = labels_s.cuda()

        # get features and logit outputs
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)

        # update the memory module
        memory_source_features[idx_source] = f_s
        memory_target_features[idx_target] = f_t
        memory_target_labels[idx_target] = y_t.argmax(1)

        # estimate the mean and covariance
        class_num = y_t.size(1)
        mean_source = CalculateMean(memory_source_features, memory_source_labels, class_num)
        mean_target = CalculateMean(memory_target_features, memory_target_labels, class_num)
        cv_target = Calculate_CV(memory_target_features, memory_target_labels, mean_target, class_num)

        # compute loss
        cls_loss = cls_criterion(model.head, f_s, y_s, labels_s, Lambda, mean_source, mean_target, cv_target)
        MI_loss = MI(y_t)
        total_loss = cls_loss - args.MI * MI_loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # print training log
        if i % args.print_freq == 0:
            print("Epoch: [{:02d}][{}/{}]	total_loss:{:.3f}	cls_loss:{:.3f}	 MI_loss:{:.3f}".format(\
                epoch, i, args.iters_per_epoch, total_loss, cls_loss, MI_loss))

def validate(val_loader: DataLoader, model: ImageClassifier, args: argparse.Namespace) -> float:
    # switch to evaluate mode
    model.eval()
    start_test = True
    with torch.no_grad():
        for i, (images, target, _) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # get logit outputs
            output, _ = model(images)
            if start_test:
                all_output = output.float()
                all_label = target.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, output.float()), 0)
                all_label = torch.cat((all_label, target.float()), 0)
        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        accuracy = accuracy * 100.0
        print(' accuracy:{:.3f}'.format(accuracy))
    return accuracy

def validate_visda(val_loader, model, epoch, config):
    dict = {0: "plane", 1: "bcybl", 2: "bus", 3: "car", 4: "horse", 5: "knife", 6: "mcyle", 7: "person", 8: "plant", \
            9: "sktb", 10: "train", 11: "truck"}
    model.eval()
    with torch.no_grad():
        tick = 0
        subclasses_correct = np.zeros(12)
        subclasses_tick = np.zeros(12)
        for i, (imgs, labels, _) in enumerate(val_loader):
            tick += 1
            imgs = imgs.cuda()
            pred, _ = model(imgs)
            pred = nn.Softmax(dim=1)(pred)
            pred = pred.data.cpu().numpy()
            pred = pred.argmax(axis=1)
            labels = labels.numpy()
            for i in range(pred.size):
                subclasses_tick[labels[i]] += 1
                if pred[i] == labels[i]:
                    subclasses_correct[pred[i]] += 1
        subclasses_result = np.divide(subclasses_correct, subclasses_tick)
        print("Epoch [:02d]:".format(epoch))
        for i in range(12):
            log_str1 = '\t{}----------({:.3f})'.format(dict[i], subclasses_result[i] * 100.0)
            print(log_str1)
            config["out_file"].write(log_str1 + "\n")
        avg = subclasses_result.mean()
        avg = avg * 100.0
        log_avg = '\taverage:{:.3f}'.format(avg)
        print(log_avg)
        config["out_file"].write(log_avg + "\n")
        config["out_file"].flush()
    return avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transferable Semantic Augmentation for Domain Adaptation')
    parser.add_argument('--arch', type=str, default='resnet50', choices=['resnet50', 'resnet101'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='5', help="device id to run")
    parser.add_argument('--dset', type=str, default='office', choices=['office', 'visda', 'office-home'], help="The dataset used")
    parser.add_argument('--s_dset_path', type=str, default='/data1/TL/data/list/office/webcam_31.txt', help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='/data1/TL/data/list/office/amazon_31.txt', help="The target dataset path list")
    parser.add_argument('--output_dir', type=str, default='log/office31', help="output directory of logs")
    parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=40, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--iters-per-epoch', default=500, type=int, help='Number of iterations per epoch')
    parser.add_argument('--print-freq', default=100, type=int, metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--batch-size', default=32, type=int, metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--lr', default=0.01, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', default=1e-3, type=float, metavar='W', help='weight decay (default: 1e-3)', dest='weight_decay')
    parser.add_argument('--seed', default=1, type=int, help='seed for initializing training. ')
    parser.add_argument('--lambda0', type=float, default=0.25, help="hyper-parameter: lambda0")
    parser.add_argument('--MI', type=float, default=0.1, help="MI_loss_tradeoff")
    args = parser.parse_args()

    config = {}
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)
    task = args.s_dset_path.split('/')[-1].split('.')[0].split('_')[0] + "-" + \
           args.t_dset_path.split('/')[-1].split('.')[0].split('_')[0]
    config["out_file"] = open(osp.join(args.output_dir, get_current_time() + "_" + task + "_log.txt"), "w")

    for arg in vars(args):
        print("{} = {}".format(arg, getattr(args, arg)))
        config["out_file"].write(str("{} = {}".format(arg, getattr(args, arg))) + "\n")
    config["out_file"].flush()
    main(args, config)
