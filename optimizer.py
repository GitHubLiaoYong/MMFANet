"""
Pytorch Optimizer and Scheduler Related Task
"""
import math
import logging
import torch
from torch import optim
from config import cfg


def get_optimizer(args, net):
    """
    Decide Optimizer (Adam or SGD)
    """
    param_groups = net.parameters()

    if args.sgd:
        if args.split_loss == 'Yes':
            fix_lr = args.lr / 10.0
            train_params = [{'params': net.module.get_1x_lr_params(), 'lr': fix_lr},
                            {'params': net.module.get_10x_lr_params(), 'lr': args.lr}]
            # optimizer = torch.optim.SGD(
            #     filter(lambda p: p.requires_grad, model.parameters()), args.lr, momentum=0.9, weight_decay=1e-4)
            optimizer = torch.optim.SGD(
                train_params, args.lr, momentum=0.9, weight_decay=1e-4)
        else:
            optimizer = optim.SGD(param_groups,
                                lr=args.lr,
                                weight_decay=args.weight_decay,
                                momentum=args.momentum,
                                nesterov=False)
    elif args.adam:
        amsgrad = False
        if args.amsgrad:
            amsgrad = True
        optimizer = optim.Adam(param_groups,
                               lr=args.lr,
                               weight_decay=args.weight_decay,
                               amsgrad=amsgrad
                               )
    else:
        raise ValueError('Not a valid optimizer')

    if args.lr_schedule == 'scl-poly':
        if cfg.REDUCE_BORDER_EPOCH == -1:
            raise ValueError('ERROR Cannot Do Scale Poly')

        rescale_thresh = cfg.REDUCE_BORDER_EPOCH
        scale_value = args.rescale
        lambda1 = lambda epoch: \
             math.pow(1 - epoch / args.max_epoch,
                      args.poly_exp) if epoch < rescale_thresh else scale_value * math.pow(
                          1 - (epoch - rescale_thresh) / (args.max_epoch - rescale_thresh),
                          args.repoly)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    elif args.lr_schedule == 'poly':
        lambda1 = lambda epoch: math.pow(1 - epoch / args.max_epoch, args.poly_exp)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    else:
        raise ValueError('unknown lr schedule {}'.format(args.lr_schedule))

    return optimizer, scheduler


def load_weights(net, optimizer, snapshot_file, restore_optimizer_bool=False):
    """
    Load weights from snapshot file
    """
    logging.info("Loading weights from model %s", snapshot_file)
    net, optimizer = restore_snapshot(net, optimizer, snapshot_file, restore_optimizer_bool)
    return net, optimizer


def restore_snapshot(net, optimizer, snapshot, restore_optimizer_bool):
    """
    Restore weights and optimizer (if needed ) for resuming job.
    """
    checkpoint = torch.load(snapshot, map_location=torch.device('cpu'))
    logging.info("Checkpoint Load Compelete")
    if optimizer is not None and 'optimizer' in checkpoint and restore_optimizer_bool:
        logging.info('optimizer load state dict')
        optimizer.load_state_dict(checkpoint['optimizer'])#加载优化器的状态
        logging.info('optimizer load complete')

    if 'state_dict' in checkpoint:
        net = forgiving_state_restore(net, checkpoint['state_dict'])
    elif 'model' in checkpoint:
        net = forgiving_state_restore(net,checkpoint['model'])
    else:
        net = forgiving_state_restore(net, checkpoint)

    return net, optimizer


def forgiving_state_restore(net, loaded_dict):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """
    net_state_dict = net.state_dict()
    # for k in net_state_dict:
    #     print(k)
    # for k in loaded_dict:
    #     print(k)

    new_loaded_dict = {}
    for k in net_state_dict:#k前面是有module. 的
        if k in loaded_dict and net_state_dict[k].size() == loaded_dict[k].size():#[7:]我加的
            new_loaded_dict[k] = loaded_dict[k]#'module.'+k是我加的，原来是没有的
            logging.info("Loading key: %s ", k)
        else:
            logging.info("Skipped loading parameter %s", k)
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)
    return net
