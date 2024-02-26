import argparse
import numpy as np
import torch

_parser = argparse.ArgumentParser(description='Configuration for LibMTL')
# general
_parser.add_argument('--seed', type=int, default=0, help='random seed')
_parser.add_argument('--gpu_id', default='0', type=str, help='gpu_id')
_parser.add_argument('--weighting', type=str, default='EW',
                     help='loss weighing strategies, option: EW, UW, GradNorm, GLS, RLW, \
        MGDA, PCGrad, GradVac, CAGrad, GradDrop, DWA, IMTL')
_parser.add_argument('--arch', type=str, default='HPS',
                     help='architecture for MTL, option: HPS, MTAN')
_parser.add_argument('--rep_grad', action='store_true', default=False,
                     help='computing gradient for representation or sharing parameters')
_parser.add_argument('--multi_input', action='store_true', default=False,
                     help='whether each task has its own input data')
# optim
_parser.add_argument('--optim', type=str, default='adam',
                     help='optimizer for training, option: adam, sgd, adagrad, rmsprop')
_parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for all types of optim')
_parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')
_parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay for all types of optim')
# scheduler
_parser.add_argument('--scheduler', type=str,  # default='step',
                     help='learning rate scheduler for training, option: step, cos, exp')
_parser.add_argument('--step_size', type=int, default=100, help='step size for StepLR')
_parser.add_argument('--gamma', type=float, default=0.5, help='gamma for StepLR')

# args for weighting
# DWA
_parser.add_argument('--T', type=float, default=2.0, help='T for DWA')
# MGDA
_parser.add_argument('--mgda_gn', default='none', type=str,
                     help='type of gradient normalization for MGDA, option: l2, none, loss, loss+')
# GradVac
_parser.add_argument('--beta', type=float, default=0.5, help='beta for GradVac')
# GradNorm
_parser.add_argument('--alpha', type=float, default=1.5, help='alpha for GradNorm')
# GradDrop
_parser.add_argument('--leak', type=float, default=0.0, help='leak for GradDrop')
# CAGrad
_parser.add_argument('--calpha', type=float, default=0.5, help='calpha for CAGrad')
_parser.add_argument('--rescale', type=int, default=1, help='rescale for CAGrad')
# Nash_MTL
_parser.add_argument('--update_weights_every', type=int, default=1, help='update_weights_every for Nash_MTL')
_parser.add_argument('--optim_niter', type=int, default=20, help='optim_niter for Nash_MTL')
_parser.add_argument('--max_norm', type=float, default=1.0, help='max_norm for Nash_MTL')
# GCS
_parser.add_argument('--pri_tasks', default=[1, 0, 0], type=float, nargs='+',
                     help='binary mask which indicates primary task (only 1 task)')
_parser.add_argument('--gcs_method', default='unweighted', type=str, choices=['unweighted', 'weighted'],
                     help='strategy of GCS')
# PriPCGrad
_parser.add_argument('--pri_idx', default=0, type=int)

# args for architecture
# CGC
_parser.add_argument('--img_size', nargs='+', help='image size for CGC')
_parser.add_argument('--num_experts', nargs='+', help='the number of experts for sharing and task-specific')
# DSelect_k
_parser.add_argument('--num_nonzeros', type=int, default=2, help='num_nonzeros for DSelect-k')
_parser.add_argument('--kgamma', type=float, default=1.0, help='gamma for DSelect-k')

LibMTL_args = _parser


def prepare_args(params):
    r"""Return the configuration of hyperparameters, optimizier, and learning rate scheduler.

    Args:
        params (argparse.Namespace): The command-line arguments.
    """
    kwargs = {'weight_args': {}, 'arch_args': {}}
    if params.weighting in ['EW', 'UW', 'GradNorm', 'GLS', 'RLW', 'MGDA', 'IMTL',
                            'PCGrad', 'GradVac', 'CAGrad', 'GradDrop', 'DWA', 'Nash_MTL', 'GCS', 'PriPCGrad',
                            'PriGradVac', 'OLAUX', 'ARML']:
        if params.weighting in ['EW']:
            assert len(params.tasks) == len(params.task_weights)
            kwargs['weight_args']['weights'] = params.task_weights
        if params.weighting in ['DWA']:
            if params.T is not None:
                kwargs['weight_args']['T'] = params.T
            else:
                raise ValueError('DWA needs keyword T')
        elif params.weighting in ['GradNorm']:
            if params.alpha is not None:
                kwargs['weight_args']['alpha'] = params.alpha
            else:
                raise ValueError('GradNorm needs keywaord alpha')
        elif params.weighting in ['MGDA']:
            if params.mgda_gn is not None:
                if params.mgda_gn in ['none', 'l2', 'loss', 'loss+']:
                    kwargs['weight_args']['mgda_gn'] = params.mgda_gn
                else:
                    raise ValueError('No support mgda_gn {} for MGDA'.format(params.mgda_gn))
            else:
                raise ValueError('MGDA needs keywaord mgda_gn')
        elif params.weighting in ['GradVac']:
            if params.beta is not None:
                kwargs['weight_args']['beta'] = params.beta
            else:
                raise ValueError('GradVac needs keywaord beta')
        elif params.weighting in ['GradDrop']:
            if params.leak is not None:
                kwargs['weight_args']['leak'] = params.leak
            else:
                raise ValueError('GradDrop needs keywaord leak')
        elif params.weighting in ['CAGrad']:
            if params.calpha is not None and params.rescale is not None:
                kwargs['weight_args']['calpha'] = params.calpha
                kwargs['weight_args']['rescale'] = params.rescale
            else:
                raise ValueError('CAGrad needs keywaord calpha and rescale')
        elif params.weighting in ['Nash_MTL']:
            if params.update_weights_every is not None and params.optim_niter is not None and params.max_norm is not None:
                kwargs['weight_args']['update_weights_every'] = params.update_weights_every
                kwargs['weight_args']['optim_niter'] = params.optim_niter
                kwargs['weight_args']['max_norm'] = params.max_norm
            else:
                raise ValueError('Nash_MTL needs update_weights_every, optim_niter, and max_norm')
        elif params.weighting in ['GCS']:
            assert len(params.tasks) == len(params.pri_tasks)
            assert sum(params.pri_tasks) == 1
            kwargs['weight_args']['pri_tasks'] = params.pri_tasks
            kwargs['weight_args']['gcs_method'] = params.gcs_method
        elif params.weighting in ['PriPCGrad']:
            kwargs['weight_args']['pri_idx'] = params.pri_idx
        elif params.weighting in ['PriGradVac']:
            kwargs['weight_args']['beta'] = params.beta
            kwargs['weight_args']['pri_idx'] = params.pri_idx
        elif params.weighting in ['OLAUX']:
            kwargs['weight_args']['beta'] = params.beta
            kwargs['weight_args']['pri_idx'] = params.pri_idx
        elif params.weighting in ['ARML']:
            kwargs['weight_args']['pri_idx'] = params.pri_idx
    else:
        raise ValueError('No support weighting method {}'.format(params.weighting))

    if params.arch in ['HPS', 'Cross_stitch', 'MTAN', 'CGC', 'PLE', 'MMoE', 'DSelect_k', 'DIY', 'LTB']:
        if params.arch in ['CGC', 'PLE', 'MMoE', 'DSelect_k']:
            kwargs['arch_args']['img_size'] = tuple(params.img_size)  # np.array(params.img_size, dtype=int).prod()
            kwargs['arch_args']['num_experts'] = [int(num) for num in params.num_experts]
        if params.arch in ['DSelect_k']:
            kwargs['arch_args']['kgamma'] = params.kgamma
            kwargs['arch_args']['num_nonzeros'] = params.num_nonzeros
    else:
        raise ValueError('No support architecture method {}'.format(params.arch))

    if params.optim in ['adam', 'sgd', 'adagrad', 'rmsprop']:
        if params.optim == 'adam':
            optim_param = {'optim': 'adam', 'lr': params.lr, 'weight_decay': params.weight_decay}
        elif params.optim == 'sgd':
            optim_param = {'optim': 'sgd', 'lr': params.lr,
                           'weight_decay': params.weight_decay, 'momentum': params.momentum}
    else:
        raise ValueError('No support optim method {}'.format(params.optim))

    if params.scheduler is not None:
        if params.scheduler in ['step', 'cos', 'exp']:
            if params.scheduler == 'step':
                scheduler_param = {'scheduler': 'step', 'step_size': params.step_size, 'gamma': params.gamma}
        else:
            raise ValueError('No support scheduler method {}'.format(params.scheduler))
    else:
        scheduler_param = None

    _display(params, kwargs, optim_param, scheduler_param)

    return kwargs, optim_param, scheduler_param


def _display(params, kwargs, optim_param, scheduler_param):
    print('=' * 40)
    print('General Configuration:')
    print('\tWighting:', params.weighting)
    print('\tArchitecture:', params.arch)
    print('\tRep_Grad:', params.rep_grad)
    print('\tMulti_Input:', params.multi_input)
    print('\tSeed:', params.seed)
    print('\tDevice: {}'.format('cuda:' + params.gpu_id if torch.cuda.is_available() else 'cpu'))
    for wa, p in zip(['weight_args', 'arch_args'], [params.weighting, params.arch]):
        if kwargs[wa] != {}:
            print('{} Configuration:'.format(p))
            for k, v in kwargs[wa].items():
                print('\t' + k + ':', v)
    print('Optimizer Configuration:')
    for k, v in optim_param.items():
        print('\t' + k + ':', v)
    if scheduler_param is not None:
        print('Scheduler Configuration:')
        for k, v in scheduler_param.items():
            print('\t' + k + ':', v)
