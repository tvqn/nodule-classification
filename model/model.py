import torch
from torch import nn

import resnet
import resnext
import no_bias_resnet

def generate_model(opt):
    assert opt.mode in ['score', 'feature']
    if opt.mode == 'score':
        last_fc = True
    elif opt.mode == 'feature':
        last_fc = False

    assert opt.model_name in ['resnet', 'densenet']

    if opt.model_name == 'resnet':
        assert opt.model_depth in [10, 16, 18, 34]

        if opt.model_depth == 10:
            if opt.bias:
                model = resnet.resnet10(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                        sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                        last_fc=last_fc)
            else:
                model = no_bias_resnet.resnet10(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                        sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                        last_fc=last_fc)
        elif opt.model_depth == 16:
            model = resnet.resnet16(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                    sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                    last_fc=last_fc)
        elif opt.model_depth == 18:
            if opt.bias:
                model = resnet.resnet18(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                        sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                        last_fc=last_fc)
            else:
                model = no_bias_resnet.resnet18(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                        sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                        last_fc=last_fc)
        elif opt.model_depth == 34:
            model = resnet.resnet34(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                    sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                    last_fc=last_fc)
    elif opt.model_name == 'densenet':
        assert opt.model_depth in [121, 169, 201, 264]

        if opt.model_depth == 121:
            model = densenet.densenet121(num_classes=opt.n_classes,
                                         sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                         last_fc=last_fc)
        elif opt.model_depth == 169:
            model = densenet.densenet169(num_classes=opt.n_classes,
                                         sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                         last_fc=last_fc)
        elif opt.model_depth == 201:
            model = densenet.densenet201(num_classes=opt.n_classes,
                                         sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                         last_fc=last_fc)
        elif opt.model_depth == 264:
            model = densenet.densenet264(num_classes=opt.n_classes,
                                         sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                         last_fc=last_fc)

    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)

    return model
