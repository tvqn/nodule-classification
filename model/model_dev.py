import torch
from torch import nn

import resnet_dev
import pre_act_resnet
import resnext
import wide_resnet

def generate_model(opt):
    assert opt.mode in ['score', 'feature']
    if opt.mode == 'score':
        last_fc = True
    elif opt.mode == 'feature':
        last_fc = False

    assert opt.model_name in ['resnet', 'preresnet', 'wideresnet', 'resnext', 'test']
    if opt.model_name == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]
        if opt.model_depth == 10:
            if opt.name_architecture == 'bottleneck':
                model = resnet_dev.resnet10_bottleneck(num_classes=opt.n_classes, 
                                    shortcut_type=opt.resnet_shortcut,
                                    sample_size=opt.sample_size,
                                    last_fc=last_fc)
            elif opt.name_architecture == 'basic':
                model = resnet_dev.resnet10_basic(num_classes=opt.n_classes, 
                                            shortcut_type=opt.resnet_shortcut,
                                            sample_size=opt.sample_size,
                                            last_fc=last_fc)
    elif opt.model_name == 'wideresnet':
        assert opt.model_depth in [50]

        if opt.model_depth == 50:
            model = wide_resnet.resnet50(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut, k=opt.wide_resnet_k,
                                         sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                         last_fc=last_fc)
    elif opt.model_name == 'resnext':
        assert opt.model_depth in [50, 101, 152]

        if opt.model_depth == 50:
            model = resnext.resnet50(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut, cardinality=opt.resnext_cardinality,
                                     sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                     last_fc=last_fc)
        elif opt.model_depth == 101:
            model = resnext.resnet101(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut, cardinality=opt.resnext_cardinality,
                                      sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                      last_fc=last_fc)
        elif opt.model_depth == 152:
            model = resnext.resnet152(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut, cardinality=opt.resnext_cardinality,
                                      sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                      last_fc=last_fc)
    elif opt.model_name == 'preresnet':
        assert opt.model_depth in [18, 34, 50, 101, 152, 200]

        if opt.model_depth == 18:
            model = pre_act_resnet.resnet18(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                            sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                            last_fc=last_fc)
        elif opt.model_depth == 34:
            model = pre_act_resnet.resnet34(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                            sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                            last_fc=last_fc)
        elif opt.model_depth == 50:
            model = pre_act_resnet.resnet50(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                            sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                            last_fc=last_fc)
        elif opt.model_depth == 101:
            model = pre_act_resnet.resnet101(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                             sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                             last_fc=last_fc)
        elif opt.model_depth == 152:
            model = pre_act_resnet.resnet152(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                             sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                             last_fc=last_fc)
        elif opt.model_depth == 200:
            model = pre_act_resnet.resnet200(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                             sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                             last_fc=last_fc)

    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)

    return model
