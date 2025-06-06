# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_ateacher_config(cfg):
    """
    Add config for semisupnet.
    """
    _C = cfg
    _C.TEST.VAL_LOSS = True

    _C.SOLVER.IMG_PER_BATCH_LABEL = 1
    _C.SOLVER.IMG_PER_BATCH_UNLABEL = 1
    _C.SOLVER.FACTOR_LIST = (1,)

    _C.DATASETS.TRAIN_LABEL = ("coco_2017_train",)
    _C.DATASETS.TRAIN_UNLABEL = ("coco_2017_train",)
    _C.DATASETS.CROSS_DATASET = True
    _C.TEST.EVALUATOR = "COCOeval"

    # Semi-supervised training
    _C.SEMISUPNET = CN()
    # Output dimension of the MLP projector after `res5` block
    _C.SEMISUPNET.MLP_DIM = 128
    _C.SEMISUPNET.Trainer = "ateacher"
    _C.SEMISUPNET.BBOX_THRESHOLD = 0.7
    _C.SEMISUPNET.PSEUDO_BBOX_SAMPLE = "thresholding"
    _C.SEMISUPNET.TEACHER_UPDATE_ITER = 1
    _C.SEMISUPNET.BURN_UP_STEP = 12000
    _C.SEMISUPNET.EMA_KEEP_RATE = 0.0
    _C.SEMISUPNET.EMA_IMBALANCE_METRIC = 0.99
    _C.SEMISUPNET.UNSUP_LOSS_WEIGHT = 4.0
    _C.SEMISUPNET.SUP_LOSS_WEIGHT = 0.5
    _C.SEMISUPNET.LOSS_WEIGHT_TYPE = "standard"
    _C.SEMISUPNET.DIS_TYPE = "res4"
    _C.SEMISUPNET.DIS_LOSS_WEIGHT = 0.1
    _C.SEMISUPNET.PSEUDO_LABEL_REG = False
    _C.SEMISUPNET.PASTE_MINORITY = False
    _C.SEMISUPNET.ATTACK_SEVERITY = 0.1
    _C.SEMISUPNET.USE_SIGN = False
