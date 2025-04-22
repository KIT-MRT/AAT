# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import time
import logging
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from fvcore.nn.precise_bn import get_bn_modules
import numpy as np
import random
from collections import OrderedDict
import torchvision.transforms as transforms

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer
from detectron2.utils.events import EventStorage
from detectron2.evaluation import verify_results, DatasetEvaluators
from detectron2.layers import cat, cross_entropy, nonzero_tuple, batched_nms

# from detectron2.evaluation import COCOEvaluator, verify_results, DatasetEvaluators

from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import hooks
from detectron2.structures.boxes import Boxes, pairwise_iou
from detectron2.structures.instances import Instances
from detectron2.utils.env import TORCH_VERSION
from detectron2.data import MetadataCatalog

from aat.data.build import (
    build_detection_semisup_train_loader,
    build_detection_test_loader,
    build_detection_semisup_train_loader_two_crops,
)
from aat.data.dataset_mapper import DatasetMapperTwoCropSeparate
from aat.engine.hooks import LossEvalHook
from aat.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from aat.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from aat.solver.build import build_lr_scheduler
from aat.evaluation import PascalVOCDetectionEvaluator, COCOEvaluator

from .probe import OpenMatchTrainerProbe
import copy
import math


# Adaptive Teacher Trainer
class ATeacherTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)

        # create an student model
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        # create an teacher model
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # Ensemble teacher and student model is for model saving and loading
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.register_hooks(self.build_hooks())

        # merlin to save memeory
        def inplace_relu(m):
            classname = m.__class__.__name__
            if classname.find("ReLU") != -1:
                m.inplace = True

        # self.probe = OpenMatchTrainerProbe(cfg)
        self.model.apply(inplace_relu)
        self.model_teacher.apply(inplace_relu)

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume:
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc_water":
            return PascalVOCDetectionEvaluator(
                dataset_name,
                target_classnames=["bicycle", "bird", "car", "cat", "dog", "person"],
            )
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparate(cfg, True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()

                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step_full_semisup()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    # =====================================================
    # ================== Pseduo-labeling ==================
    # =====================================================
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

        return new_proposal_inst

    def process_pseudo_label(
        self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            # thresholding
            if psedo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output

    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabled_data

    def get_label(self, label_data):
        label_list = []
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                label_list.append(copy.deepcopy(label_datum["instances"]))

        return label_list

    # def get_label_test(self, label_data):
    #     label_list = []
    #     for label_datum in label_data:
    #         if "instances" in label_datum.keys():
    #             label_list.append(label_datum["instances"])

    # =====================================================
    # =================== Training Flow ===================
    # =====================================================

    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[UBTeacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak
        label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data
        data_time = time.perf_counter() - start

        # burn-in stage (supervised training with labeled data)
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:

            # input both strong and weak supervised data into model
            label_data_q.extend(label_data_k)
            record_dict, _, _, _ = self.model(label_data_q, branch="supervised")

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = record_dict[key] * 1
            losses = sum(loss_dict.values())

        else:
            if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
                # update copy the the whole model
                self._update_teacher_model(keep_rate=0.00)
                # self.model.build_discriminator()

            elif (
                self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
            ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
                self._update_teacher_model(keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)

            record_dict = {}

            gt_unlabel_k = self.get_label(unlabel_data_k)

            #  0. remove unlabeled data labels
            unlabel_data_q = self.remove_label(unlabel_data_q)
            unlabel_data_k = self.remove_label(unlabel_data_k)

            #  1. generate the pseudo-label using teacher model
            with torch.no_grad():
                (
                    _,
                    proposals_rpn_unsup_k,
                    proposals_roih_unsup_k,
                    _,
                ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")

            #  2. Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

            pseudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
                proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
            )

            #  3. add pseudo-label to unlabeled data

            unlabel_data_q = self.add_label(
                unlabel_data_q, pseudo_proposals_roih_unsup_k
            )
            unlabel_data_k = self.add_label(
                unlabel_data_k, pseudo_proposals_roih_unsup_k
            )

            all_label_data = label_data_q + label_data_k
            all_unlabel_data = unlabel_data_q

            #  4. input both strongly and weakly augmented labeled data into student model
            record_all_label_data, _, _, _ = self.model(
                all_label_data, branch="supervised"
            )
            record_dict.update(record_all_label_data)

            #  5. input strongly augmented unlabeled data into model
            record_all_unlabel_data, _, _, _ = self.model(
                all_unlabel_data, branch="supervised_target"
            )
            new_record_all_unlabel_data = {}
            for key in record_all_unlabel_data.keys():
                new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[
                    key
                ]
            record_dict.update(new_record_all_unlabel_data)

            #  6. input weakly labeled data (source) and weakly unlabeled data (target) to student model
            # give sign to the target data
            for i_index in range(len(unlabel_data_k)):
                # unlabel_data_item = {}
                for k, v in unlabel_data_k[i_index].items():
                    label_data_k[i_index][k + "_unlabeled"] = v

            all_domain_data = label_data_k
            record_all_domain_data, _, _, _ = self.model(
                all_domain_data, branch="domain"
            )
            record_dict.update(record_all_domain_data)

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key.startswith("loss"):
                    if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
                        # pseudo bbox regression <- 0
                        loss_dict[key] = record_dict[key] * 0
                    elif key[-6:] == "pseudo":  # unsupervised loss
                        loss_dict[key] = (
                            record_dict[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                        )
                    elif (
                        key == "loss_D_img_s" or key == "loss_D_img_t"
                    ):  # set weight for discriminator
                        # import pdb
                        # pdb.set_trace()
                        loss_dict[key] = (
                            record_dict[key] * self.cfg.SEMISUPNET.DIS_LOSS_WEIGHT
                        )  # Need to modify defaults and yaml
                    else:  # supervised loss
                        loss_dict[key] = record_dict[key] * 1

            losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)
        # all_hg_dict = comm.gather(hg_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            # append the list
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.9996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] * (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _copy_main_model(self):
        # initialize all parameters
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher.load_state_dict(rename_model_dict)
        else:
            self.model_teacher.load_state_dict(self.model.state_dict())

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            (
                hooks.PreciseBN(
                    # Run at the same freq as (but before) evaluation.
                    cfg.TEST.EVAL_PERIOD,
                    self.model,
                    # Build a new data loader to not affect training
                    self.build_train_loader(cfg),
                    cfg.TEST.PRECISE_BN.NUM_ITER,
                )
                if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
                else None
            ),
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(self.cfg, self.model_teacher)
            return self._last_eval_results_teacher

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results_teacher))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret


class ImbalanceMetric(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super().__init__()
        self.register_buffer("rpn", torch.zeros(num_anchors))
        self.register_buffer("roi", torch.zeros((num_classes, num_classes + 1)))


# Targeted Attacked Teacher Trainer


class AATeacherTrainer(ATeacherTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)

        # create an student model
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        # create an teacher model
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.imbalance_metric = ImbalanceMetric(
            len(cfg.MODEL.ANCHOR_GENERATOR.SIZES[0])
            * len(cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS[0]),
            self.num_classes,
        )

        # Ensemble teacher and student model is for model saving and loading
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
            imbalance_metric=self.imbalance_metric,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.register_hooks(self.build_hooks())

        # merlin to save memeory
        def inplace_relu(m):
            classname = m.__class__.__name__
            if classname.find("ReLU") != -1:
                m.inplace = True

        # self.probe = OpenMatchTrainerProbe(cfg)
        self.model.apply(inplace_relu)
        self.model_teacher.apply(inplace_relu)
        self.source_crop_bank = [[] for _ in range(self.num_classes)]
        self.target_crop_bank = [[] for _ in range(self.num_classes)]

    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc.cpu())

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[
                valid_map
            ].cpu()

        return new_proposal_inst

    # =====================================================
    # =================== Training Flow ===================
    # =====================================================

    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[AATeacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak
        label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data
        label_data_q = self.add_cutout(label_data_q)
        label_data_q = self.remove_cutout_objects(label_data_q)
        data_time = time.perf_counter() - start

        # burn-in stage (supervised training with labeled data)
        if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP // 2:
            # update copy the the whole model
            self._update_teacher_model(keep_rate=0.00)
        elif (self.iter > self.cfg.SEMISUPNET.BURN_UP_STEP // 2) and (
            self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
        ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
            self._update_teacher_model(keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)

        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:
            # input both strong and weak supervised data into model
            label_data_q.extend(label_data_k)
            record_dict, local_matrix = self.model(
                label_data_q, branch="supervised", ret_confusion_matrix=True
            )
            self.update_confusion_matrix(local_matrix)

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = record_dict[key] * 1
            losses = sum(loss_dict.values())

        else:
            unlabel_data_q = self.add_cutout(unlabel_data_q)
            record_dict = {}

            #  0. remove unlabeled data labels
            gt_labels = self.get_label(unlabel_data_k)
            unlabel_data_q = self.remove_label(unlabel_data_q)
            unlabel_data_k = self.remove_label(unlabel_data_k)
            self.update_attack_mask_and_weight()

            #  1. input both strongly and weakly augmented labeled data into student model
            all_label_data = label_data_k + label_data_q
            # if self.cfg.SEMISUPNET.PASTE_MINORITY:
            #     source_crops = self.crop_source(label_data_k)
            #     self.source_crop_bank = self.store_crops(source_crops, target=False)
            #     self.paste_minority(label_data_q, target=False)

            record_all_label_data, local_matrix = self.model(
                all_label_data,
                branch="supervised",
                ret_confusion_matrix=True,  # , pertubation=pertubation_label
            )
            record_dict.update(record_all_label_data)
            #  2. calculate the EMA of confusion matrix
            self.update_confusion_matrix(local_matrix)
            #  3. generate the pseudo-label using teacher model
            with torch.no_grad():
                proposals_roih_unsup_k, _ = self.model_teacher(
                    unlabel_data_k, branch="unsup_data_weak"
                )

            #  4. Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

            pseudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
                proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
            )

            unlabel_data_k = self.add_label(
                unlabel_data_k, pseudo_proposals_roih_unsup_k
            )
            unlabel_data_q = self.add_label(
                unlabel_data_q, pseudo_proposals_roih_unsup_k
            )
            unlabel_data_q = self.remove_cutout_objects(unlabel_data_q)
            #  6. input strongly augmented unlabeled data into model
            if (
                not self.cfg.SEMISUPNET.PSEUDO_LABEL_REG
                and self.cfg.SEMISUPNET.PASTE_MINORITY
            ):
                target_crops = self.crop_target(unlabel_data_k)
                self.target_crop_bank = self.store_crops(target_crops, target=True)
                unlabel_data_q = self.paste_minority(unlabel_data_q, target=True)
            record_all_unlabel_data, _ = self.model(
                unlabel_data_q, branch="supervised_target"
            )
            new_record_all_unlabel_data = {}
            for key in record_all_unlabel_data.keys():
                new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[
                    key
                ]
            #  5. conduct targeted attack on unlabel_data_q
            if self.cfg.SEMISUPNET.PSEUDO_LABEL_REG:
                adversarial_pseudo_labels = pseudo_proposals_roih_unsup_k
                pertubation_k, _ = self.model_teacher(unlabel_data_k, branch="attack")
                if self.cfg.SEMISUPNET.USE_SIGN:
                    pertubation_k = torch.sign(pertubation_k)
                pertubation_k *= self.cfg.SEMISUPNET.ATTACK_SEVERITY
                if pertubation_k.any():
                    with torch.no_grad():
                        proposals_roih_attacked_k, _ = self.model_teacher(
                            unlabel_data_k,
                            branch="unsup_data_weak",
                            pertubation=pertubation_k,
                        )
                    # torch.save(proposals_roih_attacked_k, "attacked_pseudo_labels.pt")
                    pseudo_proposals_roih_attacked_k, _ = self.process_pseudo_label(
                        proposals_roih_attacked_k, cur_threshold, "roih", "thresholding"
                    )
                    adversarial_pseudo_labels = self.generate_adversarial_pseudo_labels(
                        unlabel_data_k, pseudo_proposals_roih_attacked_k
                    )

                if pseudo_proposals_roih_unsup_k != adversarial_pseudo_labels:
                    unlabel_data_q = self.remove_label(unlabel_data_q)
                    unlabel_data_q = self.add_label(
                        unlabel_data_q, adversarial_pseudo_labels
                    )
                    unlabel_data_q = self.remove_cutout_objects(unlabel_data_q)
                    if self.cfg.SEMISUPNET.PASTE_MINORITY:
                        unlabel_data_q = self.paste_minority(
                            unlabel_data_q, target=True
                        )

                    record_all_unlabel_data_adv, _ = self.model(
                        unlabel_data_q, branch="supervised_target"
                    )
                    for key in record_all_unlabel_data_adv.keys():
                        new_record_all_unlabel_data[key + "_pseudo"] = (
                            0.5 * new_record_all_unlabel_data[key + "_pseudo"]
                            + 0.5 * record_all_unlabel_data_adv[key]
                        )

            record_dict.update(new_record_all_unlabel_data)

            #  7. input weakly labeled data (source) and weakly unlabeled data (target) to student model
            # give sign to the target data
            for i_index in range(len(unlabel_data_k)):
                for k, v in unlabel_data_k[i_index].items():
                    label_data_k[i_index][k + "_unlabeled"] = v

            all_domain_data = label_data_k
            record_all_domain_data, _ = self.model(all_domain_data, branch="domain")
            record_dict.update(record_all_domain_data)

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key.startswith("loss"):
                    if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
                        loss_dict[key] = record_dict[key] * 0
                    elif key[-6:] == "pseudo":  # unsupervised loss
                        loss_dict[key] = (
                            record_dict[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                        )
                    elif (  # adversarial loss
                        key == "loss_D_img_s" or key == "loss_D_img_t"
                    ):
                        loss_dict[key] = (
                            record_dict[key] * self.cfg.SEMISUPNET.DIS_LOSS_WEIGHT
                        )
                    else:  # supervised loss
                        loss_dict[key] = record_dict[key] * 1

            losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    def update_confusion_matrix(self, local_matrix):
        if comm.get_world_size() > 1:
            dist.all_reduce(local_matrix, op=dist.ReduceOp.SUM)

        local_matrix = local_matrix.cpu()
        # Normalize local matrix
        sum_values = local_matrix.sum(dim=1)
        mask = sum_values != 0
        local_matrix[mask] = local_matrix[mask] / sum_values[mask].view(-1, 1)

        if (self.imbalance_metric.roi == 0).all():
            self.imbalance_metric.roi = local_matrix
        else:
            if self.imbalance_metric.roi.get_device() == -1:
                self.imbalance_metric.roi = self.imbalance_metric.roi.cpu()
            self.imbalance_metric.roi[self.imbalance_metric.roi.sum(dim=1) == 0] = (
                local_matrix[self.imbalance_metric.roi.sum(dim=1) == 0]
            )
            self.imbalance_metric.roi[mask] = (
                self.cfg.SEMISUPNET.EMA_IMBALANCE_METRIC
                * self.imbalance_metric.roi[mask]
                + (1 - self.cfg.SEMISUPNET.EMA_IMBALANCE_METRIC) * local_matrix[mask]
            )

    def update_attack_mask_and_weight(self):
        class_diff = (
            self.imbalance_metric.roi[:, :-1] - self.imbalance_metric.roi[:, :-1].T
        )
        self.attack_mask = torch.ones_like(self.imbalance_metric.roi, dtype=torch.bool)
        self.attack_mask[:, :-1] = class_diff >= 0
        self.major_mask = (
            self.imbalance_metric.roi.diag() > self.imbalance_metric.roi.diag().mean()
        )
        self.attack_weight = torch.sqrt(class_diff.abs()) * class_diff.sign() + 1

    def generate_adversarial_pseudo_labels(
        self, unlabel_data_k, attacked_pseudo_labels
    ):
        adversarial_pseudo_labels = []
        for i in range(len(unlabel_data_k)):
            pseudo_labels = unlabel_data_k[i]["instances"]
            image_shape = pseudo_labels.image_size
            new_proposal_inst_adversarial = Instances(image_shape)
            # Pseudo labels generated by teacher model
            pseudo_boxes = pseudo_labels.gt_boxes.clone()
            pseudo_classes = pseudo_labels.gt_classes.clone()
            # pseudo_labels.gt_weights = torch.ones_like(pseudo_labels.gt_classes, dtype=torch.float)
            # Attacked pseudo labels generated by teacher model
            attacked_boxes = attacked_pseudo_labels[i].gt_boxes.clone()
            attacked_classes = attacked_pseudo_labels[i].gt_classes.clone()
            # If pseudo labels or attacked predictions are empty, only keep the minor attacked predictions in adversarial pseudo labels
            if len(pseudo_labels) == 0 or len(attacked_pseudo_labels[i]) == 0:
                valid_mask = ~self.major_mask[attacked_classes]
                new_proposal_inst_adversarial.gt_boxes = Boxes(
                    attacked_boxes.tensor[valid_mask]
                )
                new_proposal_inst_adversarial.gt_classes = attacked_classes[valid_mask]
                # new_proposal_inst_adversarial.gt_weights = torch.ones(0, dtype=torch.float)
                adversarial_pseudo_labels.append(new_proposal_inst_adversarial)
                continue
            match_quality_matrix = pairwise_iou(attacked_boxes, pseudo_boxes)
            # Find one-to-one matching between attacked and initial pseudo labels
            ious, indices = match_quality_matrix.max(dim=1)
            if len(indices.unique()) != len(indices):
                for unique_index in indices.unique():
                    # Find all positions of this unique index in indices
                    positions = (indices == unique_index).nonzero(as_tuple=True)[0]
                    # If there's only one position, keep it as is
                    if len(positions) != 1:
                        # Find the position with the highest IoU
                        highest_iou_pos = positions[ious[positions].argmax()]
                        positions = positions[positions != highest_iou_pos]
                        indices[positions] = -1
            # Not matched attacked pseudo labels are set as background
            indices[ious < 0.5] = -1
            # Classification results in matched initial pseudo labels
            initial_attacked_classes = pseudo_classes[indices].clone()
            initial_attacked_classes[indices == -1] = self.num_classes
            # If the matched bboxes are attacked as a more major class, remove those proposals from adversarial pseudo labels
            valid_mask = self.attack_mask[attacked_classes, initial_attacked_classes]
            # remove unmatched major attacked pseudo labels from adversarial pseudo labels
            valid_mask[
                torch.logical_and(indices == -1, self.major_mask[attacked_classes])
            ] = False
            if self.cfg.SEMISUPNET.PASTE_MINORITY:
                for j in range(len(valid_mask)):
                    if (
                        attacked_classes[j] == initial_attacked_classes[j]
                        and ~self.major_mask[attacked_classes[j]]
                    ):
                        assert indices[j] != -1
                        box_j = pseudo_boxes.tensor[indices[j]].to(torch.int)
                        x1 = box_j[0]
                        y1 = box_j[1]
                        x2 = box_j[2]
                        y2 = box_j[3]
                        self.target_crop_bank[attacked_classes[j]].append(
                            unlabel_data_k[i]["image"][:, y1:y2, x1:x2]
                        )
                        if len(self.target_crop_bank[attacked_classes[j]]) > 50:
                            self.target_crop_bank[attacked_classes[j]].pop(0)
            # weights = self.attack_weight[attacked_classes[valid_mask], initial_attacked_classes[valid_mask]]
            new_proposal_inst_adversarial.gt_boxes = Boxes(
                attacked_boxes.tensor[valid_mask]
            )
            new_proposal_inst_adversarial.gt_classes = attacked_classes[valid_mask]
            # new_proposal_inst_adversarial.gt_weights = weights
            adversarial_pseudo_labels.append(new_proposal_inst_adversarial)
        return adversarial_pseudo_labels

    def crop_target(self, data_k):
        crops = [[] for _ in range(self.num_classes)]
        for i in range(len(data_k)):
            gt_labels = data_k[i]["instances"]
            for j in range(len(gt_labels)):
                if self.major_mask[gt_labels.gt_classes[j]]:
                    continue
                box_j = gt_labels.gt_boxes.tensor[j].to(torch.int)
                x1 = box_j[0]
                y1 = box_j[1]
                x2 = box_j[2]
                y2 = box_j[3]
                crops[gt_labels.gt_classes[j]].append(
                    data_k[i]["image"][:, y1:y2, x1:x2]
                )
        return crops

    def store_crops(self, crops, target=False):
        if target:
            crop_bank = self.target_crop_bank
        else:
            crop_bank = self.source_crop_bank
        assert len(crops) == self.num_classes
        for c in range(len(crops)):
            if len(crops[c]) == 0:
                continue
            crop_bank[c].extend(crops[c])
            if len(crop_bank[c]) > 50:
                crop_bank[c] = crop_bank[c][-50:]
        return crop_bank

    def erase(self, img_h, img_w, p, scale, ratio, mask, max_rect):
        if torch.rand(1) < p:
            area = img_h * img_w
            log_ratio = torch.log(torch.tensor(ratio))
            for _ in range(10):
                erase_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
                aspect_ratio = torch.exp(
                    torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
                ).item()

                h = int(round(math.sqrt(erase_area * aspect_ratio)))
                w = int(round(math.sqrt(erase_area / aspect_ratio)))
                if not (h < img_h and w < img_w):
                    continue

                i = torch.randint(0, img_h - h + 1, size=(1,)).item()
                j = torch.randint(0, img_w - w + 1, size=(1,)).item()
                mask[i : i + h, j : j + w] = 1
                if max_rect is None or h * w > max_rect[2] * max_rect[3]:
                    max_rect = torch.tensor([i + h / 2, j + w / 2, h, w])
                return mask, max_rect
        return mask, max_rect

    def add_cutout(self, data_q):
        for i in range(len(data_q)):
            cutout_mask = torch.zeros(data_q[i]["image"].shape[-2:], dtype=torch.bool)
            max_rect = None
            h, w = data_q[i]["image"].shape[-2:]
            cutout_mask, max_rect = self.erase(
                h, w, 0.7, [0.05, 0.5], [0.3, 3.3], cutout_mask, max_rect
            )
            cutout_mask, max_rect = self.erase(
                h, w, 0.5, [0.02, 0.2], [0.1, 6], cutout_mask, max_rect
            )
            cutout_mask, max_rect = self.erase(
                h, w, 0.3, [0.02, 0.2], [0.05, 8], cutout_mask, max_rect
            )
            noise = torch.randint(0, 255, data_q[i]["image"].shape, dtype=torch.uint8)
            data_q[i]["image"][cutout_mask.expand(3, -1, -1)] = noise[
                cutout_mask.expand(3, -1, -1)
            ]
            data_q[i]["cutout_mask"] = cutout_mask
            data_q[i]["max_rect"] = max_rect
        return data_q

    def remove_cutout_objects(self, data_q):
        for image_index in range(len(data_q)):
            boxes = data_q[image_index]["instances"].gt_boxes.tensor
            cutout_mask = data_q[image_index]["cutout_mask"]
        if len(boxes):
            valid_mask = torch.ones(len(boxes), dtype=torch.bool)
            for i in range(boxes.shape[0]):
                box_i = boxes[i].to(torch.int)
                x1 = box_i[0]
                y1 = box_i[1]
                x2 = box_i[2]
                y2 = box_i[3]
                cutout_area = cutout_mask[y1:y2, x1:x2].sum()
                ratio = cutout_area / ((y2 - y1) * (x2 - x1))
                if ratio > 0.8:
                    valid_mask[i] = False
            new_instance = Instances(data_q[image_index]["image"].shape[-2:])
            new_instance.gt_boxes = Boxes(boxes[valid_mask])
            new_instance.gt_classes = data_q[image_index]["instances"].gt_classes[
                valid_mask
            ]
            data_q[image_index]["instances"] = new_instance
        return data_q

    def paste_minority(self, data_q, target):
        crop_bank = self.target_crop_bank if target else self.source_crop_bank
        for i in range(len(data_q)):
            c = torch.randint(len(self.major_mask), size=(1,))
            while self.major_mask[c]:
                c = torch.randint(len(self.major_mask), size=(1,))
            if data_q[i]["max_rect"] is None or len(crop_bank[c]) == 0:
                continue
            y_center, x_center, h, w = data_q[i]["max_rect"]

            crop = crop_bank[c][random.randint(0, len(crop_bank[c]) - 1)]
            if h / w < 3 / 4 * crop.shape[-2] / crop.shape[-1]:
                w = h * (4 / 3 * crop.shape[-1] / crop.shape[-2])
            elif h / w > 4 / 3 * crop.shape[-2] / crop.shape[-1]:
                h = w * (4 / 3 * crop.shape[-2] / crop.shape[-1])
            ratio = random.uniform(0.5, 1.0)
            h *= ratio
            w *= ratio
            y1, y2 = int(y_center - h / 2), int(y_center + h / 2)
            x1, x2 = int(x_center - w / 2), int(x_center + w / 2)
            noise_ratio = random.uniform(0.0, 0.7)
            data_q[i]["image"][:, y1:y2, x1:x2] = noise_ratio * data_q[i]["image"][
                :, y1:y2, x1:x2
            ].float() + (1 - noise_ratio) * F.interpolate(
                crop.unsqueeze(0).float(),
                size=(y2 - y1, x2 - x1),
                align_corners=False,
                mode="bilinear",
            ).squeeze(
                0
            )
            data_q[i]["instances"].gt_boxes.tensor = torch.cat(
                [
                    data_q[i]["instances"].gt_boxes.tensor,
                    torch.tensor([[x1, y1, x2, y2]], dtype=torch.float),
                ],
                dim=0,
            )
            data_q[i]["instances"].gt_classes = torch.cat(
                [data_q[i]["instances"].gt_classes, torch.tensor([c], dtype=torch.long)]
            )
        return data_q
