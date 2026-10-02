import numpy as np
import torch

from utils import common_utils
from utils.evaluate_panoptic import eval_one_scan_w_fname

from .pcd_seg_v2 import GASNv2


class Base(torch.nn.Module):
    """Shared frozen GASNv2 semantic backbone and panoptic result helpers."""

    def __init__(self, cfg):
        super().__init__()
        sem_cfg = dict(
            lims=cfg.MODEL.LIMS,
            offset=cfg.MODEL.OFFSET,
            target_scale=cfg.MODEL.TARGET_SCALE,
            grid_meters=cfg.MODEL.GRID_METERS,
            scales=cfg.MODEL.SCALES,
            pooling_scale=cfg.MODEL.POOLING_SCALE,
            sizes=cfg.MODEL.get('SIZES', None),
            n_class=cfg.MODEL.NCLASS,
            pretrained=cfg.MODEL.get('SEM_PRETRAIN', None),
        )
        self.plus = 1
        self.sem_backbone = GASNv2(sem_cfg)
        self.merge_func_name = cfg.MODEL.POST_PROCESSING.MERGE_FUNC

    def update_evaluator(self, evaluator, sem_preds, ins_preds, inputs):
        for index in range(len(sem_preds)):
            eval_one_scan_w_fname(
                evaluator,
                inputs['points_label'][index].cpu().detach().numpy().reshape(-1),
                inputs['inst_label'][index].cpu().detach().numpy().reshape(-1),
                sem_preds[index],
                ins_preds[index],
                inputs['pcd_fname'][index],
            )

    def merge_ins_sem(self, sem_preds, pred_ins_ids, logits=None, inputs=None):
        merged_sem_preds = []
        for index in range(len(sem_preds)):
            if self.merge_func_name == 'merge_ins_sem':
                merged_sem_preds.append(common_utils.merge_ins_sem(sem_preds[index], pred_ins_ids[index]))
            elif self.merge_func_name == 'merge_ins_sem_logits_size_based':
                merged_sem_preds.append(
                    common_utils.merge_ins_sem_logits_size_based(
                        sem_preds[index], pred_ins_ids[index], index, logits, inputs
                    )
                )
            elif self.merge_func_name == 'none':
                merged_sem_preds.append(sem_preds[index])
            else:
                raise ValueError('Unknown semantic merge function: {}'.format(self.merge_func_name))
        return merged_sem_preds

    def calc_sem_label(self, sem_logits, inputs, need_add_one=True):
        point_labels = torch.argmax(sem_logits, dim=1).cpu().detach().numpy()
        if need_add_one:
            point_labels += self.plus

        labels_by_scan = []
        start_idx = 0
        for points in inputs['points']:
            end_idx = start_idx + len(points)
            labels_by_scan.append(point_labels[start_idx:end_idx])
            start_idx = end_idx
        return labels_by_scan

    def forward(self, batch, is_test=False, before_merge_evaluator=None, after_merge_evaluator=None, require_merge=True):
        example = {
            key: [item.cuda() if item is not None else None for item in value]
            for key, value in batch.items()
            if key not in ['i_iter', 'pcd_fname', 'rank', 'epoch']
        }
        if not is_test:
            loss_dict = self.sem_backbone(return_loss=True, **example)
            output = dict(loss_dict)
            output['loss'] = sum(loss_dict.values())
            return output

        semantic_output = self.sem_backbone(return_loss=False, **example)
        sem_preds = self.calc_sem_label(semantic_output['semantic_logits'], batch)
        ins_preds = [np.zeros_like(sem_pred) for sem_pred in sem_preds]
        if after_merge_evaluator is not None:
            self.update_evaluator(after_merge_evaluator, sem_preds, ins_preds, batch)
        return {
            'loss': torch.zeros(1, requires_grad=True, device=semantic_output['semantic_logits'].device)[0],
            'sem_preds': sem_preds,
            'ins_preds': ins_preds,
        }
