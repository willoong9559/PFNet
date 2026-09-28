"""PFNet with a frozen GASNv2 semantic backbone and polar-BEV instance head."""

import numpy as np
import torch

from .base import Base
from .polar_instance import PolarInstanceHead


class PFNet(Base):
    """Use GASNv2 for semantics and Panoptic-PolarNet-style center voting for instances."""

    def __init__(self, cfg):
        super().__init__(cfg)
        instance_cfg = cfg.MODEL.POLAR_INSTANCE
        self.instance_head = PolarInstanceHead(
            in_channels=64 * 6,
            radial_bins=instance_cfg.RADIAL_BINS,
            angular_bins=instance_cfg.ANGULAR_BINS,
            max_radius=instance_cfg.MAX_RADIUS,
            center_threshold=instance_cfg.CENTER_THRESHOLD,
            nms_kernel=instance_cfg.NMS_KERNEL,
            top_k=instance_cfg.TOP_K,
            center_loss_weight=instance_cfg.CENTER_LOSS_WEIGHT,
            offset_loss_weight=instance_cfg.OFFSET_LOSS_WEIGHT,
            center_sigma=instance_cfg.CENTER_SIGMA,
        )
        self.fix_parameters()

    def fix_parameters(self):
        for name, parameter in self.named_parameters():
            if name.startswith('sem_backbone.'):
                parameter.requires_grad = False

    def forward(self, batch, is_test=False, before_merge_evaluator=None, after_merge_evaluator=None, require_merge=True):
        example = {
            key: [item.cuda() if item is not None else None for item in value]
            for key, value in batch.items()
            if key not in ['i_iter', 'pcd_fname', 'rank', 'epoch']
        }
        self.sem_backbone.eval()
        with torch.no_grad():
            semantic_output = self.sem_backbone(return_loss=False, **example)
            semantic_logits = semantic_output['semantic_logits']

        batch_size = len(example['points'])
        points = torch.cat([point_cloud[:, :3] for point_cloud in example['points']], dim=0)
        batch_idx = torch.cat([
            torch.full((point_cloud.shape[0],), index, dtype=torch.long, device=points.device)
            for index, point_cloud in enumerate(example['points'])
        ])
        semantic_labels = semantic_logits.argmax(dim=1) + self.plus
        instance_labels = None
        if not is_test:
            semantic_labels = torch.cat(example['points_label'], dim=0)
            instance_labels = torch.cat(example['inst_label'], dim=0)

        instance_output = self.instance_head(
            points=points,
            point_features=semantic_output['points_fea'],
            batch_idx=batch_idx,
            semantic_labels=semantic_labels,
            instance_labels=instance_labels,
            compute_loss=not is_test,
        )
        if is_test:
            output = {'loss': points.sum() * 0}
            point_semantics = self.calc_sem_label(semantic_logits, batch, need_add_one=True)
            point_instances = [
                instance_output['instance_ids'][batch_idx == index].detach().cpu().numpy()
                for index in range(batch_size)
            ]
            merged_semantics = self.merge_ins_sem(point_semantics, point_instances) if require_merge else point_semantics
            if before_merge_evaluator is not None:
                self.update_evaluator(before_merge_evaluator, point_semantics, point_instances, batch)
            if after_merge_evaluator is not None:
                self.update_evaluator(after_merge_evaluator, merged_semantics, point_instances, batch)
            output.update(sem_preds=merged_semantics, ins_preds=point_instances)
            return output

        center_loss = instance_output['loss_center']
        offset_loss = instance_output['loss_offset']
        return {'loss_center': center_loss, 'loss_offset': offset_loss, 'loss': center_loss + offset_loss}
