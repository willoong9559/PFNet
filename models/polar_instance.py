"""Panoptic-PolarNet-style instance head for point features."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter


class PolarInstanceHead(nn.Module):
    """Predict a polar-BEV center heatmap and offsets from GASN point features."""

    def __init__(
        self, in_channels, radial_bins=480, angular_bins=360, max_radius=70.0,
        thing_classes=(1, 2, 3, 4, 5, 6, 7, 8), center_threshold=0.1,
        nms_kernel=5, top_k=100, center_loss_weight=100.0,
        offset_loss_weight=1.0, center_sigma=2.0,
    ):
        super().__init__()
        self.radial_bins = radial_bins
        self.angular_bins = angular_bins
        self.max_radius = max_radius
        self.thing_classes = tuple(thing_classes)
        self.center_threshold = center_threshold
        self.nms_kernel = nms_kernel
        self.top_k = top_k
        self.center_loss_weight = center_loss_weight
        self.offset_loss_weight = offset_loss_weight
        self.center_sigma = center_sigma

        self.feature_projection = nn.Sequential(
            nn.Linear(in_channels, 64, bias=False), nn.BatchNorm1d(64), nn.ReLU(inplace=True)
        )
        self.stem = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.down = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.up = nn.Sequential(nn.ConvTranspose2d(128, 64, 2, stride=2, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.center_head = nn.Conv2d(64, 1, 1)
        self.offset_head = nn.Conv2d(64, 2, 1)

    def polar_indices(self, points, batch_idx):
        radius = torch.norm(points[:, :2], dim=1)
        angle = torch.atan2(points[:, 1], points[:, 0])
        radial_idx = torch.floor(radius / self.max_radius * self.radial_bins).long().clamp_(0, self.radial_bins - 1)
        angular_idx = torch.floor((angle + math.pi) / (2 * math.pi) * self.angular_bins).long().remainder(self.angular_bins)
        return torch.stack([batch_idx.long(), radial_idx, angular_idx], dim=1)

    def build_bev_features(self, point_features, polar_indices, batch_size):
        projected = self.feature_projection(point_features)
        unique_indices, inverse = torch.unique(polar_indices, return_inverse=True, dim=0)
        voxel_features = torch_scatter.scatter_mean(projected, inverse, dim=0)
        bev = projected.new_zeros((batch_size, self.radial_bins, self.angular_bins, projected.shape[1]))
        bev[unique_indices[:, 0], unique_indices[:, 1], unique_indices[:, 2]] = voxel_features
        return bev.permute(0, 3, 1, 2).contiguous()

    def predict(self, bev_features):
        stem = self.stem(bev_features)
        upsampled = self.up(self.down(stem))
        if upsampled.shape[-2:] != stem.shape[-2:]:
            upsampled = F.interpolate(upsampled, size=stem.shape[-2:], mode='bilinear', align_corners=False)
        features = self.fuse(torch.cat([stem, upsampled], dim=1))
        return self.center_head(features), self.offset_head(features)

    def thing_mask(self, semantic_labels):
        class_ids = torch.as_tensor(self.thing_classes, device=semantic_labels.device)
        return (semantic_labels[:, None] == class_ids[None, :]).any(dim=1)

    def build_targets(self, polar_indices, semantic_labels, instance_labels, batch_size):
        device = polar_indices.device
        center_target = torch.zeros((batch_size, 1, self.radial_bins, self.angular_bins), device=device)
        offset_sum = torch.zeros((batch_size, self.radial_bins, self.angular_bins, 2), device=device)
        offset_count = torch.zeros((batch_size, self.radial_bins, self.angular_bins), device=device)
        thing_mask = self.thing_mask(semantic_labels)
        radial_grid = torch.arange(self.radial_bins, device=device, dtype=torch.float32)[:, None]
        angular_grid = torch.arange(self.angular_bins, device=device, dtype=torch.float32)[None, :]
        instance_keys = torch.unique(torch.stack([polar_indices[:, 0], instance_labels.long()], dim=1), dim=0)

        for batch_id, instance_id in instance_keys:
            instance_mask = (polar_indices[:, 0] == batch_id) & (instance_labels.long() == instance_id) & thing_mask
            if instance_mask.sum() == 0:
                continue
            instance_cells = polar_indices[instance_mask, 1:].float()
            center = instance_cells.mean(dim=0)
            angular_distance = torch.abs(angular_grid - center[1])
            angular_distance = torch.minimum(angular_distance, self.angular_bins - angular_distance)
            distance_sq = (radial_grid - center[0]).square() + angular_distance.square()
            gaussian = torch.exp(-distance_sq / (2 * self.center_sigma ** 2))
            center_target[batch_id, 0] = torch.maximum(center_target[batch_id, 0], gaussian)

            point_cells = polar_indices[instance_mask]
            offsets = center[None] - point_cells[:, 1:].float()
            offsets[:, 1] = torch.remainder(offsets[:, 1] + self.angular_bins / 2, self.angular_bins) - self.angular_bins / 2
            flat_cells = point_cells[:, 0] * (self.radial_bins * self.angular_bins) + point_cells[:, 1] * self.angular_bins + point_cells[:, 2]
            flat_offsets = offset_sum.reshape(-1, 2)
            flat_offsets.index_add_(0, flat_cells, offsets)
            offset_count.reshape(-1).index_add_(0, flat_cells, torch.ones_like(offsets[:, 0]))

        offset_mask = offset_count > 0
        offset_target = (offset_sum / offset_count.unsqueeze(-1).clamp_min(1.0)).permute(0, 3, 1, 2)
        return center_target, offset_target, offset_mask.unsqueeze(1)

    def instance_loss(self, center_logits, offsets, polar_indices, semantic_labels, instance_labels, batch_size):
        center_target, offset_target, offset_mask = self.build_targets(polar_indices, semantic_labels, instance_labels, batch_size)
        center_loss = F.mse_loss(torch.sigmoid(center_logits), center_target)
        if offset_mask.any():
            offset_loss = F.smooth_l1_loss(offsets[offset_mask.expand_as(offsets)], offset_target[offset_mask.expand_as(offset_target)])
        else:
            offset_loss = offsets.sum() * 0
        return {'loss_center': center_loss * self.center_loss_weight, 'loss_offset': offset_loss * self.offset_loss_weight}

    @torch.no_grad()
    def decode(self, center_logits, offsets, polar_indices, semantic_labels, batch_size):
        center_probs = torch.sigmoid(center_logits)
        padding = (self.nms_kernel - 1) // 2
        padded_probs = F.pad(center_probs, (padding, padding, 0, 0), mode="circular")
        pooled_probs = F.max_pool2d(padded_probs, self.nms_kernel, 1, (padding, 0))[:, :, :, padding:-padding]
        peaks = (center_probs == pooled_probs) & (center_probs >= self.center_threshold)
        instance_ids = torch.zeros((polar_indices.shape[0],), dtype=torch.long, device=polar_indices.device)
        thing_mask = self.thing_mask(semantic_labels)
        for batch_id in range(batch_size):
            peak_cells = torch.nonzero(peaks[batch_id, 0], as_tuple=False)
            if peak_cells.shape[0] == 0:
                continue
            peak_scores = center_probs[batch_id, 0, peak_cells[:, 0], peak_cells[:, 1]]
            if peak_cells.shape[0] > self.top_k:
                peak_cells = peak_cells[torch.topk(peak_scores, self.top_k).indices]
            point_mask = (polar_indices[:, 0] == batch_id) & thing_mask
            if point_mask.sum() == 0:
                continue
            point_cells = polar_indices[point_mask]
            point_offsets = offsets[batch_id, :, point_cells[:, 1], point_cells[:, 2]].transpose(0, 1)
            shifted_cells = point_cells[:, 1:].float() + point_offsets
            distances = shifted_cells[:, None] - peak_cells[None].float()
            distances[:, :, 1] = torch.remainder(distances[:, :, 1] + self.angular_bins / 2, self.angular_bins) - self.angular_bins / 2
            instance_ids[point_mask] = distances.square().sum(dim=-1).argmin(dim=1) + 1
        return instance_ids

    def forward(self, points, point_features, batch_idx, semantic_labels, instance_labels=None, compute_loss=False):
        batch_size = int(batch_idx.max().item()) + 1
        polar_indices = self.polar_indices(points, batch_idx)
        center_logits, offsets = self.predict(self.build_bev_features(point_features, polar_indices, batch_size))
        output = {'center_logits': center_logits, 'offsets': offsets}
        if compute_loss:
            output.update(self.instance_loss(center_logits, offsets, polar_indices, semantic_labels, instance_labels, batch_size))
        output['instance_ids'] = self.decode(center_logits, offsets, polar_indices, semantic_labels, batch_size)
        return output
