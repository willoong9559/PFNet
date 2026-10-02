"""Polar-BEV instance head with Panoptic-PHNet pseudo-heatmap decoding."""

import math

import numpy as np
from scipy.spatial import cKDTree

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter


class PolarInstanceHead(nn.Module):
    """Regress offsets; derive instance centers from shifted-voxel counts."""

    def __init__(
        self, in_channels, radial_bins=480, angular_bins=360, max_radius=70.0,
        thing_classes=(1, 2, 3, 4, 5, 6, 7, 8),
        pseudo_grid_size=0.2, bev_x_range=(-48.0, 48.0), bev_y_range=(-48.0, 48.0),
        pseudo_nms_kernel=5, pseudo_class_kernel=5,
        center_group_radii=(2.0, 1.0, 1.0, 4.0, 4.0, 1.0, 1.0, 1.0),
        offset_loss_weight=1.0,
    ):
        super().__init__()
        self.radial_bins = radial_bins
        self.angular_bins = angular_bins
        self.max_radius = max_radius
        self.thing_classes = tuple(thing_classes)
        self.pseudo_grid_size = pseudo_grid_size
        self.bev_x_range = tuple(bev_x_range)
        self.bev_y_range = tuple(bev_y_range)
        self.bev_height = int(round((bev_x_range[1] - bev_x_range[0]) / pseudo_grid_size))
        self.bev_width = int(round((bev_y_range[1] - bev_y_range[0]) / pseudo_grid_size))
        self.pseudo_nms_kernel = pseudo_nms_kernel
        self.pseudo_class_kernel = pseudo_class_kernel
        self.center_group_radii = tuple(center_group_radii)
        self.offset_loss_weight = offset_loss_weight

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
        self.offset_head = nn.Conv2d(64, 2, 1)

    def polar_indices(self, points, batch_idx):
        radius = torch.norm(points[:, :2], dim=1)
        angle = torch.atan2(points[:, 1], points[:, 0])
        radial = torch.floor(radius / self.max_radius * self.radial_bins).long().clamp_(0, self.radial_bins - 1)
        angular = torch.floor((angle + math.pi) / (2 * math.pi) * self.angular_bins).long().remainder(self.angular_bins)
        return torch.stack([batch_idx.long(), radial, angular], dim=1)

    def build_bev_features(self, point_features, polar_indices, batch_size):
        projected = self.feature_projection(point_features)
        unique_indices, inverse = torch.unique(polar_indices, return_inverse=True, dim=0)
        voxel_features = torch_scatter.scatter_mean(projected, inverse, dim=0)
        bev = projected.new_zeros((batch_size, self.radial_bins, self.angular_bins, projected.shape[1]))
        bev[unique_indices[:, 0], unique_indices[:, 1], unique_indices[:, 2]] = voxel_features
        return bev.permute(0, 3, 1, 2).contiguous()

    def predict_offsets(self, bev_features):
        stem = self.stem(bev_features)
        upsampled = self.up(self.down(stem))
        if upsampled.shape[-2:] != stem.shape[-2:]:
            upsampled = F.interpolate(upsampled, size=stem.shape[-2:], mode='bilinear', align_corners=False)
        return self.offset_head(self.fuse(torch.cat([stem, upsampled], dim=1)))

    def thing_mask(self, semantic_labels):
        class_ids = torch.as_tensor(self.thing_classes, device=semantic_labels.device)
        return (semantic_labels[:, None] == class_ids[None, :]).any(dim=1)

    def build_offset_targets(self, polar_indices, semantic_labels, instance_labels, batch_size):
        device = polar_indices.device
        offset_sum = torch.zeros((batch_size, self.radial_bins, self.angular_bins, 2), device=device)
        offset_count = torch.zeros((batch_size, self.radial_bins, self.angular_bins), device=device)
        thing_mask = self.thing_mask(semantic_labels)
        instance_keys = torch.unique(torch.stack([polar_indices[:, 0], instance_labels.long()], dim=1), dim=0)

        for batch_id, instance_id in instance_keys:
            mask = (polar_indices[:, 0] == batch_id) & (instance_labels.long() == instance_id) & thing_mask
            if mask.sum() == 0:
                continue
            cells = polar_indices[mask]
            center = cells[:, 1:].float().mean(dim=0)
            offset = center[None] - cells[:, 1:].float()
            offset[:, 1] = torch.remainder(offset[:, 1] + self.angular_bins / 2, self.angular_bins) - self.angular_bins / 2
            flat_cells = cells[:, 0] * (self.radial_bins * self.angular_bins) + cells[:, 1] * self.angular_bins + cells[:, 2]
            offset_sum.reshape(-1, 2).index_add_(0, flat_cells, offset)
            offset_count.reshape(-1).index_add_(0, flat_cells, torch.ones_like(offset[:, 0]))

        mask = offset_count > 0
        target = (offset_sum / offset_count.unsqueeze(-1).clamp_min(1.0)).permute(0, 3, 1, 2)
        return target, mask.unsqueeze(1)

    def offset_loss(self, offsets, polar_indices, semantic_labels, instance_labels, batch_size):
        target, mask = self.build_offset_targets(polar_indices, semantic_labels, instance_labels, batch_size)
        if not mask.any():
            return offsets.sum() * 0
        return F.smooth_l1_loss(offsets[mask.expand_as(offsets)], target[mask.expand_as(target)]) * self.offset_loss_weight

    def shifted_xy(self, polar_indices, offsets):
        point_offsets = offsets[polar_indices[:, 0], :, polar_indices[:, 1], polar_indices[:, 2]]
        shifted_radius = (polar_indices[:, 1].float() + point_offsets[:, 0] + 0.5) / self.radial_bins * self.max_radius
        shifted_angle = (polar_indices[:, 2].float() + point_offsets[:, 1] + 0.5) / self.angular_bins * (2 * math.pi) - math.pi
        return torch.stack([shifted_radius * torch.cos(shifted_angle), shifted_radius * torch.sin(shifted_angle)], dim=1)

    def pseudo_class_image(self, polar_indices, offsets, semantic_labels, batch_size):
        """Count shifted thing voxels per semantic class in Cartesian BEV."""
        unique_indices, inverse = torch.unique(polar_indices, return_inverse=True, dim=0)
        semantic_votes = torch_scatter.scatter_add(
            F.one_hot((semantic_labels.long() - 1).clamp_(0, 18), num_classes=19).float(),
            inverse,
            dim=0,
            dim_size=unique_indices.shape[0],
        )
        voxel_labels = semantic_votes.argmax(dim=1) + 1
        shifted_xy = self.shifted_xy(unique_indices, offsets)
        thing_mask = self.thing_mask(voxel_labels)
        x_cell = torch.floor((shifted_xy[:, 0] - self.bev_x_range[0]) / self.pseudo_grid_size).long()
        y_cell = torch.floor((shifted_xy[:, 1] - self.bev_y_range[0]) / self.pseudo_grid_size).long()
        in_bounds = (x_cell >= 0) & (x_cell < self.bev_height) & (y_cell >= 0) & (y_cell < self.bev_width)
        valid = thing_mask & in_bounds
        classes = voxel_labels[valid].long() - 1
        cells = unique_indices[valid, 0] * (19 * self.bev_height * self.bev_width) + classes * (self.bev_height * self.bev_width) + x_cell[valid] * self.bev_width + y_cell[valid]
        counts = torch.bincount(cells, minlength=batch_size * 19 * self.bev_height * self.bev_width)
        return counts.reshape(batch_size, 19, self.bev_height, self.bev_width).float()

    def pseudo_centers(self, class_image, batch_id):
        heatmap = class_image.sum(dim=0, keepdim=True)
        padding = (self.pseudo_nms_kernel - 1) // 2
        local_max = F.max_pool2d(heatmap, self.pseudo_nms_kernel, 1, padding)
        center_cells = torch.nonzero((heatmap == local_max) & (heatmap > 0), as_tuple=False)
        if center_cells.shape[0] == 0:
            return center_cells.new_zeros((0, 2)), center_cells.new_zeros((0,), dtype=torch.long)

        class_padding = (self.pseudo_class_kernel - 1) // 2
        class_votes = F.avg_pool2d(class_image, self.pseudo_class_kernel, 1, class_padding)
        center_xy = center_cells[:, 1:]
        labels = class_votes[:, center_xy[:, 0], center_xy[:, 1]].argmax(dim=0) + 1
        return center_xy, labels

    def group_center_ids(self, center_cells, center_labels):
        """Merge same-class centers using a spatial index rather than all pairs."""
        parents = list(range(center_cells.shape[0]))

        def find(index):
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        centers_np = center_cells.detach().cpu().numpy()
        labels_np = center_labels.detach().cpu().numpy()
        for class_index, class_id in enumerate(self.thing_classes):
            member_indices = np.flatnonzero(labels_np == class_id)
            if member_indices.size < 2:
                continue
            radius = self.center_group_radii[class_index] / self.pseudo_grid_size
            tree = cKDTree(centers_np[member_indices])
            for first, second in tree.query_pairs(radius):
                first_index = int(member_indices[first])
                second_index = int(member_indices[second])
                parents[find(second_index)] = find(first_index)

        roots = [find(index) for index in range(center_cells.shape[0])]
        root_to_instance = {}
        grouped_ids = []
        for root in roots:
            if root not in root_to_instance:
                root_to_instance[root] = len(root_to_instance) + 1
            grouped_ids.append(root_to_instance[root])
        return torch.as_tensor(grouped_ids, dtype=torch.long, device=center_cells.device)

    @torch.no_grad()
    def decode(self, polar_indices, offsets, semantic_labels, batch_size):
        shifted_xy = self.shifted_xy(polar_indices, offsets)
        class_images = self.pseudo_class_image(polar_indices, offsets, semantic_labels, batch_size)
        instance_ids = torch.zeros((polar_indices.shape[0],), dtype=torch.long, device=polar_indices.device)
        thing_mask = self.thing_mask(semantic_labels)
        x_cell = torch.floor((shifted_xy[:, 0] - self.bev_x_range[0]) / self.pseudo_grid_size).long()
        y_cell = torch.floor((shifted_xy[:, 1] - self.bev_y_range[0]) / self.pseudo_grid_size).long()
        valid_points = thing_mask & (x_cell >= 0) & (x_cell < self.bev_height) & (y_cell >= 0) & (y_cell < self.bev_width)

        for batch_id in range(batch_size):
            centers, center_labels = self.pseudo_centers(class_images[batch_id], batch_id)
            if centers.shape[0] == 0:
                continue
            grouped_ids = self.group_center_ids(centers, center_labels)
            point_mask = valid_points & (polar_indices[:, 0] == batch_id)
            if point_mask.sum() == 0:
                continue
            points = shifted_xy[point_mask]
            centers_xy = torch.stack([
                self.bev_x_range[0] + (centers[:, 0].float() + 0.5) * self.pseudo_grid_size,
                self.bev_y_range[0] + (centers[:, 1].float() + 0.5) * self.pseudo_grid_size,
            ], dim=1)
            nearest = torch.cat([
                torch.cdist(points[start:start + 2048], centers_xy).argmin(dim=1)
                for start in range(0, points.shape[0], 2048)
            ])
            instance_ids[point_mask] = grouped_ids[nearest]
        return instance_ids, class_images

    def forward(self, points, point_features, batch_idx, semantic_labels, instance_labels=None, compute_loss=False):
        batch_size = int(batch_idx.max().item()) + 1
        polar_indices = self.polar_indices(points, batch_idx)
        offsets = self.predict_offsets(self.build_bev_features(point_features, polar_indices, batch_size))
        output = {'offsets': offsets}
        if compute_loss:
            output['loss_offset'] = self.offset_loss(offsets, polar_indices, semantic_labels, instance_labels, batch_size)
        else:
            instance_ids, pseudo_heatmap = self.decode(polar_indices, offsets, semantic_labels, batch_size)
            output['instance_ids'] = instance_ids
            output['pseudo_heatmap'] = pseudo_heatmap
        return output
