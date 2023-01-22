# This file contains classes for local and global offline mapping (not running semantic prediction)
import torch
import open3d
import torch.nn.functional as F
import numpy as np
import time
from Models.ConvBKI import ConvBKI

# TODO: Trilinear interpolation

# Save grid in CPU memory, load to GPU when needed for update step
# Voxels are stored in a matrix [X | Y | Z | C_0 | ... C_N] where C is semantic class

class LocalMap(ConvBKI):
    def __init__(self, grid_size, min_bound, max_bound, weights, filter_size, num_classes=21, ignore_labels = None, prior=0.001, device="cpu",
                 datatype=torch.float32, sparse=True, delete_time=10):
        super().__init__(grid_size, min_bound, max_bound, filter_size=filter_size,
                 num_classes=num_classes, prior=prior, device=device, datatype=datatype)

        self.initial_pose = None
        self.global_pose = None
        self.prior = prior
        self.ego_to_map = None
        self.translation = torch.zeros_like(self.voxel_sizes)
        self.ignore_labels = ignore_labels
        self.weights = weights
        self.reset_grid()

        self.ConvLayer = torch.nn.Conv3d(num_classes, num_classes, filter_size, padding="same", groups=num_classes,
                                         device=device, dtype=datatype, bias=False)
        self.ConvLayer.weight.requires_grad = False
        self.ConvLayer.weight[:, :, :, :, :] = weights.detach()[:, :, :, :, :]

        self.ConvLayer.eval()
        self.delete_time = delete_time

    def reset_grid(self):
        self.ego_to_map = None
        self.global_pose = None
        self.translation = torch.zeros_like(self.voxel_sizes)
        self.global_map = None
        self.map_times = None
        self.initial_pose = None
        self.translation_discretized = np.zeros(3)
        self.points_rotation = torch.eye(3, dtype=self.dtype, device=self.device)
        self.points_translation = torch.zeros(3, dtype=self.dtype, device=self.device)


    def gen_points_loc(self):
        LABEL_COLORS = np.array([
            (255, 255, 255),  # None
            (70, 70, 70),  # Building
            (100, 40, 40),  # Fences
            (55, 90, 80),  # Other
            (255, 255, 0),  # Pedestrian
            (153, 153, 153),  # Pole
            (157, 234, 50),  # RoadLines
            (0, 0, 255),  # Road
            (255, 255, 255),  # Sidewalk
            (0, 155, 0),  # Vegetation
            (255, 0, 0),  # Vehicle
            (102, 102, 156),  # Wall
            (220, 220, 0),  # TrafficSign
            (70, 130, 180),  # Sky
            (255, 255, 255),  # Ground
            (150, 100, 100),  # Bridge
            (230, 150, 140),  # RailTrack
            (180, 165, 180),  # GuardRail
            (250, 170, 30),  # TrafficLight
            (110, 190, 160),  # Static
            (170, 120, 50),  # Dynamic
            (45, 60, 150),  # Water
            (145, 170, 100),  # Terrain
        ]) / 255.0  # normalize each channel [0-1] since is what Open3D uses

        map = self.global_map
        map = torch.argmax(map, axis=3)
        pc = torch.where(map < 25)
        pc = torch.vstack((pc[0], pc[1], pc[2]))
        labels = map[pc[0], pc[1], pc[2]]
        mask = torch.where(labels != 0)[0]
        pc = pc[:, mask].detach().cpu().numpy()
        labels = labels[mask].detach().cpu().numpy()

        int_color = LABEL_COLORS[labels]
        point_list = open3d.geometry.PointCloud()
        point_list.points = open3d.utility.Vector3dVector(pc.T)
        point_list.colors = open3d.utility.Vector3dVector(int_color)
        return point_list


    def inside_mask(self, min_bounds, max_bounds):
        inside = np.all((self.global_map[:, :3] >= min_bounds) & (self.global_map[:, :3] < max_bounds), axis=1)
        return inside

    def get_local_map(self, min_bound=None, max_bound=None):
        # Fetch local map from CPU (anything not seen is prior)
        local_map = self.initialize_grid()
        inside_mask = None
        if min_bound is None:
            min_bound = self.min_bound
        if max_bound is None:
            max_bound = self.max_bound
        local_min_bound = min_bound + torch.from_numpy(self.voxel_translation).to(self.device)
        local_max_bound = max_bound + torch.from_numpy(self.voxel_translation).to(self.device)
        if self.global_map is not None:
            inside_mask = self.inside_mask(local_min_bound.detach().cpu().numpy(), local_max_bound.detach().cpu().numpy())
            allocated_map = torch.tensor(self.global_map[inside_mask], device=self.device, dtype=self.dtype)
            grid_map = self.grid_ind(allocated_map, min_bound=local_min_bound, max_bound=local_max_bound)
            grid_indices = grid_map[:, :3].to(torch.long)
            local_map[grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2], :] = allocated_map[:, 3:]
        return local_map, local_min_bound, local_max_bound, inside_mask

    def axis_limits(self, offset, axis_size):
        min_bound = 0
        max_bound = axis_size
        to_min = min_bound - offset
        to_max = max_bound - offset
        from_min = min_bound + offset
        from_max = max_bound + offset
        bounds = torch.clamp(torch.stack([to_min, to_max, from_min, from_max]), min=min_bound, max=max_bound).long()
        return bounds

    def grid_limits(self, grid, voxel_translation):
        x_bounds = self.axis_limits(voxel_translation[0], grid.shape[0])
        y_bounds = self.axis_limits(voxel_translation[1], grid.shape[1])
        z_bounds = self.axis_limits(voxel_translation[2], grid.shape[2])
        return [x_bounds, y_bounds, z_bounds]
    # Uses saved weights instead of generating a filter
    def propup(self, new_pose, semantic_preds):
        current_map = self.global_map
        prev_pose = self.global_pose
        self.global_pose = new_pose
        if self.initial_pose is None:
            current_map = torch.zeros(self.grid_size[0], self.grid_size[1], self.grid_size[2],
                           self.num_classes, device=self.device, requires_grad=True,
                           dtype=self.dtype) + self.prior
            prev_pose = new_pose
            self.initial_pose = new_pose
            # self.initial_pose = new_pose
            # return torch.eye(4).to(new_pose.device), current_map
        # Find the closest translation in voxels
        prev_to_initial = torch.matmul(torch.linalg.inv(self.initial_pose), prev_pose)
        prev_translation = prev_to_initial[:3, 3]
        prev_voxel = torch.round(prev_translation / self.voxel_sizes)

        new_to_initial = torch.matmul(torch.linalg.inv(self.initial_pose), new_pose)
        new_translation = new_to_initial[:3, 3]
        R = new_to_initial[:3, :3]
        new_voxel = torch.round(new_translation / self.voxel_sizes)

        self.translation = new_voxel * self.voxel_sizes
        voxel_translation = new_voxel - prev_voxel
        # Transform Map
        new_map = torch.zeros_like(current_map) + self.prior
        b = self.grid_limits(current_map, voxel_translation)
        new_map[b[0][0]:b[0][1], b[1][0]:b[1][1], b[2][0]:b[2][1], :] = \
            current_map[b[0][2]:b[0][3], b[1][2]:b[1][3], b[2][2]:b[2][3], :]
        # Transform from ego to map frame
        self.ego_to_map = torch.zeros_like(new_pose)
        self.ego_to_map[:3, 3] = new_translation - self.translation
        self.ego_to_map[:3, :3] = R
        self.ego_to_map[3, 3] = 1
        self.global_map = new_map
        # return self.ego_to_map, new_map

        semantic_preds = semantic_preds.to(self.dtype)
        # local_map, local_min_bound, local_max_bound, inside_mask = self.get_local_map() # combines sensor data, and previous map in local frame, .08

        # Rotate the point cloud and translate to global frame
        # global_pose = torch.from_numpy(self.global_pose).to(self.device)
        semantic_preds[:, :3] = torch.matmul(self.ego_to_map[:3, :3], semantic_preds[:, :3].T).T + self.ego_to_map[:3, 3]

        # Change to indices using our global frame bounds
        grid_pc = self.grid_ind(semantic_preds)
        # Update local map
        update = torch.zeros_like(self.global_map, requires_grad=False)

        continuous = False
        N, C = semantic_preds.shape
        if C == self.num_classes + 3:
            continuous = True
        update = self.add_to_update(update, grid_pc, continuous)

        # Apply BKI filters .125s
        update = torch.unsqueeze(update.permute(3, 0, 1, 2), 0)
        update = self.ConvLayer(update)
        new_update = torch.squeeze(update).permute(1, 2, 3, 0)

        # Find updated cells
        self.global_map = self.global_map + new_update
        return self.global_map


    # Propagate map given a transformation matrix
    def propagate(self, pose):
        self.global_pose = pose
        # Was just initialized
        if self.initial_pose is None:
            self.initial_pose = pose
        # Relative transformation between origin and current point
        relative_translation = pose[:3, 3] - self.initial_pose[:3, 3]
        # To select voxels from memory, find the nearest voxel
        voxel_sizes = self.voxel_sizes.detach().cpu().numpy()
        self.voxel_translation = np.round(relative_translation / voxel_sizes) * voxel_sizes #the translation in number of voxels
        self.nearest_voxel = self.initial_pose[:3, 3] + self.voxel_translation


    # Predict labels for points after propagating pose
    def label_points(self, points):
        points = torch.from_numpy(points).to(self.device)
        global_pose = torch.from_numpy(self.global_pose).to(self.device)
        points = torch.matmul(global_pose[:3, :3], points.T).T + global_pose[:3, 3]
        labels = torch.zeros((points.shape[0], self.num_classes), dtype=torch.float32, device=self.device)

        local_map, local_min_bound, local_max_bound, __ = self.get_local_map()

        local_mask = torch.all((points < local_max_bound) & (points >= local_min_bound), dim=1)

        local_points = points[local_mask]

        grid_inds = torch.floor((local_points - local_min_bound) / self.voxel_sizes)
        maxes = (self.grid_size - 1).view(1, 3)
        clipped_inds = torch.clamp(grid_inds, torch.zeros_like(maxes), maxes).to(torch.long)

        labels[local_mask, :] = local_map[clipped_inds[:, 0], clipped_inds[:, 1], clipped_inds[:, 2], :]
        labels[~local_mask, :] = self.prior

        # TODO: Add some sort of thresholding based on variance
        # TODO: Add calculation of expectation, variance
        predictions = torch.argmax(labels, dim=1)
        predictions[~local_mask] = self.ignore_labels[0]

        return predictions, local_mask

class GlobalMap(ConvBKI):
    def __init__(self, grid_size, min_bound, max_bound, weights, filter_size, num_classes=21, ignore_labels = None, prior=0.001, device="cpu",
                 datatype=torch.float32, sparse=True, delete_time=10):
        super().__init__(grid_size, min_bound, max_bound, filter_size=filter_size,
                 num_classes=num_classes, prior=prior, device=device, datatype=datatype)
        self.ignore_labels = ignore_labels
        self.weights = weights
        self.reset_grid()

        self.ConvLayer = torch.nn.Conv3d(num_classes, num_classes, filter_size, padding="same", groups=num_classes,
                                         device=device, dtype=datatype, bias=False)
        self.ConvLayer.weight.requires_grad = False
        self.ConvLayer.weight[:, :, :, :, :] = weights.detach()[:, :, :, :, :]

        self.ConvLayer.eval()
        self.delete_time = delete_time

    def reset_grid(self):
        self.global_map = None
        self.map_times = None
        self.initial_pose = None
        self.translation_discretized = np.zeros(3)
        self.points_rotation = torch.eye(3, dtype=self.dtype, device=self.device)
        self.points_translation = torch.zeros(3, dtype=self.dtype, device=self.device)
    def gen_points_glo(self, num):

        LABEL_COLORS = np.array([
            (255, 255, 255),  # None
            (70, 70, 70),  # Building
            (100, 40, 40),  # Fences
            (55, 90, 80),  # Other
            (255, 255, 0),  # Pedestrian
            (153, 153, 153),  # Pole
            (157, 234, 50),  # RoadLines
            (0, 0, 255),  # Road
            (255, 255, 255),  # Sidewalk
            (0, 155, 0),  # Vegetation
            (255, 0, 0),  # Vehicle
            (102, 102, 156),  # Wall
            (220, 220, 0),  # TrafficSign
            (70, 130, 180),  # Sky
            (255, 255, 255),  # Ground
            (150, 100, 100),  # Bridge
            (230, 150, 140),  # RailTrack
            (180, 165, 180),  # GuardRail
            (250, 170, 30),  # TrafficLight
            (110, 190, 160),  # Static
            (170, 120, 50),  # Dynamic
            (45, 60, 150),  # Water
            (145, 170, 100),  # Terrain
        ]) / 255.0  # normalize each channel [0-1] since is what Open3D uses

        map = self.global_map
        grid_indices = map[:, :3]
        allocated_map = map[:, 3:]
        # if num==0:
        #     grid_indices[:, 1] += 100
        labels = np.argmax(allocated_map, axis=1)

        int_color = LABEL_COLORS[labels]
        point_list = open3d.geometry.PointCloud()
        point_list.points = open3d.utility.Vector3dVector(grid_indices)
        point_list.colors = open3d.utility.Vector3dVector(int_color)
        return point_list
    def inside_mask(self, min_bounds, max_bounds):
        inside = np.all((self.global_map[:, :3] >= min_bounds) & (self.global_map[:, :3] < max_bounds), axis=1)
        return inside

    def get_local_map(self, min_bound=None, max_bound=None):
        # Fetch local map from CPU (anything not seen is prior)
        local_map = self.initialize_grid()
        inside_mask = None
        if min_bound is None:
            min_bound = self.min_bound
        if max_bound is None:
            max_bound = self.max_bound
        local_min_bound = min_bound + torch.from_numpy(self.voxel_translation).to(self.device)
        local_max_bound = max_bound + torch.from_numpy(self.voxel_translation).to(self.device)
        if self.global_map is not None:
            inside_mask = self.inside_mask(local_min_bound.detach().cpu().numpy(), local_max_bound.detach().cpu().numpy())
            allocated_map = torch.tensor(self.global_map[inside_mask], device=self.device, dtype=self.dtype)
            grid_map = self.grid_ind(allocated_map, min_bound=local_min_bound, max_bound=local_max_bound)
            grid_indices = grid_map[:, :3].to(torch.long)
            local_map[grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2], :] = allocated_map[:, 3:]
        return local_map, local_min_bound, local_max_bound, inside_mask

    # Uses saved weights instead of generating a filter
    def update_map(self, semantic_preds):

        semantic_preds = semantic_preds.to(self.dtype)
        local_map, local_min_bound, local_max_bound, inside_mask = self.get_local_map() # combines sensor data, and previous map in local frame, .08

        # Rotate the point cloud and translate to global frame
        global_pose = torch.from_numpy(self.global_pose).to(self.device)
        semantic_preds[:, :3] = torch.matmul(global_pose[:3, :3], semantic_preds[:, :3].T).T + global_pose[:3, 3]

        # Change to indices using our global frame bounds
        grid_pc = self.grid_ind(semantic_preds, min_bound=local_min_bound, max_bound=local_max_bound)
        # Update local map
        update = torch.zeros_like(local_map, requires_grad=False)

        continuous = False
        N, C = semantic_preds.shape
        if C == self.num_classes + 3:
            continuous = True
        update = self.add_to_update(update, grid_pc, continuous)

        # Apply BKI filters .125s
        update = torch.unsqueeze(update.permute(3, 0, 1, 2), 0)
        update = self.ConvLayer(update)
        new_update = torch.squeeze(update).permute(1, 2, 3, 0)

        # Find updated cells
        local_map = local_map + new_update
        updated_cells = (torch.mean(local_map, dim=3) > self.prior).view(-1)

        updated_centroids = self.centroids[updated_cells, :] + torch.from_numpy(self.voxel_translation).to(self.device)
        local_values = local_map.view(-1, self.num_classes)[updated_cells]
        new_cells = torch.cat((updated_centroids, local_values), dim=1)

        # Visited Times = 0
        visited_times = torch.zeros(new_cells.shape[0], 1).detach().cpu().numpy()
        # If empty

        if self.global_map is None:
            self.global_map = new_cells.detach().cpu().numpy()
            self.map_times = visited_times
        else:
            # Replace local cells .065
            outside_mask = ~ inside_mask
            # Add new cells
            self.global_map = np.vstack((self.global_map[outside_mask, :], new_cells.detach().cpu().numpy()))
            self.map_times = np.vstack((self.map_times[outside_mask, :], visited_times))

        # Garbage Collection .026
        self.garbage_collection()


        return self.global_map

    def garbage_collection(self):
        self.map_times += 1
        # Remove cells with T > self.delete_time
        recent_mask = self.map_times < self.delete_time
        recent_mask = np.squeeze(recent_mask)
        self.map_times = self.map_times[recent_mask, :]
        self.global_map = self.global_map[recent_mask, :]

    # Propagate map given a transformation matrix
    def propagate(self, pose):
        self.global_pose = pose
        # Was just initialized
        if self.initial_pose is None:
            self.initial_pose = pose
        # Relative transformation between origin and current point
        relative_translation = pose[:3, 3] - self.initial_pose[:3, 3]
        # To select voxels from memory, find the nearest voxel
        voxel_sizes = self.voxel_sizes.detach().cpu().numpy()
        self.voxel_translation = np.round(relative_translation / voxel_sizes) * voxel_sizes #the translation in number of voxels
        self.nearest_voxel = self.initial_pose[:3, 3] + self.voxel_translation

    # Predict labels for points after propagating pose
    def label_points(self, points):
        points = torch.from_numpy(points).to(self.device)
        global_pose = torch.from_numpy(self.global_pose).to(self.device)
        points = torch.matmul(global_pose[:3, :3], points.T).T + global_pose[:3, 3]
        labels = torch.zeros((points.shape[0], self.num_classes), dtype=torch.float32, device=self.device)

        local_map, local_min_bound, local_max_bound, __ = self.get_local_map()

        local_mask = torch.all((points < local_max_bound) & (points >= local_min_bound), dim=1)

        local_points = points[local_mask]

        grid_inds = torch.floor((local_points - local_min_bound) / self.voxel_sizes)
        maxes = (self.grid_size - 1).view(1, 3)
        clipped_inds = torch.clamp(grid_inds, torch.zeros_like(maxes), maxes).to(torch.long)

        labels[local_mask, :] = local_map[clipped_inds[:, 0], clipped_inds[:, 1], clipped_inds[:, 2], :]
        labels[~local_mask, :] = self.prior

        # TODO: Add some sort of thresholding based on variance
        # TODO: Add calculation of expectation, variance
        predictions = torch.argmax(labels, dim=1)
        predictions[~local_mask] = self.ignore_labels[0]

        return predictions, local_mask



