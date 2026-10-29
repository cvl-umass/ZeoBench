# In solo/data/custom_transforms.py

import torch
import torch.nn.functional as F
import numpy as np

def coordinates_3d(start=0, H=100, W=100, D=100):
    xs = torch.linspace(start, end=H-1, steps=H)
    ys = torch.linspace(start, end=W-1, steps=W)
    zs = torch.linspace(start, end=D-1, steps=D)
    xc = xs.repeat(W*D, 1).transpose(0, 1).contiguous().view(-1)
    yc = ys.repeat(D, 1).transpose(0, 1).contiguous().view(-1) \
        .repeat(H, 1).contiguous().view(-1)
    zc = zs.repeat(H*W)
    new_coord = torch.cat((xc.unsqueeze(1), yc.unsqueeze(1), zc.unsqueeze(1)), 1)
    return new_coord

class ZeoAugmentation:
    def __init__(self, crop_size=100, grid_resolution=0.45, rotate=True, translate=True,
                 grid_res_sigma=0.0, vector_reference_resolution=0.15):
        if isinstance(crop_size, int):
            self.H = self.W = self.D = crop_size
        elif isinstance(crop_size, (tuple, list)) and len(crop_size) == 3:
            self.H, self.W, self.D = crop_size
        else:
            raise ValueError("crop_size must be an int or a sequence of 3 ints (H, W, D)")
            
        self.crop_size = crop_size
        self.base_grid_resolution = grid_resolution
        self.rotate = rotate
        self.translate = translate
        self.grid_res_sigma = grid_res_sigma
        self.vector_reference_resolution = vector_reference_resolution
        
        self.coord = coordinates_3d(0, self.H, self.W, self.D)
        
        # Create the distribution for sampling resolution noise if sigma is provided
        if self.grid_res_sigma > 0:
            self.dist = torch.distributions.Normal(0, self.grid_res_sigma)

    def _get_Ainv_matrix(self, xvec, yvec, zvec, grid_res_x, grid_res_y, grid_res_z):
        # This logic is now part of the transform
        x_unit = xvec / (grid_res_x / self.vector_reference_resolution * np.linalg.norm(xvec))
        y_unit = yvec / (grid_res_y / self.vector_reference_resolution * np.linalg.norm(yvec))
        z_unit = zvec / (grid_res_z / self.vector_reference_resolution * np.linalg.norm(zvec))
        A = np.vstack([x_unit, y_unit, z_unit])
        A = torch.Tensor(A)
        return torch.inverse(A)

    def _augment(self, coord, bounds):
        if self.translate:
            coord += torch.rand(3) * bounds # random translation

        if self.rotate:
            p = torch.randn(3, 3) # random rotation
            rotation_matrix, _ = torch.linalg.qr(p)
            rand_int = torch.randint(0, 2, (1,)) # 0 or 1
            rotation_matrix = (2*rand_int-1) * rotation_matrix # -1 or 1
            rotation_matrix = torch.transpose(rotation_matrix,0,1) 
            new_coord = torch.mm(coord, rotation_matrix) #torch.Size([1000000, 3])
        else:
            new_coord = coord

        return new_coord
    
    def coord_to_grid(self, new_coord, A_inv, nx, ny, nz, bounds, grid):
        grid3d = torch.remainder(new_coord @ A_inv, torch.Tensor([nx, ny, nz])) # coordinates in voxel coordinate system
        
        scalar = torch.eye(3) # [-1,1]
        scalar[0, 0] = 2/nx
        scalar[1, 1] = 2/ny
        scalar[2, 2] = 2/nz
        
        scalar = torch.diag(2 / bounds) # [-1,1]
        grid3d = torch.add(torch.mm(grid3d, scalar), -1)

        transf_grid = grid3d.reshape(self.H, self.W, self.D, -1)
        grid= grid.permute(0, 3, 2, 1)
        grid = grid[None, :, :, :, :]
        grid_res = F.grid_sample(grid, transf_grid[None, :, :, :, :], align_corners=True) #[1, 1, H, W, D] 
        grid_res = torch.squeeze(grid_res,1) # [1, H, W, D]
        return grid_res

    def __call__(self, data_item):
        grid, xvec, yvec, zvec, bounds, nx, ny, nz = data_item

        if self.grid_res_sigma > 0:
            noise = self.dist.sample((3,))
            grid_res_x = max(0.05, self.base_grid_resolution + noise[0].item())
            grid_res_y = max(0.05, self.base_grid_resolution + noise[1].item())
            grid_res_z = max(0.05, self.base_grid_resolution + noise[2].item())
        else:
            grid_res_x = grid_res_y = grid_res_z = self.base_grid_resolution

        A_inv = self._get_Ainv_matrix(xvec, yvec, zvec, grid_res_x, grid_res_y, grid_res_z)

        new_coord = self._augment(self.coord.clone(), bounds)
        grid_res = self.coord_to_grid(new_coord, A_inv, nx, ny, nz, bounds, grid)
        
        return grid_res


class ZeoValidationTransform:
    def __init__(self, crop_size=100, grid_resolution=0.45,
                 vector_reference_resolution=0.15):
        if isinstance(crop_size, int):
            self.H = self.W = self.D = crop_size
        elif isinstance(crop_size, (tuple, list)) and len(crop_size) == 3:
            self.H, self.W, self.D = crop_size
        else:
            raise ValueError("crop_size must be an int or a sequence of 3 ints (H, W, D)")
            
        self.crop_size = crop_size
        self.grid_resolution = grid_resolution
        self.vector_reference_resolution = vector_reference_resolution
        
        self.coord = coordinates_3d(0, self.H, self.W, self.D)

    def _get_Ainv_matrix(self, xvec, yvec, zvec, grid_res):
        x_unit = xvec / (grid_res / self.vector_reference_resolution * np.linalg.norm(xvec))
        y_unit = yvec / (grid_res / self.vector_reference_resolution * np.linalg.norm(yvec))
        z_unit = zvec / (grid_res / self.vector_reference_resolution * np.linalg.norm(zvec))
        A = np.vstack([x_unit, y_unit, z_unit])
        A = torch.Tensor(A)
        return torch.inverse(A)
    
    def coord_to_grid(self, new_coord, A_inv, nx, ny, nz, bounds, grid):
        grid3d = torch.remainder(new_coord @ A_inv, torch.Tensor([nx, ny, nz])) # coordinates in voxel coordinate system
        
        scalar = torch.eye(3) # [-1,1]
        scalar[0, 0] = 2/nx
        scalar[1, 1] = 2/ny
        scalar[2, 2] = 2/nz
        
        scalar = torch.diag(2 / bounds) # [-1,1]
        grid3d = torch.add(torch.mm(grid3d, scalar), -1)

        transf_grid = grid3d.reshape(self.H, self.W, self.D, -1)
        grid= grid.permute(0, 3, 2, 1)
        grid = grid[None, :, :, :, :]
        grid_res = F.grid_sample(grid, transf_grid[None, :, :, :, :], align_corners=True) #[1, 1, 100,100,100] # padding_mode="reflection"?
        grid_res = torch.squeeze(grid_res,1) # [1, 100,100,100]
        return grid_res

    def __call__(self, data_item):
        grid, xvec, yvec, zvec, bounds, nx, ny, nz = data_item

        A_inv = self._get_Ainv_matrix(xvec, yvec, zvec, self.grid_resolution)
        new_coord = self.coord.clone()
        grid_res = self.coord_to_grid(new_coord, A_inv, nx, ny, nz, bounds, grid)
        
        return grid_res
    
class DirectSliceTransform:
    def __init__(self, image_size=224, grid_resolution=0.15, method='xyz', d=15, rotate=True, translate=True, **kwargs):
        self.image_size = image_size
        self.grid_resolution = grid_resolution
        self.method = method
        self.d = d
        self.rotate = rotate
        self.translate = translate
        
        xs = torch.linspace(0, end=image_size-1, steps=image_size)
        ys = torch.linspace(0, end=image_size-1, steps=image_size)
        self.grid_2d_x, self.grid_2d_y = torch.meshgrid(xs, ys, indexing='ij')

    def _get_Ainv_matrix(self, xvec, yvec, zvec):
        x_unit = xvec / (self.grid_resolution / 0.15 * np.linalg.norm(xvec))
        y_unit = yvec / (self.grid_resolution / 0.15 * np.linalg.norm(yvec))
        z_unit = zvec / (self.grid_resolution / 0.15 * np.linalg.norm(zvec))
        return torch.inverse(torch.Tensor(np.vstack([x_unit, y_unit, z_unit])))

    def _augment(self, flat_coords, bounds):
        if self.translate:
            flat_coords += torch.rand(3) * bounds

        if self.rotate:
            p = torch.randn(3, 3) 
            rotation_matrix, _ = torch.linalg.qr(p)
            rand_int = torch.randint(0, 2, (1,)) 
            rotation_matrix = (2*rand_int-1) * rotation_matrix 
            rotation_matrix = torch.transpose(rotation_matrix, 0, 1) 
            flat_coords = torch.mm(flat_coords, rotation_matrix) 
        return flat_coords

    def _get_slice_coords(self, nx, ny, nz):
        if self.method == 'xyz':
            idx_x = torch.randint(0, nx, (1,)).item()
            idx_y = torch.randint(0, ny, (1,)).item()
            idx_z = torch.randint(0, nz, (1,)).item()
            s1 = torch.stack([torch.full_like(self.grid_2d_x, idx_x), self.grid_2d_x, self.grid_2d_y], dim=-1)
            s2 = torch.stack([self.grid_2d_x, torch.full_like(self.grid_2d_x, idx_y), self.grid_2d_y], dim=-1)
            s3 = torch.stack([self.grid_2d_x, self.grid_2d_y, torch.full_like(self.grid_2d_x, idx_z)], dim=-1)
            return torch.stack([s1, s2, s3], dim=0)
        else:
            axis = torch.randint(0, 3, (1,)).item()
            max_idx = nx if axis == 0 else (ny if axis == 1 else nz)
            
            if self.method == 'random_perpendicular':
                indices = torch.randint(0, max_idx, (3,)).tolist()
            elif self.method == 'equidistant':
                low_bound = min(self.d, max_idx)
                high_bound = max(low_bound + 1, max_idx - self.d)
                center = torch.randint(low_bound, high_bound, (1,)).item()
                indices = [center - self.d, center, center + self.d]
            
            slices = []
            for idx in indices:
                if axis == 0: slices.append(torch.stack([torch.full_like(self.grid_2d_x, idx), self.grid_2d_x, self.grid_2d_y], dim=-1))
                elif axis == 1: slices.append(torch.stack([self.grid_2d_x, torch.full_like(self.grid_2d_x, idx), self.grid_2d_y], dim=-1))
                else: slices.append(torch.stack([self.grid_2d_x, self.grid_2d_y, torch.full_like(self.grid_2d_x, idx)], dim=-1))
            return torch.stack(slices, dim=0)

    def __call__(self, data_item):
        grid, xvec, yvec, zvec, bounds, nx, ny, nz = data_item
        A_inv = self._get_Ainv_matrix(xvec, yvec, zvec)
        
        coords = self._get_slice_coords(nx, ny, nz) 
        flat_coords = coords.view(-1, 3)
        
        # --- FIXED: Apply full 3D affine augmentation to the 2D slice planes ---
        flat_coords = self._augment(flat_coords, bounds)
        
        grid3d = torch.remainder(flat_coords @ A_inv, torch.Tensor([nx, ny, nz]))
        
        scalar = torch.diag(2 / bounds)
        grid3d = torch.add(torch.mm(grid3d, scalar), -1)
        transf_grid = grid3d.view(1, 3, self.image_size, self.image_size, 3)

        grid_volume = grid.permute(0, 3, 2, 1).unsqueeze(0) 
        sampled_slices = F.grid_sample(grid_volume, transf_grid, align_corners=True)
        final_image = sampled_slices.squeeze(0).squeeze(0)
        
        c_min = final_image.view(3, -1).min(dim=1)[0].view(3, 1, 1)
        c_max = final_image.view(3, -1).max(dim=1)[0].view(3, 1, 1)
        final_image = (final_image - c_min) / (c_max - c_min + 1e-8)
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (final_image - mean) / std

class RectangularVolTransform:
    def __init__(self, crop_size=224, depth=64, grid_resolution=0.45, rotate=True, translate=True, **kwargs):
        self.crop_size = crop_size
        self.depth = depth
        self.grid_resolution = grid_resolution
        self.rotate = rotate
        self.translate = translate
        
        xs = torch.linspace(0, end=crop_size-1, steps=crop_size)
        ys = torch.linspace(0, end=crop_size-1, steps=crop_size)
        zs = torch.linspace(0, end=depth-1, steps=depth)
        xc = xs.repeat(crop_size*depth, 1).transpose(0, 1).contiguous().view(-1)
        yc = ys.repeat(depth, 1).transpose(0, 1).contiguous().view(-1).repeat(crop_size, 1).contiguous().view(-1)
        zc = zs.repeat(crop_size*crop_size)
        self.coord = torch.cat((xc.unsqueeze(1), yc.unsqueeze(1), zc.unsqueeze(1)), 1)

        # Precompute cyclical ImageNet stats
        rgb_mean = torch.tensor([0.485, 0.456, 0.406])
        rgb_std = torch.tensor([0.229, 0.224, 0.225])
        repeats = int(np.ceil(self.depth / 3.0))
        self.d_mean = rgb_mean.repeat(repeats)[:self.depth].view(self.depth, 1, 1)
        self.d_std = rgb_std.repeat(repeats)[:self.depth].view(self.depth, 1, 1)

    def _get_Ainv_matrix(self, xvec, yvec, zvec):
        x_unit = xvec / (self.grid_resolution / 0.15 * np.linalg.norm(xvec))
        y_unit = yvec / (self.grid_resolution / 0.15 * np.linalg.norm(yvec))
        z_unit = zvec / (self.grid_resolution / 0.15 * np.linalg.norm(zvec))
        return torch.inverse(torch.Tensor(np.vstack([x_unit, y_unit, z_unit])))

    def _augment(self, coord, bounds):
        if self.translate:
            coord += torch.rand(3) * bounds

        if self.rotate:
            p = torch.randn(3, 3) 
            rotation_matrix, _ = torch.linalg.qr(p)
            rand_int = torch.randint(0, 2, (1,)) 
            rotation_matrix = (2*rand_int-1) * rotation_matrix 
            rotation_matrix = torch.transpose(rotation_matrix, 0, 1) 
            coord = torch.mm(coord, rotation_matrix) 
        return coord

    def __call__(self, data_item):
        grid, xvec, yvec, zvec, bounds, nx, ny, nz = data_item
        A_inv = self._get_Ainv_matrix(xvec, yvec, zvec)
        
        # --- FIXED: Apply full 3D affine augmentation ---
        new_coord = self._augment(self.coord.clone(), bounds)

        grid3d = torch.remainder(new_coord @ A_inv, torch.Tensor([nx, ny, nz]))
        scalar = torch.diag(2 / bounds)
        grid3d = torch.add(torch.mm(grid3d, scalar), -1)

        transf_grid = grid3d.reshape(self.crop_size, self.crop_size, self.depth, 3)
        grid_vol = grid.permute(0, 3, 2, 1).unsqueeze(0)
        
        grid_res = F.grid_sample(grid_vol, transf_grid.unsqueeze(0), align_corners=True).squeeze(0).squeeze(0)
        grid_res = grid_res.permute(2, 0, 1)

        g_min, g_max = grid_res.min(), grid_res.max()
        grid_res = (grid_res - g_min) / (g_max - g_min + 1e-8)
        
        return (grid_res - self.d_mean) / self.d_std

class QuantileNormalize3DINO:
    """Normalizes a tensor based on 3DINO's 0.05% and 99.95% quantiles."""
    def __init__(self, q_min=0.0005, q_max=0.9995):
        self.q_min = q_min
        self.q_max = q_max

    def __call__(self, img):
        min_val = torch.quantile(img, self.q_min)
        max_val = torch.quantile(img, self.q_max)
        
        # Prevent division by zero for uniform blocks
        if max_val - min_val == 0:
            return torch.zeros_like(img)
            
        img = (img - min_val) / (max_val - min_val)
        return torch.clip(img * 2 - 1, -1, 1)
