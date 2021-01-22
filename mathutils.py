import numpy as np
import torch
from torch.nn.functional import grid_sample


# meshgrid : [3, ...]
# returns: [2, ...]
def project_onto_detector(meshgrid, pmat):
    mshape = meshgrid.shape
    meshgrid = meshgrid.view(3, -1).float()
    meshgrid = torch.cat(
        [
            meshgrid,
            torch.ones(1, meshgrid.shape[-1], device=meshgrid.device)
        ],
        dim=0)
    hom_det = torch.mm(pmat, meshgrid)  # [3, ...]
    detector = hom_det[:2].clone()  # [2, ...]
    detector[0] /= hom_det[2]
    detector[1] /= hom_det[2]
    detector = detector.view(2, *mshape[1:])
    return detector[0], detector[1]


# image: [W, H]
# samples_x: [W, H]
# samples_y: [W, H]
def bilinear_interpolate(image, samples_x, samples_y):
    W, H = image.shape
    samples_x = samples_x.unsqueeze(0)
    samples_x = samples_x.unsqueeze(3)
    samples_y = samples_y.unsqueeze(0)
    samples_y = samples_y.unsqueeze(3)
    samples = torch.cat([samples_x, samples_y], 3)
    samples[:, :, :, 0] = (samples[:, :, :, 0]/(W - 1))
    samples[:, :, :, 1] = (samples[:, :, :, 1]/(H - 1))
    samples = samples*2 - 1
    return grid_sample(
        image[None, None, ...],
        samples,
        align_corners=True,
        )[0, 0]


# uv_grid: [u, v, 2]
# returns [u, v, 3]
def uv_to_raydir(uv_grid, projection_matrix):
    # make coordinates homogeneous
    uvw_grid = torch.cat([
        uv_grid,
        torch.ones(*uv_grid.shape[:2], 1, dtype=torch.double)
    ], dim=-1)

    M = projection_matrix[:, :3]
    q, r = torch.qr(M.transpose(0, 1))
    r_sign = r.diag().prod().sign()

    back_sub = torch.triangular_solve(
        uvw_grid.unsqueeze(-1).cuda(),
        r,
        transpose=True)[0]  # (u, v, 3, 1)
    raydirs = r_sign*torch.matmul(q, back_sub)[..., 0]
    norm_raydirs = raydirs/torch.norm(raydirs, dim=-1, keepdim=True)
    return norm_raydirs


# x, y: (*, n) or broadcastable
# returns (*, 1), i.e. inner product along last axis
def bdot(x, y):
    return torch.sum(x*y, dim=-1, keepdim=True)


# source_position: numpy array, 3d vector
# ray direction: [*, 3], unit vectors
# returns [*, 3]
def point_of_interest(source_position, ray_direction):
    # x0 = np.array([0.0]*3)
    # n = np.array([0.0, 0.0, 1.0])
    scaling = bdot(source_position[:2], ray_direction[..., :2]) /\
        (1. - ray_direction[..., 2:]*ray_direction[..., 2:])
    return source_position - scaling*ray_direction


def compute_single_g_approx(g, poi_grid, pmat):
    pmat_t = torch.from_numpy(np.array(pmat)).float().cuda()
    g_coords = project_onto_detector(poi_grid, pmat_t)
    g_interp = bilinear_interpolate(g, g_coords[0], g_coords[1])
    return g_interp


def interpolate_projections(projection_a, projection_b, pmat_a, pmat_b,
                            pmat_int, uv_grid):
    pmat_int_mat = torch.from_numpy(np.array(pmat_int)).cuda()
    dir_grid = uv_to_raydir(uv_grid, pmat_int_mat)  # [u, v, 3]
    source_pos_int = torch.from_numpy(
        np.array(pmat_int.source_position())[:, 0]).cuda()
    poi_int = point_of_interest(source_pos_int, dir_grid)  # [u, v, 3]
    g0 = projection_a[0]
    g1 = projection_b[0]
    g_approx_0 = compute_single_g_approx(g0, poi_int.permute(2, 0, 1), pmat_a)
    g_approx_1 = compute_single_g_approx(g1, poi_int.permute(2, 0, 1), pmat_b)
    return g_approx_0, g_approx_1
