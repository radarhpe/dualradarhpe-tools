"""
Clone from https://github.com/nghiaho12/rigid_transform_3D
with some adjustment for input feeding
"""

import numpy as np
import os
import argparse
from scipy.spatial.transform import Rotation
from extrinsic_calib import create_transformation_matrix, src2dst_transformation, T_lidar_ti

def get_R_t(radar_points, lidar_points):
    print(radar_points.shape, lidar_points.shape)
    ## Centering
    centroid_radar = np.mean(radar_points, axis=1)
    centroid_radar = centroid_radar.reshape(-1, 1)
    centroid_lidar = np.mean(lidar_points, axis=1)
    centroid_lidar = centroid_lidar.reshape(-1, 1)
    # print("Mean radar: ", mean_radar)
    # print("Mean lidar: ", mean_lidar)

    # centering matrices
    tile_radar = radar_points - centroid_radar
    tile_lidar = lidar_points - centroid_lidar
    # print("Centering radar: ", tile_radar)
    # print("Centering lidar: ", tile_lidar)

    ## Compute covariance matrix
    cov_mat = tile_radar @ tile_lidar.T
    # print(cov_mat.shape)

    ## matrix decomposition on rotation, scaling and rotation matrices
    [U, S, Vt] = np.linalg.svd(cov_mat)

    ## Rotation mat
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T
    
    # print(R.shape)

    # R = np.array([[0.8889418, 0.4564966, -0.0373286],
    #               [0.4568565, -0.8895395,  0.0012595],
    #               [-0.0326303, -0.0181735, -0.9993023]])
    t = -R @ centroid_radar + centroid_lidar

    return R, t

def main():
    parser = argparse.ArgumentParser(
        prog="centroid_calculate.py",
        description="For calculating centroid of reflector on lidar"
    )

    parser.add_argument(
        "-f", "--file",
        type=str,
        help="Path to picked list"
    )

    args = parser.parse_args()

    # # Load data
    # data = None
    # with open(args.file, "r") as f:
    #     data = f.read().split("\n")[:-1]
    
    # n_point = len(data)

    # pairs = [[(3, 1.5, -0.53),(-0.348034, 1.20692, -0.235197)],
    #         [(3.5, 1.9, -0.15),(-0.374312,1.635597,-0.835999)],
    #         [(2.7, 1.8, -0.48),(-0.725562,1.265908,-0.103301)],
    #         [(3.6, 1.5, -0.43),(-0.02597,1.38417,-0.787136)],
    #         [(3.2, 1.7, -0.22),(-0.469156,1.584865,-0.544484)]]
    
    # lidar2optitrack
    pairs_06_03 = [[(-0.28, 1.8, -0.5), (3.4, 1.8, -0.0026)],
            [(-0.37, 1.3, 1.1), (1.9, 1.1, -0.54)],
            [(-0.11, 1.7, 0.5), (2.4, 1.1, -0.066)],
            [(-0.25, 1.7, 0.47), (2.5, 1.2, -0.17)],
            [(-0.078, 1.7, -2), (4.7, 2.4, -0.0037)]]
    
    pairs_01_04 = [[(0.41, 1.7, 0.46), (3.6, 1.2, -0.28)],
                [(0.63, 1.6, 1.2), (3.1, 0.63, -0.3)],
                [(-0.0028, 1.7, 0.9), (3.1, 1.5, -0.29)],
                [(-0.085, 1.7, 1.3), (2.7, 1.3, -0.26)],
                [(-0.4, 1.7, 0.12), (3.5, 2.1, -0.25)],
                [(1.2, 1.4, 2.4), (2.1, -0.29, -0.47)]]

    pairs_08_04 = [[(-0.33, 0.8, -0.62),(2.8, 1.6, -1)],
                    [(-0.39, 0.83, -0.67),(2.9, 1.7, -0.97)],
                    [(-0.045, 1.2, -0.81),(3.2, 1.6, -0.48)],
                    [(-1, 1.7, -1.3), (2.9, 2.6, -0.098)],
                    [(-1.6, 1.3, 0.11),(1.5, 2.3, -0.65)],
                    [(0.22, 1.6, 0.86),(1.9, 0.41, -0.21)],
                    [(0.24, 1.3, 0.86), (1.9, 0.43, -0.51)]]
    
    pairs_22_04 = [[(-0.28, 1.7, -2), (4.3, 2.3, -0.0033)], 
                # [(-0.25, 1.4, -2),(4.2, 2.3, -0.35)],
                [(-0.34, 1.4, 0.4),(2.1, 1.1, -0.49)],
                [(-0.74, 1.2, -1.1),(3.2, 2.2, -0.72)],
                [(-0.29, 1.2, -1.7),(4, 2.1, -0.44)],
                [(-0.058, 1.5, -2.1), (4.4, 2.1, -0.3)],
                [(-0.54, 1, -1.6), (3.8, 2.3, -0.76)]]
    
    pairs_30_04 = [[(-0.35, 1.7, -0.46), (3.1, 1.7, -0.088)],
                    [(-0.32, 1.4, -0.42), (3, 1.7, -0.42)],
                    [(0.71, 1.4, 0.32), (2.8, 0.39, -0.45)],
                    [(0.69, 1.7, 0.32), (2.8, 0.44, -0.14)],
                    [(-1.1, 1.7, 0.2), (2.1, 2, -0.18)]]
    
    radar2lidar = [[( 3.25   ,1.825, -0.35 ),(3.76666667 ,0.08666667 ,0.50333333)],
                    [( 2.4   ,1.85 ,-0.4 ),(3.03333333 ,0.52      , 0.35333333)],
                    [( 2.825 , 0.975, -0.385),( 3.03333333, -0.51666667 , 0.43666667)],
                    [(1.85   , 1.025 , -0.4775),( 2.16666667 ,0.033   ,   0.37)],
                    [( 3.15  ,  1.7   , -0.3925),(3.66666667 ,-0.028   ,   0.43333333)],
                   [(3.47, 1.68, -0.37), (3.89, -0.18, 0.36)],
                   [(2.94, 0.62, -0.40), (2.93, -0.89, 0.40)],
                   [(2.41, 1.67, -0.46), (2.99, 0.30, 0.30)],
                   [(2.10, 1.05, -0.46), (2.41, -0.09, 0.34)],
                   [(3.61, 1.22, -0.37), (3.77, -0.65, 0.44)],
                   [(1.80, 1.11, -0.46), (2.16, 0.07, 0.35)],
                   [(3.15, 1.12, -0.45), (3.37, -0.50, 0.47)]]

    # [(0.15, 1.7, -0.82),(3.1, 1.4, -0.083)]]
    # [(-0.0034, 1.4, -0.8), (3.1, 1.5, -0.33)]]
    
    # [(-1.2, 1.5, -2.4),(3.8, 3.5, -0.38)],
    #             [(-0.65, 1, -2.3),(4.1, 2.9, -0.73)],
    #             [(-1.1, 1.4, 0.071),(1.9, 2, -0.47)],
    # [(-0.99, 1.7, -2.3),(3.7, 3.1, -0.0034)],

    pairs = radar2lidar
    n_point = len(pairs)

    radar_points = np.zeros((n_point, 3), dtype=np.float64)
    lidar_points = np.zeros((n_point, 3), dtype=np.float64)
    
    # for i, line in enumerate(data):
    #     p_txt = line.split(" --- ")
    #     radar_points[i] = np.array([np.float64(p) for p in p_txt[0].split(",")])
    #     lidar_points[i] = np.array([np.float64(p) for p in p_txt[1].split(",")])

    for i, pair in enumerate(pairs):
        lidar_points[i] = np.array(pair[0])
        radar_points[i] = np.array(pair[1])

    # print(radar_points)
    # print(lidar_points)
    # Calibrate
    ## Reshaping to 3xn_point
    radar_points = radar_points.T
    lidar_points = lidar_points.T

    #
    T_arr = []
    err_arr = []
    homo_radar_points = np.ones((4, radar_points.shape[1]))
    homo_radar_points[:-1,:] = radar_points
    for i in range(radar_points.shape[1]):
        rad_points = np.append(radar_points[:,:i], radar_points[:,i+1:], axis=1)
        lid_points = np.append(lidar_points[:,:i], lidar_points[:,i+1:], axis=1)

        R, t = get_R_t(rad_points, lid_points)

        # Error calculation
        T_lid_rad = np.eye(4, 4)
        T_lid_rad[:3,:3] = R
        T_lid_rad[:3,3] = np.squeeze(t)
        T_arr.append(T_lid_rad)

        # make estimation
        # print(homo_radar_points)
        err = get_error(T_lid_rad, homo_radar_points, lidar_points)
        err_arr.append(err)
        # print(lidar_points)
        # print(np.mean(np.linalg.norm(estimation[:3,:]-lidar_points)))


    min_idx = np.argmin(err_arr)
    T_lid_rad = T_arr[min_idx]
    R, t = T_lid_rad[:3,:3], T_lid_rad[:3,3]

    print("Min err: ", err_arr[min_idx])
    print("R: ", R)
    print("R in quaternion: ", ", ".join(Rotation.from_matrix(R).as_quat().astype(str)))
    print("t: ", ", ".join(t.astype(str)))

    print(get_error(T_lidar_ti, homo_radar_points, lidar_points))

def get_error(T_lid_rad, homo_radar_points, lidar_points):
    estimation = (T_lid_rad @ homo_radar_points)[:3,:]
    return np.mean(np.linalg.norm(estimation - lidar_points, axis=0))

if __name__ == "__main__":
    main()