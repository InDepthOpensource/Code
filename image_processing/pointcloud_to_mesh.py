import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    color_raw = o3d.io.read_image('123_rgb.jpg')
    depth_raw = o3d.io.read_image('123_original_depth_z16.png')

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_trunc=16.3, convert_rgb_to_intensity=False)
    print(rgbd_image)

    # plt.subplot(1, 2, 1)
    # plt.title('RGB image')
    # plt.imshow(rgbd_image.color)
    # plt.subplot(1, 2, 2)
    # plt.title('Depth image')
    # plt.imshow(rgbd_image.depth)
    # plt.show()

    start_time = time.time()

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsic(320, 256, 269, 269, 160, 128)))
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # Select pointcloud by index.
    # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2)
    # pcd = pcd.select_by_index(ind)

    # cl, ind = pcd.remove_radius_outlier(nb_points=10, radius=0.5)
    # pcd = pcd.select_by_index(ind)

    pcd.estimate_normals()
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    # Convex Hull - did not work
    # hull_mesh = pcd.compute_convex_hull()
    # o3d.visualization.draw_geometries(hull_mesh)


    # BPA meshing
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(
        [radius, radius * 2]))

    # poisson mesh
    # mesh, densities = \
    # o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)
    # vertices_to_remove = densities < np.quantile(densities, 0.02)
    # mesh.remove_vertices_by_mask(vertices_to_remove)

    dec_mesh = mesh.simplify_quadric_decimation(100000)
    dec_mesh.remove_degenerate_triangles()
    dec_mesh.remove_duplicated_triangles()
    dec_mesh.remove_duplicated_vertices()
    dec_mesh.remove_non_manifold_edges()

    # dec_mesh = dec_mesh.filter_smooth_taubin(number_of_iterations=50)
    dec_mesh.compute_vertex_normals()

    print(type(dec_mesh))
    print(time.time() - start_time)

    o3d.visualization.draw_geometries([dec_mesh], mesh_show_back_face=True)


