import pandaset
import os

import matplotlib.cm as cm
import numpy as np
from PIL import Image

# load dataset
dataset = pandaset.DataSet("./data/pandaset")
seq002 = dataset["001"]
seq002.load()

print("avaliable cameras: ", seq002.camera.keys())



from pandaset import geometry

# generate projected points
seq_idx = 0
camera_name = "front_right_camera"
lidar = seq002.lidar
point_cloud = lidar.data[seq_idx].to_numpy()

points3d_lidar_xyz = point_cloud[point_cloud[:, -1] == 0, :3]
choosen_camera = seq002.camera[camera_name]
projected_points2d, camera_points_3d, inner_indices = geometry.projection(lidar_points=points3d_lidar_xyz, 
                                                                          camera_data=choosen_camera[seq_idx],
                                                                          camera_pose=choosen_camera.poses[seq_idx],
                                                                          camera_intrinsics=choosen_camera.intrinsics,
                                                                          filter_outliers=True)
print("projection 2d-points inside image count:", projected_points2d.shape)



from matplotlib import pyplot as plt


# image before projection
ori_image = seq002.camera[camera_name][seq_idx]
# Get original image size
h, w = np.asarray(ori_image).shape[:2]



# image after projection
plt.imshow(ori_image)
distances = np.sqrt(np.sum(np.square(camera_points_3d), axis=-1))
colors = cm.jet(distances / np.max(distances))
# plt.gca().scatter(projected_points2d[:, 0], projected_points2d[:, 1], color=colors, s=0.2)

# plt.savefig("projection.png")


# Save at original pixel size
dpi = 100
fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
ax = fig.add_axes([0, 0, 1, 1])  # no margins

ax.imshow(ori_image)
ax.scatter(
    projected_points2d[:, 0],
    projected_points2d[:, 1],
    color=colors,
    s=4,          # smaller points; try 0.01, 0.03, 0.05
    linewidths=0
)

ax.set_xlim(0, w)
ax.set_ylim(h, 0)
ax.axis("off")

plt.savefig(
    "projection.png",
    dpi=dpi,
    bbox_inches=None,
    pad_inches=0
)

plt.close(fig)

# Optional check
saved = Image.open("projection.png")
print("saved image size:", saved.size)
print("original image size:", (w, h))