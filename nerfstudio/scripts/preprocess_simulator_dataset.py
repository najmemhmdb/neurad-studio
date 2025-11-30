import json
import os
import numpy as np
import cv2

# ---------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------
JSON_INTRINSIC_PATH = "./data/Simulator_1Nov2025_undistorted/labels/sensor_parameters.json"
CSV_DISTORTION_PATH = "./data/Simulator_1Nov2025_undistorted/labels/lens_distortion_i49.csv"

# Folder with distorted images
INPUT_IMAGE_DIR = "./data/Simulator_1Nov2025_undistorted/images/agent_1/camera_4/"
# Folder where undistorted images will be stored
OUTPUT_IMAGE_DIR = "./data/Simulator_1Nov2025_undistorted/images/agent_1/camera_4_undistorted/"
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

# Output JSON (same structure as original, new intrinsics)
OUT_JSON_INTRINSIC_PATH = "./data/Simulator_1Nov2025_undistorted/labels/sensor_parameters_undistorted.json"

# ---------------------------------------------------------
# 2. Load intrinsics from the meta dict
# ---------------------------------------------------------
def get_K_from_meta(meta, agent="agent_1", camera="camera_4"):
    intr = meta[agent][camera]["intrinsic"]
    fx = intr["fx"]
    fy = intr["fy"]
    cx = intr["cx"]
    cy = intr["cy"]

    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return K

# ---------------------------------------------------------
# 3. Read CSV and fit fisheye distortion coefficients
# ---------------------------------------------------------
def load_height_theta_table(csv_path):
    with open(csv_path, "r") as f:
        lines = f.read().splitlines()

    delta_height = float(lines[0].split(",")[1])

    heights = []
    thetas = []
    for line in lines[2:]:
        if not line.strip():
            continue
        h_str, t_str = line.split(",")
        heights.append(float(h_str))
        thetas.append(float(t_str))

    heights = np.array(heights, dtype=np.float64)
    thetas = np.array(thetas, dtype=np.float64)
    return delta_height, heights, thetas


def fit_fisheye_coeffs_from_table(heights, thetas):
    """
    OpenCV fisheye model:
        r = f * (θ + k1 θ^3 + k2 θ^5 + k3 θ^7 + k4 θ^9)

    Table gives r (height on sensor, mm) vs θ.
    We fit:
        r = a1 θ + a2 θ^3 + a3 θ^5 + a4 θ^7 + a5 θ^9
    Then:
        f = a1
        k1..k4 = a2..a5 divided by f
    """
    mask = thetas > 0.0
    t = thetas[mask]
    r = heights[mask]

    X = np.stack([t, t**3, t**5, t**7, t**9], axis=1)  # (N, 5)
    a, *_ = np.linalg.lstsq(X, r, rcond=None)

    f_mm = a[0]
    k1 = a[1] / f_mm
    k2 = a[2] / f_mm
    k3 = a[3] / f_mm
    k4 = a[4] / f_mm

    D = np.array([[k1], [k2], [k3], [k4]], dtype=np.float64)
    return D


# ---------------------------------------------------------
# 4. Undistort one image, returning new camera matrix
# ---------------------------------------------------------
def undistort_image_fisheye(image, K, D, balance=0.0, scale=1.0):
    """
    balance in [0,1]: 0 = crop more, 1 = keep full FOV.
    scale: additional scaling of the new camera matrix.
    """
    h, w = image.shape[:2]

    K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=balance, new_size=(w, h)
    )

    K_new[:2, :2] *= scale

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K_new, (w, h), cv2.CV_16SC2
    )

    undistorted = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
    return undistorted, K_new


# ---------------------------------------------------------
# 5. Main pipeline
# ---------------------------------------------------------
def main():
    # 5.1 Load original JSON (full structure)
    with open(JSON_INTRINSIC_PATH, "r") as f:
        meta = json.load(f)

    # 5.2 Get K for a reference camera (camera_4 here)
    agent_name = "agent_1"
    ref_camera_name = "camera_4"
    K = get_K_from_meta(meta, agent=agent_name, camera=ref_camera_name)

    # 5.3 Load distortion table and fit fisheye coefficients
    _, heights, thetas = load_height_theta_table(CSV_DISTORTION_PATH)
    D = fit_fisheye_coeffs_from_table(heights, thetas)

    print("Original camera matrix K:\n", K)
    print("Estimated fisheye distortion D (k1..k4):\n", D.ravel())

    K_new_global = None

    # 5.4 Process all images in INPUT_IMAGE_DIR
    for name in sorted(os.listdir(INPUT_IMAGE_DIR)):
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(INPUT_IMAGE_DIR, name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Could not read {img_path}")
            continue

        undist, K_new = undistort_image_fisheye(img, K, D,
                                               balance=0.0,
                                               scale=1.0)
        # store one representative K_new (same for all images of same size)
        K_new_global = K_new

        out_path = os.path.join(OUTPUT_IMAGE_DIR, name)
        cv2.imwrite(out_path, undist)
        print(f"Processed {name}")

    if K_new_global is None:
        print("No images processed, not writing new sensor file.")
        return

    print("Final new camera matrix K_new:\n", K_new_global)

    # 5.5 Update intrinsics in the meta dict for all cameras
    fx_new = float(K_new_global[0, 0])
    fy_new = float(K_new_global[1, 1])
    cx_new = float(K_new_global[0, 2])
    cy_new = float(K_new_global[1, 2])

    # Here we assume all cameras share the same intrinsics originally.
    # If you only want to change one camera, modify this loop.
    for cam_name, cam_data in meta[agent_name].items():
        if cam_name.startswith("camera_") and "intrinsic" in cam_data:
            cam_data["intrinsic"]["fx"] = fx_new
            cam_data["intrinsic"]["fy"] = fy_new
            cam_data["intrinsic"]["cx"] = cx_new
            cam_data["intrinsic"]["cy"] = cy_new

    # 5.6 Save a new sensor_pareameters json with same structure
    with open(OUT_JSON_INTRINSIC_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved updated sensor parameters to {OUT_JSON_INTRINSIC_PATH}")


if __name__ == "__main__":
    main()
