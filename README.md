
# 🛰️ Lidar Project - 3D Object Detector Evaluation using KITTI-360 Dataset

## 📌 Introduction

This project evaluates a 3D object detection pipeline using LiDAR data from the **KITTI-360 dataset** and 2D segmentation results from **YOLOv11**. We fuse 2D camera detections with 3D LiDAR point clouds to isolate car objects and compare them against labeled 3D bounding boxes (ground truth). This project was part of the "Lidar and Radar Systems" course (Chapter 8 focused heavily) under Prof. Dr. Stefan Elser.

---

## 🚦 Step-by-Step Pipeline

### 1. **YOLOv11 Segmentation**
- **Input**: Left camera images from KITTI-360.
- **Model**: `YOLOv11-seg.pt` (Ultralytics).
- **Goal**: Segment the “car” class only.
- **Output**: Binary masks saved as `.npy` files.

```python
model = YOLO("yolo11n-seg.pt")
for image_file in os.listdir(image_folder):
    ...
    masks = res.masks.data.cpu().numpy() if res.masks is not None else np.empty((0,))
    np.save(os.path.join(mask_folder, f"{frame_id}.npy"), masks)
```

---

### 2. **LiDAR Preprocessing**
- Load `.bin` files containing raw LiDAR point clouds.
- Extract `x, y, z` coordinates.

```python
points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
xyz = points[:, :3]
```

---

### 3. **Calibration (Transform LiDAR to Camera Space)**

#### Matrix Operation:

\[
\begin{bmatrix}
u \\
v \\
w \\
\end{bmatrix}
=
P_{\text{rect}} \cdot T_{\text{velo}\rightarrow\text{cam}} \cdot
\begin{bmatrix}
x \\
y \\
z \\
1 \\
\end{bmatrix}
\]

Then normalize:

\[
u' = \frac{u}{w}, \quad v' = \frac{v}{w}
\]

```python
points_hom = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))
points_cam = (Tr_velo_to_cam @ points_hom.T).T
points_2d = (P_rect_00 @ points_cam.T).T
u = (points_2d[:, 0] / points_2d[:, 2]).astype(np.int32)
v = (points_2d[:, 1] / points_2d[:, 2]).astype(np.int32)
```

---

### 4. **Object Association (Mask Filtering)**

```python
mask_hits = mask_bool[v, u]
subcloud = points_original[mask_hits]
```

---

### 5. **Bounding Box Calibration**

```python
Tr_cam_to_velo = np.linalg.inv(Tr_velo_to_cam)
bbox_hom = np.hstack((bbox_cam, np.ones((8, 1))))
bbox_velo = (Tr_cam_to_velo @ bbox_hom.T).T[:, :3]
bbox_velo += np.array(offset)  # minor correction
```

---

### 6. **Evaluation**

```python
for bbox in bboxes_velo:
    inside = get_points_inside_bbox(subcloud, bbox)
    ratio = len(inside) / len(subcloud)
    if ratio > 0.5:
        matched
```

---

## 📊 Results

| Car ID | 3D BBox ID | Total Points | Inside Points | Result (%) |
|--------|------------|--------------|----------------|------------|
| Car 1  | 218        | 2793         | 2190           | 78.41%     |
| Car 2  | 216        | 630          | 393            | 62.38%     |
| Car 3  | 215        | 338          | 263            | 77.81%     |
| Car 4  | 210        | 229          | 54             | 23.58%     |

> ✅ For GitHub: To convert your `Results.txt` file into a table like this, create a Markdown table inside your `README.md`.

---

## 🧠 Conclusion

While the algorithm can reliably segment and associate LiDAR points with YOLO masks, the approach is not robust enough for real-time applications due to:
- Occasional **YOLO segmentation errors**,
- **Bounding box misalignments** from calibration,
- Scene dynamics like **moving vehicles**.

### ✅ Improvements
- Custom training of YOLO on KITTI-360 images
- Better calibration with temporal smoothing
- Considering **3D segmentation** instead of 2D-to-3D projection

---

## 💡 How to Convert `Results.txt` to GitHub Table

```python
with open('Results.txt', 'r') as file:
    lines = file.readlines()

print("| Car | 3DBbox | Total Points | Inside Points | Result (%) |")
print("|-----|--------|---------------|----------------|------------|")

for line in lines[1:]:  # Skip header
    parts = line.strip().split()
    if len(parts) == 5:
        print(f"| {parts[0]} | {parts[1]} | {parts[2]} | {parts[3]} | {parts[4]} |")
```
