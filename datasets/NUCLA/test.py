import os
import re
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg

# ================= 配置区域 =================
# Kinect V1 (20关节) 标准连接树
# 格式: (起点索引, 终点索引)
# 0:HipCenter, 1:Spine, 2:ShoulderCenter, 3:Head
# 4:ShoulderLeft, 5:ElbowLeft, 6:WristLeft, 7:HandLeft
# 8:ShoulderRight, 9:ElbowRight, 10:WristRight, 11:HandRight
# 12:HipLeft, 13:KneeLeft, 14:AnkleLeft, 15:FootLeft
# 16:HipRight, 17:KneeRight, 18:AnkleRight, 19:FootRight
BONES = [
    (0, 1), (1, 2), (2, 3),  # 躯干
    (2, 4), (4, 5), (5, 6), (6, 7),  # 左臂
    (2, 8), (8, 9), (9, 10), (10, 11),  # 右臂
    (0, 12), (12, 13), (13, 14), (14, 15),  # 左腿
    (0, 16), (16, 17), (17, 18), (18, 19)  # 右腿
]
# ===========================================

DATASET_ROOT = r''
VIEW = 'view_1'
OUTPUT_CSV = 'n_ucla_missing_frames.csv'

# 定义需要查找的后缀
REQUIRED_MODALITIES = {
    'rgb': '_rgb.jpg',
    'depth': '_depth.png',
    'map': '_maprgbd.png',
    'skeleton': '_skeletons.txt',
    'depth_vis': '_depth_vis.jpg'  # 新增可视化的深度图后缀
}


def parse_skeletons_file(file_path):
    with open(file_path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    if not lines:
        return None, None

    skel_id = int(lines[0])
    joints = []
    for line in lines[1:]:
        # 解析: -0.619699,0.328046,3.18186,2
        parts = line.split(',')
        if len(parts) >= 4:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            conf = int(parts[3])
            joints.append([x, y, z, conf])

    return skel_id, np.array(joints)


def load_clip_data(clip_folder):
    # 修改逻辑：扫描文件夹下所有文件，按帧号聚合 RGB, Depth, Depth_Vis 和 Skeleton
    files = os.listdir(clip_folder)

    frame_map = {}
    # 结构: { fid: {'rgb': path, 'depth': path, 'depth_vis': path, 'skel_path': path} }
    pattern = re.compile(r'frame_(\d+)_', re.IGNORECASE)

    print(f'Loading frames from {clip_folder}...')

    for fname in files:
        match = pattern.match(fname)
        if match:
            fid = int(match.group(1))
            if fid not in frame_map:
                frame_map[fid] = {'rgb': None, 'depth': None, 'depth_vis': None, 'skel_path': None}

            fpath = os.path.join(clip_folder, fname)
            lower_name = fname.lower()

            # 匹配不同模态的文件
            if lower_name.endswith(REQUIRED_MODALITIES['skeleton']):
                frame_map[fid]['skel_path'] = fpath
            elif lower_name.endswith(REQUIRED_MODALITIES['rgb']):
                frame_map[fid]['rgb'] = fpath
            elif lower_name.endswith(REQUIRED_MODALITIES['depth']):
                frame_map[fid]['depth'] = fpath
            elif lower_name.endswith(REQUIRED_MODALITIES['depth_vis']):
                frame_map[fid]['depth_vis'] = fpath


    # 整理数据列表
    sorted_fids = sorted(frame_map.keys())

    skeletons = []
    rgb_images = []
    depth_images = []
    depth_vis_images = []

    valid_skeleton_found = False

    for fid in sorted_fids:
        item = frame_map[fid]

        # --- 1. 处理骨架 ---
        # 即使没有骨架文件，也要占位 (用 NaN 填充)，保证时间轴对齐
        current_skel = np.full((20, 3), np.nan)  # 默认全为 NaN

        if item['skel_path']:
            _, skel_data = parse_skeletons_file(item['skel_path'])
            if skel_data is not None and skel_data.shape[0] == 20:
                current_skel = skel_data[:, :3]
                valid_skeleton_found = True

        skeletons.append(current_skel)

        # --- 2. 读取 RGB ---
        if item['rgb'] and os.path.exists(item['rgb']):
            rgb_images.append(mpimg.imread(item['rgb']))
        else:
            rgb_images.append(np.zeros((240, 320, 3), dtype=np.uint8))

        # --- 3. 读取 Depth ---
        if item['depth'] and os.path.exists(item['depth']):
            img = mpimg.imread(item['depth'])
            depth_images.append(img)
        else:
            depth_images.append(np.zeros((240, 320, 3), dtype=np.uint8))

        # --- 4. 读取 Depth Vis ---
        if item['depth_vis'] and os.path.exists(item['depth_vis']):
            depth_vis_images.append(mpimg.imread(item['depth_vis']))
        else:
            depth_vis_images.append(np.zeros((240, 320, 3), dtype=np.uint8))

    if not skeletons:
        return None

    return {
        'skeletons': np.array(skeletons),
        'rgb': rgb_images,
        'depth': depth_images,
        'depth_vis': depth_vis_images,
        'has_valid_skeleton': valid_skeleton_found
    }


def visualize_3d_clip(clip_data):
    if clip_data is None:
        print("无数据可显示")
        return

    skeleton_seq = clip_data['skeletons']
    rgb_seq = clip_data['rgb']
    depth_seq = clip_data['depth']
    depth_vis_seq = clip_data['depth_vis']

    frames, num_joints, dims = skeleton_seq.shape

    fig = plt.figure(figsize=(12, 10))

    main_title = fig.suptitle(f'Frame 1/{frames}', fontsize=16, fontweight='bold')

    # --- 子图1 (Top-Left): RGB ---
    ax_rgb = fig.add_subplot(221)
    ax_rgb.set_title("RGB Video")
    ax_rgb.axis('off')
    img_plot_rgb = ax_rgb.imshow(rgb_seq[0])

    # --- 子图2 (Top-Right): 3D Skeleton ---
    ax_3d = fig.add_subplot(222, projection='3d')
    ax_3d.set_title("3D Pose")

    if clip_data['has_valid_skeleton']:
        all_x = skeleton_seq[:, :, 0]
        all_y = skeleton_seq[:, :, 2]  # Swap Y-Z
        all_z = skeleton_seq[:, :, 1]
        ax_3d.set_xlim(np.nanmin(all_x), np.nanmax(all_x))
        ax_3d.set_ylim(np.nanmin(all_y), np.nanmax(all_y))
        ax_3d.set_zlim(np.nanmin(all_z), np.nanmax(all_z))
    else:
        ax_3d.set_xlim(-1, 1)
        ax_3d.set_ylim(1, 4)
        ax_3d.set_zlim(-1, 1)

    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Depth')
    ax_3d.set_zlabel('Height')

    # 初始化 3D 元素
    scat = ax_3d.scatter([], [], [], c='r', s=20)
    lines = [ax_3d.plot([], [], [], 'b-', linewidth=2)[0] for _ in BONES]

    ax_3d.view_init(elev=0, azim=-90)

    # --- 子图3 (Bottom-Left): Depth ---
    ax_depth = fig.add_subplot(223)
    ax_depth.set_title("Depth Map")
    ax_depth.axis('off')
    img_plot_depth = ax_depth.imshow(depth_seq[0])

    # --- 子图4 (Bottom-Right): Depth Vis ---
    ax_depth_vis = fig.add_subplot(224)
    ax_depth_vis.set_title("Depth Visualization")
    ax_depth_vis.axis('off')
    img_plot_depth_vis = ax_depth_vis.imshow(depth_vis_seq[0])

    plt.tight_layout(rect=[0, 0, 1, 1])

    def update(frame_idx):
        img_plot_rgb.set_data(rgb_seq[frame_idx])
        img_plot_depth.set_data(depth_seq[frame_idx])
        img_plot_depth_vis.set_data(depth_vis_seq[frame_idx])

        current_pose = skeleton_seq[frame_idx]

        if not np.isnan(current_pose).any():
            xs = current_pose[:, 0]
            ys = current_pose[:, 2]
            zs = current_pose[:, 1]

            scat._offsets3d = (xs, ys, zs)

            for line, (start, end) in zip(lines, BONES):
                line.set_data([xs[start], xs[end]], [ys[start], ys[end]])
                line.set_3d_properties([zs[start], zs[end]])
        else:
            scat._offsets3d = ([], [], [])
            for line in lines:
                line.set_data([], [])
                line.set_3d_properties([])

        main_title.set_text(f'Frame {frame_idx + 1}/{frames}')

        return lines + [scat, img_plot_rgb, img_plot_depth, img_plot_depth_vis]

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
    plt.show()


def check_dataset_integrity(root_path):

    report_rows = []
    view_dirs = glob(os.path.join(root_path, 'view_*'))
    print(f"scaning root path: {root_path}")

    for view_dir in view_dirs:
        view_name = os.path.basename(view_dir)
        clip_dirs = [d for d in glob(os.path.join(view_dir, '*')) if os.path.isdir(d)]
        for clip_dir in tqdm(clip_dirs, desc=f"scanning {view_name}"):
            clip_name = os.path.basename(clip_dir)
            frames_status = {}
            files = os.listdir(clip_dir)
            pattern = re.compile(r'frame_(\d+)_.*', re.IGNORECASE)
            for fname in files:
                match = pattern.match(fname)
                if match:
                    fid = int(match.group(1))
                    if fid not in frames_status:
                        frames_status[fid] = {k: False for k in REQUIRED_MODALITIES.keys()}
                    for mod_name, suffix in REQUIRED_MODALITIES.items():
                        if fname.endswith(suffix):
                            frames_status[fid][mod_name] = True
                            break
            all_fids = sorted(frames_status.keys())

        if not all_fids:
        # 如果 all_fids 为空，说明文件夹内没有匹配到任何有效的帧文件
            report_rows.append({
                'View': view_name,
                'Clip': clip_name,
                'Frame_ID': 'N/A',
                'Missing': 'ALL'
            })
            # 打印警告信息并跳过这个文件夹，避免 IndexError
            tqdm.write(f"[警告] 文件夹 {clip_name} 中未找到符合 'frame_XXX' 模式的文件。跳过。")
            continue
        min_f, max_f = all_fids[0], all_fids[-1]

        for fid in all_fids:
            status = frames_status[fid]
            missing = [k for k, v in status.items() if not v]

            if missing:
                report_rows.append({
                    'View': view_name,
                    'Clip': clip_name,
                    'Frame_ID': fid,
                    'Missing': "|".join(missing)
                })

        if report_rows:
            df = pd.DataFrame(report_rows)
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"{len(df)} missing")
            print(f"saved {OUTPUT_CSV}")
            print(df.head())


if __name__ == '__main__':
    # check_dataset_integrity(DATASET_ROOT)
    view, action, subject, episode = [1, 1, 8, 2]
    # action in [01, 02, 03, 04, 05, 06, 08, 09, 11, 12]
    # subject in range(1, 10)
    # episode in [00, 01, 02, 03, 04, *05, *06, ...]
    CLIP_PATH = f'view_{view}\\a{action:02}_s{subject:02}_e{episode:02}'

    # 加载数据
    data = load_clip_data(CLIP_PATH)

    # 可视化
    visualize_3d_clip(data)