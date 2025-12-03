import os
import re
import glob
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path

# ================= 配置区域 =================
root_path = Path(__file__).parent.parent
DATASET_ROOT = os.path.join(root_path, r"datasets\NUCLA")  # N-UCLA 数据集根目录
OUTPUT_DIR = DATASET_ROOT  # 输出目录
print(OUTPUT_DIR)

# N-UCLA 参数
NUM_JOINTS = 20
MAX_BODY = 1
NUM_CHANNELS = 3


def parse_filename(folder_name):
    match = re.search(r'a(\d+)_s(\d+)_e(\d+)', folder_name, re.IGNORECASE)
    if match:
        return {
            'action': int(match.group(1)),
            'subject': int(match.group(2)),
            'episode': int(match.group(3))
        }
    return None


def get_master_frame_list(clip_path):
    """
    读取 fileList.txt 作为 Master Timeline
    """
    filelist_path = os.path.join(clip_path, 'fileList.txt')
    if not os.path.exists(filelist_path):
        return None

    frame_ids = []
    try:
        with open(filelist_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts and parts[0].isdigit():
                    frame_ids.append(int(parts[0]))
    except:
        return None
    return frame_ids


def read_single_frame_skeleton(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        joints = []
        for line in lines[1:]:
            parts = line.split(',')
            if len(parts) >= 3:
                joints.append([float(parts[0]), float(parts[1]), float(parts[2])])

        if len(joints) != NUM_JOINTS:
            return None

        return np.array(joints, dtype=np.float32)
    except Exception:
        return None


def read_clip_skeletons_aligned(clip_path, master_frame_ids):
    """
    返回: (data_array, valid_frame_count)
    data_array shape: (C, T, V, M)
    """
    if not master_frame_ids:
        return None, 0

    skel_files = glob.glob(os.path.join(clip_path, '*_skeletons.txt'))
    skel_map = {}
    pattern = re.compile(r'frame_(\d+)_', re.IGNORECASE)

    for fp in skel_files:
        match = pattern.search(os.path.basename(fp))
        if match:
            fid = int(match.group(1))
            skel_map[fid] = fp

    # 2. 初始化全 0 容器
    total_frames = len(master_frame_ids)
    # (T, V, C) -> (T, 20, 3)
    raw_numpy = np.zeros((total_frames, NUM_JOINTS, NUM_CHANNELS), dtype=np.float32)

    # 3. 填充
    valid_count = 0
    for idx, fid in enumerate(master_frame_ids):
        if fid in skel_map:
            data = read_single_frame_skeleton(skel_map[fid])
            if data is not None:
                raw_numpy[idx] = data
                valid_count += 1

    if valid_count == 0:
        return None, 0

    # 4. 维度转换 (T, V, C) -> (C, T, V, M)
    data = raw_numpy.transpose(2, 0, 1)
    data = data[:, :, :, np.newaxis]

    return data, valid_count


def process():
    print(f"Scanning N-UCLA dataset from: {DATASET_ROOT}")

    meta_list = []
    skeleton_list = []

    # 用于记录分类统计
    stats = {
        'intact': [],
        'partial': [],
        'missing': []
    }

    view_dirs = sorted(glob.glob(os.path.join(DATASET_ROOT, 'view_*')))
    sample_index = 0

    for view_dir in view_dirs:
        view_name = os.path.basename(view_dir)
        try:
            view_id = int(view_name.split('_')[-1])
        except:
            view_id = 0

        print(f"Processing {view_name} ...")
        clip_dirs = sorted(glob.glob(os.path.join(view_dir, 'a*_s*_e*')))

        for clip_path in tqdm(clip_dirs):
            folder_name = os.path.basename(clip_path)
            name = f"{view_name}_{folder_name}"  # Unique Key

            info = parse_filename(folder_name)
            if info is None: continue

            # === 1. 读取 Master Frame List (fileList.txt) ===
            master_frames = get_master_frame_list(clip_path)

            data = None
            valid_count = 0

            if master_frames:
                # === 2. 读取骨架并对齐 ===
                data, valid_count = read_clip_skeletons_aligned(clip_path, master_frames)

            # === 3. 动态判别状态 ===
            if data is None:
                status = 'missing'
                # 缺失样本：生成一个空的 (3, 0, 20, 1) 占位
                data = np.zeros((NUM_CHANNELS, 0, NUM_JOINTS, MAX_BODY), dtype=np.float32)
                is_missing = True
                total_expected = 0
            else:
                total_expected = len(master_frames)
                is_missing = False

                if valid_count == total_expected:
                    status = 'intact'
                else:
                    status = 'partial'  # 有效帧数 < 总帧数，已补0

            # 记录状态
            stats[status].append(name)

            # =================================================================
            # 【新增】在此处直接打印 Partial 和 Missing 的样本名称及详细信息
            # =================================================================
            if status != 'intact':
                msg = f"[{status.upper()}] {name}"
                if status == 'partial':
                    msg += f" -> (Valid Frames: {valid_count}/{total_expected})"
                elif status == 'missing':
                    if not master_frames:
                        msg += " -> (Reason: No fileList.txt found)"
                    else:
                        msg += " -> (Reason: fileList exists but no matching skeleton files)"

                tqdm.write(msg)  # 使用 tqdm.write 防止打乱进度条
            # =================================================================

            # 添加到总列表
            skeleton_list.append(data)
            meta_list.append({
                'sample_name': name,
                'file_name': folder_name,
                'label': info['action'] - 1,
                'view_id': view_id,
                'subject_id': info['subject'],
                'episode_id': info['episode'],
                'num_frames': data.shape[1],  # 对齐后的总长度
                'valid_frames': valid_count,  # 实际有效骨架数
                'missing_skeleton': is_missing,
                'quality': status,  # 'intact', 'partial', 'missing'
                'original_index': sample_index
            })
            sample_index += 1

    # === 保存统计文件 ===
    print(f"\nProcessing complete.")
    for cat, items in stats.items():
        print(f"  - {cat.capitalize()}: {len(items)}")
        with open(os.path.join(OUTPUT_DIR, f"nucla_{cat}.txt"), 'w') as f:
            for item in sorted(items):
                f.write(item + '\n')

    print("\nSaving skeletons.npy ...")
    all_skeletons = np.empty(len(skeleton_list), dtype=object)
    all_skeletons[:] = skeleton_list
    np.save(os.path.join(OUTPUT_DIR, 'skeletons.npy'), all_skeletons)

    print("Saving meta.pkl ...")
    with open(os.path.join(OUTPUT_DIR, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta_list, f)

    print(f"All files saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    process()