import os
import pickle
import numpy as np
from pathlib import Path

# ================= 配置区域 =================
# 必须与 preprocess_data_NUCLA.py 中的 OUTPUT_DIR 一致
root_path = Path(__file__).parent.parent.parent

DATA_DIR = os.path.join(root_path, r'datasets\NUCLA')
META_PATH = os.path.join(DATA_DIR, 'meta.pkl')
SKELETON_PATH = os.path.join(DATA_DIR, 'skeletons.npy')


# ===========================================

def check_generated_data():
    if not os.path.exists(META_PATH) or not os.path.exists(SKELETON_PATH):
        print(f"[错误] 找不到文件，请检查路径:\n {META_PATH}\n {SKELETON_PATH}")
        return

    # 1. 加载数据
    print(f"Loading meta from: {META_PATH}")
    with open(META_PATH, 'rb') as f:
        all_meta = pickle.load(f)

    print(f"Loading skeletons from: {SKELETON_PATH}")
    # allow_pickle=True 是必须的，因为我们要加载 object array
    all_skeletons = np.load(SKELETON_PATH, allow_pickle=True)

    # 2. 基础数量检查
    num_samples = len(all_meta)
    if len(all_skeletons) != num_samples:
        print(f"[严重错误] Meta 数量 ({len(all_meta)}) 与 Skeleton 数量 ({len(all_skeletons)}) 不一致！")
        return

    print(f"Total Samples Loaded: {num_samples}")
    print("=" * 100)
    # 打印表头
    header = f"{'Idx':<5} | {'Sample Name':<25} | {'Quality':<8} | {'Meta T':<6} | {'Pose Shape (C,T,V,M)':<22} | {'Status'}"
    print(header)
    print("-" * 100)

    # 3. 遍历所有实例
    # 计数器
    count_mismatch = 0
    count_missing = 0

    for idx, meta in enumerate(all_meta):
        # 获取元数据信息
        name = meta['sample_name']
        quality = meta['quality']
        frames_recorded = meta['num_frames']  # fileList.txt 里的长度

        # 获取骨架数据 (利用索引对齐)
        pose_data = all_skeletons[idx]
        pose_shape = pose_data.shape  # 预期 (3, T, 20, 1)

        # 获取实际骨架的时间维度 T (shape[1])
        frames_actual = pose_shape[1]

        # 检查一致性
        # 如果 quality 是 missing，长度应该都是 0
        # 如果是 intact/partial，meta 中的 num_frames 应该等于 pose 的第二维
        is_consistent = (frames_recorded == frames_actual)

        status_str = "OK"
        if not is_consistent:
            status_str = "MISMATCH!"
            count_mismatch += 1

        if quality == 'missing':
            count_missing += 1
            # 对于 missing，shape 应该是 (3, 0, 20, 1)
            if frames_actual != 0:
                status_str = "ERR_MISSING"

        # 打印行
        # 为了不刷屏太快，你可以选择只打印前 N 个，或者只打印有问题的
        # 这里按照你的要求“遍历所有实例”并输出
        print(f"{idx:<5} | {name:<25} | {quality:<8} | {frames_recorded:<6} | {str(pose_shape):<22} | {status_str}")

    print("=" * 100)
    print("Summary:")
    print(f"  - Total Samples: {num_samples}")
    print(f"  - Length Mismatches: {count_mismatch}")
    print(f"  - Missing Samples: {count_missing}")

    if count_mismatch == 0:
        print("\n✅ 数据完整性检查通过：所有样本的 Meta 记录帧数与 Skeleton 实际维度完全一致。")
    else:
        print("\n❌ 数据存在问题，请检查 MISMATCH 项。")


if __name__ == "__main__":
    check_generated_data()