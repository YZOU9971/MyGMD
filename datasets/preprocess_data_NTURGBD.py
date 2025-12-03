# Generating meta file for NTU RGB+D 60/120 Datasets,
# and gathering .skeleton files into .npy file, bad_samples skipped (with T_raw = 0)
import os
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path

root_path = Path(__file__).parent.parent
DATA_ROOT = os.path.join(root_path, r"datasets\NTURGBD")
RAW_SKELETON_DIR = os.path.join(DATA_ROOT, "skeletons")
RGB_DIR= os.path.join(DATA_ROOT, "rgb")
OUTPUT_DIR = r"NTURGBD"
BAD_SAMPLES_FILE = r"NTURGBD\NTU_RGBD120_samples_with_missing_skeletons.txt"
CHECK_MODALITIES = ['rgb', 'depth', 'ir']


def load_bad_samples(txt_file):
    if not os.path.exists(txt_file): return set()
    bad_samples = set()
    with open(txt_file, 'r') as f:
        for line in f:
            name = line.strip()
            if name: bad_samples.add(name)
    return bad_samples


def read_raw_skeleton(file_path):
    """
    Returning (3, T, 25, 2) skeleton data for a single file
    """
    if not os.path.exists(file_path): return None
    with open(file_path, 'r') as f:
        try: num_frames = int(f.readline())
        except: return None

        raw_data = np.zeros((num_frames, 2, 25, 3), dtype=np.float32)
        current_line = f.readline()

        for t in range(num_frames):
            if not current_line: break
            num_bodies = int(current_line)
            for m in range(num_bodies):
                f.readline()
                num_joints = int(f.readline())
                for v in range(num_joints):
                    line = f.readline()
                    if m < 2:
                        raw_data[t, m, v, :] = list(map(float, line.split()))[:3]
            current_line = f.readline()
    # (T, M, V, C) -> (C, T, V, M)
    # print(raw_data)
    return raw_data.transpose(3, 0, 2, 1)


def check_files(name, root, mods):
    for m in mods:
        if m == 'rgb' and not os.path.exists(os.path.join(root, 'rgb', f"{name}_rgb.avi")): return False
        if m == 'masked_depth' and not os.path.exists(os.path.join(root, 'depth_masked', name)): return False
        if m == 'ir' and not os.path.exists(os.path.join(root, 'ir', f"{name}_ir.avi")): return False
    return True


def process():
    bad_samples = load_bad_samples(BAD_SAMPLES_FILE)
    print(bad_samples)

    print("Scanning samples...")
    all_names = [f.replace('_rgb.avi', '') for f in os.listdir(RGB_DIR) if f.endswith('_rgb.avi')]
    all_names.sort()

    meta_list = []
    skeleton_list = []
    print(f"Processing {len(all_names)} samples...")

    for name in tqdm(all_names):
        if not check_files(name, DATA_ROOT, CHECK_MODALITIES):
            continue
        try:
            action_id = int(name[name.find('A') + 1:name.find('A') + 4])
            setup_id = int(name[name.find('S') + 1:name.find('S') + 4])
            cam_id = int(name[name.find('C') + 1:name.find('C') + 4])
            subject_id = int(name[name.find('P') + 1:name.find('P') + 4])
        except: continue

        skeleton_path = os.path.join(RAW_SKELETON_DIR, name+".skeleton")
        is_bad = name in bad_samples
        if is_bad: print(name)
        skeleton_data = read_raw_skeleton(skeleton_path) if not is_bad else None
        if skeleton_data is None:
            skeleton_data = np.zeros((3, 0, 25, 2), dtype=np.float32)
            is_bad = True
        skeleton_list.append(skeleton_data)

        meta_list.append({
            'name': name,
            'label': action_id - 1,
            'setup_id': setup_id,
            'cam_id': cam_id,
            'subject_id': subject_id,
            'missing_skeleton': is_bad
        })
    print("Saving skeleton data...")
    all_skeletons = np.empty(len(skeleton_list), dtype=object)
    all_skeletons[:] = skeleton_list
    np.save(os.path.join(OUTPUT_DIR, 'skeletons.npy'), all_skeletons)
    print("Saving meta data...")
    with open(os.path.join(OUTPUT_DIR, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta_list, f)
    print(f"Complete, {len(meta_list)} samples saved.")


if __name__ == "__main__":
    process()

