from data.dataset import get_dataset
from torch.utils.data import DataLoader

args = {
    'benchmark': 'xsub',
    'modalities': ['rgb', 'pose', 'depth', 'ir'],
    'num_frames': 64,
    'use_val': False  # 暂时关闭 val 切分，简化调试
}

train_loader = DataLoader(get_dataset('NTU120', 'train', args), batch_size=4)

for batch in train_loader:
    print(batch['pose'].shape)
