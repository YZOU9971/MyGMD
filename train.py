import time
import torch

from torch.utils.data import DataLoader

from data.dataset import get_dataset
from models.UnifyModel import UnifyModel
from models.solver import Solver


class MockDataset(torch.utils.data.Dataset):
    def __init__(self, length=64):
        self.length = length
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        return {
            'rgb': torch.randn(3, 64, 224, 224),        #
            'ir': torch.randn(3, 64, 224, 224),
            'depth': torch.randn(3, 64, 224, 224),
            'pose': torch.randn(3, 64, 25, 2),
        }

def get_dataloader(args, BATCH_SIZE):
    train_set = get_dataset('NTU120', 'train', args)
    # train_set = get_dataset('NUCLA', 'train', args)
    # train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
    return DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=8, pin_memory=True, prefetch_factor=2, persistent_workers=True)

args = {
    'benchmark': 'xsub',
    'modalities': ['rgb', 'pose', 'depth', 'ir'],
    # 'modalities': ['pose'],
    'num_frames': 32,
    'use_val': False  # 暂时关闭 val 切分，简化调试
}
BATCH_SIZE = 4
ACCUM_STEPS = 8
# 等效batch_size = 4 * 8 = 32

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UnifyModel(num_classes=120).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    model_params_count = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_count = sum(p.numel() for p in trainable_params) / 1e6
    print(f"Model Total Params: {model_params_count:.2f} M")
    print(f"Trainable Params:   {trainable_count:.2f} M (Visual Backbones Frozen)")
    optimizer = torch.optim.Adam(trainable_params, lr=0.001, weight_decay=1e-4)

    solver = Solver(model, optimizer, mode='base', lambda_spec=1, accum=ACCUM_STEPS)
    train_loader = get_dataloader(args, BATCH_SIZE)
    print(f"Data Ready. Batches per Epoch: {len(train_loader)}")

    for epoch in range(3):
        model.train()
        start = time.time()
        total_loss = 0

        print(f"\nEpoch {epoch + 1}/{3}")

        for i, batch in enumerate(train_loader):
            loss_val, loss_dict = solver.train_step(batch, device)

            total_loss += loss_val
            if i % 10 == 0:
                log_str = f"Iter {i} | Total: {loss_val:.4f} | Shared: {loss_dict['shared']:.4f}"
                if 'spec_total' in loss_dict:
                    log_str += f" | Spec: {loss_dict['spec_total']:.4f}"
                # 打印具体的 specific loss (如果有)
                for k in ['rgb', 'pose', 'depth', 'ir']:
                    if k in loss_dict: log_str += f" | {k}: {loss_dict[k]:.4f}"
                print(log_str)

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch Finished. Avg Loss: {avg_loss:.4f}. Time: {time.time() - start:.1f}s")

if __name__ == '__main__':
    main()

