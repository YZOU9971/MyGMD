import time
import torch

from torch.utils.data import DataLoader

from data.dataset import get_dataset
from models.UnifyModel import UnifyModel
from models.solver import Solver

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
EPOCH = 3
BASE_LR = 1e-4
WEIGHT_DECAY = 1e-4

MODE = 'base'
LAMBDA_SPEC = 1.0
LAMBDA_ORTH = 0.1

def get_dataloader(args, BATCH_SIZE):
    train_set = get_dataset('NTU120', 'train', args)
    # train_set = get_dataset('NUCLA', 'train', args)
    # train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
    return DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=True
    )

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Start Training | Device: {device} | Mode: {MODE}")
    print(f"Physical Batch: {BATCH_SIZE} | Accumulation: {ACCUM_STEPS} | Effective Batch: {BATCH_SIZE * ACCUM_STEPS}")

    model = UnifyModel(num_classes=120).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    model_params_count = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_count = sum(p.numel() for p in trainable_params) / 1e6
    print(f"Model Total Params: {model_params_count:.2f} M")
    print(f"Trainable Params:   {trainable_count:.2f} M (Visual Backbones Frozen)")

    optimizer = torch.optim.Adam(trainable_params, lr=0.001, weight_decay=1e-4)

    solver = Solver(
        model,
        optimizer,
        mode=MODE,
        lambda_spec=LAMBDA_SPEC,
        lambda_orth=LAMBDA_ORTH,
        accum_steps=ACCUM_STEPS
    )

    train_loader = get_dataloader(args, BATCH_SIZE)
    print(f"Data Ready. Batches per Epoch: {len(train_loader)}")

    global_step = 0

    for epoch in range(1, EPOCH + 1):
        model.train()
        start = time.time()
        total_loss = 0

        print(f"\n=== Epoch {epoch}/{EPOCH} ===")

        for i, batch in enumerate(train_loader):
            loss_val, loss_dict = solver.train_step(batch, device)

            total_loss += loss_val
            global_step += 1

            if i % 10 == 0:
                log_str = f"Iter {global_step} | Total: {loss_val:.4f}"

                if 'shared' in loss_dict:
                    log_str += f" | Shared: {loss_dict['shared']:.4f}"

                if 'orth' in loss_dict and MODE == 'GGR':
                    log_str += f" | Orth: {loss_dict['orth']:.4f}"

                for k in ['rgb', 'pose', 'depth', 'ir']:
                    if k in loss_dict:
                        log_str += f" | {k}: {loss_dict[k]:.4f}"

                print(log_str)

        avg_loss = total_loss / len(train_loader)
        epoch_time = (time.time() - start) / 60
        print(f"Epoch {epoch} Finished. Avg Loss: {avg_loss:.4f}. Time: {epoch_time:.1f} min")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), f'epoch_{epoch}.pth')

if __name__ == '__main__':
    main()

