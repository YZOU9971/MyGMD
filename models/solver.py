import torch
import torch.nn as nn
import torch.nn.functional as F

from models.GMDtools import GMD  # 确保 GMDtools 在同一目录或路径下


class Solver:
    def __init__(self, model, optimizer, mode='base', lambda_spec=1.0, accum=1):
        """
        mode:
            'base': Joint training (L_shared + L_spec)
            'DGL' : Detached Gradient Learning (Fusion detached)
            'GMD' : Gradient Modulation (PCGrad style between Shared vs Spec)
            'GGR' : Gradient-Guided Routing (Your Method - Orthogonalization)
        """
        self.model = model
        self.mode = mode
        self.lambda_spec = lambda_spec
        self.accum_steps = accum
        self.criterion = nn.CrossEntropyLoss()

        self.step_count = 0

        if self.mode == 'GMD':
            self.optimizer = GMD(optimizer, reduction='mean')
        else:
            self.optimizer = optimizer

        print(f"Solver initialized in mode: [{self.mode.upper()}] | Accumulation: {self.accum_steps}")

    def train_step(self, batch, device):
        """
        Execute one training iteration.
        """
        # 1. Prepare Data
        x_rgb = batch['rgb'].to(device)
        x_ir = batch['ir'].to(device)
        x_depth = batch['depth'].to(device)
        x_pose = batch['pose'].to(device)
        targets = batch['label'].to(device)

        # ==================================================
        # MODE: GGR (Ours - Gradient Orthogonalization)
        # ==================================================
        # 将 Ours 放在最前，因为逻辑最特殊
        self.step_count += 1
        is_update_step = (self.step_count % self.accum_steps == 0)
        if self.mode == 'GGR':
            # 1. Forward (Retain Grad enabled internally)
            logits_shared, logits_spec = self.model( x_rgb, x_ir, x_depth, x_pose, gradient_control='GGR')

            # 2. Compute Individual Losses (No backward on total yet!)
            l_shared = self.criterion(logits_shared, targets)
            l_specs = {k: self.criterion(v, targets) for k, v in logits_spec.items()}

            l_shared_scaled = l_shared / self.accum_steps
            l_specs_scaled = {k: v / self.accum_steps for k, v in l_specs.items()}

            # 3. Apply Gradient Routing
            # 修正：传入计算好的 loss
            self._apply_ggr_routing(l_shared_scaled, l_specs_scaled)

            # 4. Step
            if is_update_step:
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Logging
            l_spec_sum = sum(l_specs.values())
            loss_total_log = l_shared + l_spec_sum * self.lambda_spec
            loss_dict = {'shared': l_shared.item(), 'spec_total': l_spec_sum.item()}
            return loss_total_log.item(), loss_dict

        # ==================================================
        # MODE: Base (Joint Training) / DGL (Detached Gradient)
        # ==================================================
        elif self.mode in ['base', 'DGL']:
            ctrl = 'DGL' if self.mode == 'DGL' else 'base'
            logits_shared, logits_spec = self.model(x_rgb, x_ir, x_depth, x_pose, gradient_control=ctrl)

            loss_total, loss_dict = self._compute_sum_loss(logits_shared, logits_spec, targets)

            loss_scaled = loss_total / self.accum_steps
            loss_scaled.backward()

            if is_update_step:
                self.optimizer.step()
                self.optimizer.zero_grad()

            return loss_total.item(), loss_dict

        # ==================================================
        # MODE: GMD (PCGrad)
        # ==================================================
        elif self.mode == 'GMD':
            logits_shared, logits_spec = self.model(x_rgb, x_ir, x_depth, x_pose, gradient_control='GMD')

            l_shared = self.criterion(logits_shared, targets)
            l_spec_sum = sum([self.criterion(v, targets) for v in logits_spec.values()])
            l_spec_weighted = l_spec_sum * self.lambda_spec


            objectives = [l_shared / self.accum_steps, l_spec_weighted / self.accum_steps]

            self.optimizer.pc_backward(objectives)

            if is_update_step:
                self.optimizer.step()
                self.optimizer.zero_grad()

            loss_total = l_shared + l_spec_weighted
            loss_dict = {'shared': l_shared.item(), 'spec_total': l_spec_sum.item()}
            return loss_total.item(), loss_dict

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _apply_ggr_routing(self, l_shared, l_specs):
        """
        GGR 核心逻辑：
        拦截 l_shared 和 l_spec 的梯度，在 Feature (z) 层做正交化，
        然后手动反向传播给 Backbone。
        """
        # 1. 清空之前的梯度 (Backbone parameters)
        self.optimizer.zero_grad()

        features = self.model.features

        for modality, z in features.items():
            # z 是中间变量，它可能有残留梯度，先清空
            if z.grad is not None:
                z.grad.zero_()

            # --- Step A: 分别获取 Shared 和 Specific 对 z 的梯度 ---

            # 1. g_shared = d(L_shared) / d(z)
            # retain_graph=True 是必须的，因为我们要对 z 求两次导
            g_shared = torch.autograd.grad(l_shared, z, retain_graph=True, allow_unused=True)[0]

            # 2. g_spec = d(L_spec) / d(z)
            # 找到对应模态的 specific loss
            l_this_spec = l_specs[modality] * self.lambda_spec
            g_spec = torch.autograd.grad(l_this_spec, z, retain_graph=True, allow_unused=True)[0]

            # 处理 None (有些模态可能未参与特定 Loss 计算)
            if g_shared is None: g_shared = torch.zeros_like(z)
            if g_spec is None: g_spec = torch.zeros_like(z)

            # --- Step B: 正交化处理 (Orthogonalization) ---
            # 目标：从 Shared 梯度中剔除 Specific 方向的分量
            # g_shared_orth = g_shared - Proj(g_shared on g_spec)

            # 1. 展平梯度
            g_sh_flat = g_shared.view(g_shared.size(0), -1)
            g_sp_flat = g_spec.view(g_spec.size(0), -1)

            # 2. 计算投影 (Proj Coef)
            # dot: (B, 1)
            dot_product = (g_sh_flat * g_sp_flat).sum(dim=1, keepdim=True)
            norm_sq = (g_sp_flat * g_sp_flat).sum(dim=1, keepdim=True) + 1e-8
            proj_coef = dot_product / norm_sq

            # 3. 剔除分量 (Rejection)
            # g_shared_orth: (B, Dim)
            g_shared_orth = g_shared - proj_coef.view(-1, 1) * g_spec

            # --- Step C: 合并梯度并回传 ---
            # 最终传给 Backbone 的是：原始 Specific 梯度 + 处理后的 Shared 梯度
            g_final = g_spec + g_shared_orth

            # 手动 Backward：将修改后的梯度传给 z 之前的层 (Backbone)
            z.backward(g_final)

    def _compute_sum_loss(self, logits_shared, logits_spec, targets):
        l_shared = self.criterion(logits_shared, targets)
        l_specs_dict = {k: self.criterion(v, targets) for k, v in logits_spec.items()}
        l_spec_sum = sum(l_specs_dict.values())
        total_loss = l_shared + self.lambda_spec * l_spec_sum

        loss_dict = {'shared': l_shared.item()}
        loss_dict.update({k: v.item() for k, v in l_specs_dict.items()})
        return total_loss, loss_dict