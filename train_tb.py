import os
import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.tensorboard import SummaryWriter
from model import LTCNetwork          # 你的模型

# ======= 可调超参数 =======
DATA_DIR   = 'datasets'
MODEL_DIR  = 'models'
LOG_ROOT   = 'runs'
NUM_EPOCHS = 2500
BATCH_SIZE = 256
LR         = 1e-2
HIDDEN_SZ  = 10
HIST_EVERY = 200          # 多少 epoch 记录一次权重直方图
# =========================

os.makedirs(MODEL_DIR, exist_ok=True)

def grad_global_norm(model):
    """计算当前模型所有参数的 L2 总范数（√∑‖g_i‖²）"""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().data.norm(2).item() ** 2
    return total ** 0.5

def train_and_save_models(data_dir: str):
    for file_name in os.listdir(data_dir):
        if not file_name.endswith('.pt'):
            continue
        country_code = file_name.split('.')[0]
        print(f'\n==== Training [{country_code}] ====')

        # ---------- 数据 ----------
        X_data, y_data = torch.load(os.path.join(data_dir, file_name))
        X_tensor = torch.as_tensor(X_data, dtype=torch.float32)
        y_tensor = torch.as_tensor(y_data, dtype=torch.float32).view(-1, 1)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_tensor, y_tensor, test_size=0.3, random_state=42)
        train_loader = DataLoader(TensorDataset(X_tr, y_tr),
                                  batch_size=BATCH_SIZE, shuffle=True)
        test_loader  = DataLoader(TensorDataset(X_te, y_te),
                                  batch_size=BATCH_SIZE, shuffle=False)

        # ---------- 模型 ----------
        model = LTCNetwork(input_size=2, hidden_size=HIDDEN_SZ, output_size=1)
        criterion = torch.nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.5)

        # ---------- TensorBoard ----------
        writer = SummaryWriter(log_dir=os.path.join(LOG_ROOT, country_code))

        # 历史指标缓存
        history = {k: [] for k in ('train_mse', 'train_mae', 'test_mse',
                                   'test_mae', 'test_r2', 'grad_norm')}

        # ========== 训练循环 ==========
        for epoch in range(NUM_EPOCHS):
            # ----- Train -----
            model.train()
            y_pred_tr, y_true_tr = [], []
            for x_b, y_b in train_loader:
                optimizer.zero_grad()
                out, _ = model(x_b)
                loss = criterion(out.squeeze(), y_b.squeeze())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

                # 记录当前 batch 梯度（累积，稍后取平均）
                optimizer.step()
                y_pred_tr.append(out.detach().cpu().numpy())
                y_true_tr.append(y_b.detach().cpu().numpy())

            scheduler.step()

            y_pred_tr = np.concatenate(y_pred_tr)
            y_true_tr = np.concatenate(y_true_tr)
            train_mse = np.mean((y_pred_tr - y_true_tr) ** 2)
            train_mae = mean_absolute_error(y_true_tr, y_pred_tr)

            # 梯度范数（取刚刚更新完的最后一次 backward 的结果就足够）
            g_norm = grad_global_norm(model)

            # ----- Eval -----
            model.eval()
            with torch.no_grad():
                y_pred_te, y_true_te = [], []
                for x_b, y_b in test_loader:
                    out, _ = model(x_b)
                    y_pred_te.append(out.cpu().numpy())
                    y_true_te.append(y_b.cpu().numpy())
                y_pred_te = np.concatenate(y_pred_te)
                y_true_te = np.concatenate(y_true_te)

            test_mse = np.mean((y_pred_te - y_true_te) ** 2)
            test_mae = mean_absolute_error(y_true_te, y_pred_te)
            test_r2  = r2_score(y_true_te, y_pred_te)

            # ----- TensorBoard -----
            writer.add_scalars('MSE',  {'train': train_mse, 'test': test_mse}, epoch)
            writer.add_scalars('MAE',  {'train': train_mae, 'test': test_mae}, epoch)
            writer.add_scalar ('R2/test',    test_r2, epoch)
            writer.add_scalar ('LR',         optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar ('GradNorm',   g_norm, epoch)

            # 参数直方图（每 HIST_EVERY epoch）
            if (epoch + 1) % HIST_EVERY == 0 or epoch == 0:
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.detach().cpu(), epoch)

            # ----- Save history -----
            history['train_mse'].append(train_mse)
            history['train_mae'].append(train_mae)
            history['test_mse'].append(test_mse)
            history['test_mae'].append(test_mae)
            history['test_r2'].append(test_r2)
            history['grad_norm'].append(g_norm)

            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1:4d}/{NUM_EPOCHS} | "
                      f"Train MSE {train_mse:.5f} | "
                      f"Test MSE {test_mse:.5f} | R2 {test_r2:.4f} | "
                      f"Grad {g_norm:.3f} | LR {optimizer.param_groups[0]['lr']:.6f}")

        # ---------- 保存 ----------
        weight_path = os.path.join(MODEL_DIR, f'{country_code}_model.pth')
        torch.save(model.state_dict(), weight_path)
        print(f'✓ Saved weights to {weight_path}')

        np.save(os.path.join(MODEL_DIR, f'{country_code}_history.npy'), history)
        with open(os.path.join(MODEL_DIR, f'{country_code}_history.json'), 'w') as f:
            json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f)

        writer.close()

if __name__ == "__main__":
    train_and_save_models(DATA_DIR)
