import os
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
from tqdm import tqdm

from load_dataset import TrailerHitchSequenceDataset
from collate import collate_fn, collate_fn_pointnet
from models import GATSpatialTemporal
from loss_functions import combined_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau

# -------------------------------
# 🔧 Config
root_dir = "/home/furive-zo/Workspace/Study/Research/trailer-HAE/src/dataset"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 512
epochs = 100
lr = 1e-3
save_every = 10

TEMPORAL_TYPE = 'TCN' #  'TCN', 'LSTM', 'Transformer', 'MLP' 중 선택
FUSION_TYPE = 'gating' # 'gating', 'concat', 'cross', 'gating+cross', 'gating+cross+residual' 중 선택

save_dir = f"./ckpts/{TEMPORAL_TYPE}/{FUSION_TYPE}"
os.makedirs(save_dir, exist_ok=True)

# -------------------------------
# 📦 Dataset & Loader
train_dataset = TrailerHitchSequenceDataset(root_dir, sequence_length=32, sensor_sample_len=10, mode="train")
val_dataset = TrailerHitchSequenceDataset(root_dir, sequence_length=32, sensor_sample_len=10, mode="val")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# -------------------------------
# Model declare
model = GATSpatialTemporal(
    graph_input_dim=6,
    graph_hidden_dim=64,
    temporal_hidden_dim=64,
    num_heads=4,
    output_dim=1,
    temporal_type=TEMPORAL_TYPE,
    fusion_type=FUSION_TYPE,
    use_dgcnn=False 
)

# ✅ 파라미터 수 출력 추가
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ Loaded model 'GAT-{TEMPORAL_TYPE}-{FUSION_TYPE}' with {total_params:,} trainable parameters.\n")

# -------------------------------
# ⚙️ Optimizer & Scheduler
optimizer = Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# -------------------------------
# 📊 학습 기록 및 Early Stopping
best_val_loss = float('inf')
best_epoch = -1
early_stop_counter = 0
early_stop_patience = 10
best_ckpt_path = os.path.join(save_dir, f"BEST_GAT-{TEMPORAL_TYPE}-{FUSION_TYPE}.pt")

# -------------------------------
# 🏋️ Training Loop
for epoch in range(epochs):
    model.to(device)  
    model.train()
    total_loss = 0.0
    total_rmse = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
        angular = batch["angular"].to(device)
        vel     = batch["vel"].to(device)
        steer   = batch["steer"].to(device)
        graph   = batch["graph"]
        graph = graph.to(device)  
        target  = batch["hitch_angle"].to(device)
        batch_index = graph.batch.to(device) 

        preds = model(graph, vel, steer, angular, batch_index).squeeze(1)
        loss, rmse_deg = combined_loss(preds, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_rmse += rmse_deg.item()

    avg_loss = total_loss / len(train_loader)
    avg_rmse = total_rmse / len(train_loader)

    # -------------------------------
    model.eval()
    val_loss = 0.0
    val_rmse = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]"):
            angular = batch["angular"].to(device)
            vel     = batch["vel"].to(device)
            steer   = batch["steer"].to(device)
            graph   = batch["graph"]
            graph = graph.to(device)
            target  = batch["hitch_angle"].to(device)
            batch_index = graph.batch.to(device)

            preds = model(graph, vel, steer, angular, batch_index).squeeze(1)
            loss, rmse_deg = combined_loss(preds, target)
            val_loss += loss.item()
            val_rmse += rmse_deg.item()

    avg_val_loss = val_loss / len(val_loader)
    avg_val_rmse = val_rmse / len(val_loader)
    scheduler.step(avg_val_loss)

    print(f"[Epoch {epoch+1:03d}] "
          f"Train Loss: {avg_loss:.4f}, RMSE: {avg_rmse:.2f}° | "
          f"Valid Loss: {avg_val_loss:.4f}, RMSE: {avg_val_rmse:.2f}°")

    # 💾 체크포인트 저장
    if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
        ckpt_path = os.path.join(save_dir, f"GAT-{TEMPORAL_TYPE}-{FUSION_TYPE}_epoch{epoch+1:03d}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_loss,
            "train_rmse_deg": avg_rmse,
            "val_loss": avg_val_loss,
            "val_rmse_deg": avg_val_rmse
        }, ckpt_path)
        print(f"📦 Checkpoint saved: {ckpt_path}")

    # ✅ Best model 저장
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1
        early_stop_counter = 0
        torch.save(model.state_dict(), best_ckpt_path)
        print(f"✅ New best model saved at epoch {best_epoch}: Val Loss = {best_val_loss:.4f}")
    else:
        early_stop_counter += 1
        print(f"⏳ No improvement for {early_stop_counter} epoch(s).")

    # 🛑 Early stopping
    if early_stop_counter >= early_stop_patience:
        print(f"🛑 Early stopping triggered at epoch {epoch+1}")
        break
