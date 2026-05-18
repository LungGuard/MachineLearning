"""Quick validation on the best checkpoint."""
import sys, pathlib, torch, torch.nn as nn
sys.path.insert(0, '.')
import lightning as L
from DetectionModel.src.models.resnet_nodule_model import ResNetNoduleModel
from DetectionModel.src.data_modules.regression_dataset_module import RegressionDataModule
from DetectionModel.constants.constants.regression_model import RegressionModelConstants
from DetectionModel.constants.constants.dataset import DatasetConstants
from common.constants import Accelerator

import pathlib as _pathlib
class _FakePath(_pathlib.PurePosixPath):
    pass
try:
    torch.serialization.add_safe_globals([pathlib.WindowsPath, pathlib.PosixPath, nn.MSELoss])
except Exception:
    pass
torch.serialization.add_safe_globals([_FakePath])

dm = RegressionDataModule(
    metadata_csv=DatasetConstants.DATASET_DIR,
    dataset_root=DatasetConstants.PROJECT_ROOT,
    crop_size=224, batch_size=32, num_workers=0,  # 0 workers to avoid spawn issue
)

import glob
ckpts = sorted(glob.glob(str(RegressionModelConstants.CHECKPOINT_DIR / "best_reg_model*.ckpt")))
if not ckpts:
    print("No checkpoint found")
    sys.exit(1)

ckpt_path = ckpts[-1]
print(f"Loading checkpoint: {ckpt_path}")

model = ResNetNoduleModel.load_from_checkpoint(ckpt_path)
trainer = L.Trainer(accelerator=Accelerator.AUTO, devices=1, logger=False, enable_progress_bar=True)
result = trainer.validate(model=model, datamodule=dm, verbose=True)
print("\nBest checkpoint validation:", result)
