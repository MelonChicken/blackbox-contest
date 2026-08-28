from src.config import DEVICE
from src.train import fit_stage1, fit_stage2, fit_stage3

print('device:',DEVICE)
fit_stage1(); print('Stage 1 완료')
fit_stage2(); print('Stage 2 완료')
fit_stage3(); print('Stage 3 완료')