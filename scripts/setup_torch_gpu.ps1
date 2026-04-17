Write-Host "[GPU setup] Installing CUDA 12.8 PyTorch wheels..." -ForegroundColor Cyan

poetry run python -m pip uninstall -y torch torchvision torchaudio | Out-Null

poetry run python -m pip install --index-url https://download.pytorch.org/whl/cu128 `
  torch==2.7.1+cu128 `
  torchvision==0.22.1+cu128 `
  torchaudio==2.7.1+cu128

if ($LASTEXITCODE -ne 0) {
  Write-Host "[GPU setup] Installation failed." -ForegroundColor Red
  exit 1
}

Write-Host "[GPU setup] Verifying CUDA availability..." -ForegroundColor Cyan
poetry run python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

if ($LASTEXITCODE -ne 0) {
  Write-Host "[GPU setup] Verification failed." -ForegroundColor Red
  exit 1
}

Write-Host "[GPU setup] Done." -ForegroundColor Green
