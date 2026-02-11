# 4090 = sm_89, A800 = sm_80, H100/H200 = sm_90
TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0" pip wheel . --no-deps --no-build-isolation -w dist/