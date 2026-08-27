# Check GPU path
import mlx.core as mx

print("MLX default device:", mx.default_device())
# Expect: Device(gpu, 0) on Apple Silicon