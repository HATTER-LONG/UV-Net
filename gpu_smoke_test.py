"""Verify the modern CUDA/PyG environment on a real NVIDIA GPU."""

import sys

import torch
import torch_geometric
from torch import nn
from torch_geometric.nn import NNConv, global_max_pool


def main() -> int:
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"PyTorch Geometric: {torch_geometric.__version__}")

    if not torch.cuda.is_available():
        print("CUDA is not available to this process.", file=sys.stderr)
        print("Check `nvidia-smi` and the /dev/nvidia* device nodes.", file=sys.stderr)
        return 1

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    print(f"GPU: {properties.name}")
    print(f"Compute capability: {properties.major}.{properties.minor}")
    print(f"VRAM: {properties.total_memory / 1024**3:.1f} GiB")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_float32_matmul_precision("high")

    # Exercise dense CUDA kernels similar to UV-Net's curve/surface encoders.
    surface_encoder = nn.Sequential(
        nn.Conv2d(7, 64, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),
        nn.AdaptiveAvgPool2d(1),
    ).to(device)
    surface_grid = torch.randn(64, 7, 10, 10, device=device)
    surface_embedding = surface_encoder(surface_grid).flatten(1)

    # Exercise edge-conditioned graph message passing and pooling on CUDA.
    node_features = torch.randn(8, 16, device=device, requires_grad=True)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 0, 5, 6, 7, 4]],
        device=device,
    )
    edge_features = torch.randn(edge_index.size(1), 8, device=device)
    edge_network = nn.Sequential(nn.Linear(8, 16 * 32)).to(device)
    graph_conv = NNConv(16, 32, edge_network, aggr="add").to(device)
    node_embedding = graph_conv(node_features, edge_index, edge_features)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device=device)
    graph_embedding = global_max_pool(node_embedding, batch)

    loss = surface_embedding.square().mean() + graph_embedding.square().mean()
    loss.backward()
    torch.cuda.synchronize(device)

    print(f"Surface encoder output: {tuple(surface_embedding.shape)}")
    print(f"Graph encoder output: {tuple(graph_embedding.shape)}")
    print(f"Loss: {loss.item():.6f}")
    print("CUDA dense + PyG forward/backward: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
