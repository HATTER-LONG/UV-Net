"""Modern PyTorch Geometric loader for neutral MFCAD NPZ samples."""

import json
from pathlib import Path
import random

import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset


def _center_and_scale(face_uv, edge_uv):
    visible_points = face_uv[..., :3][face_uv[..., 6] == 1]
    if visible_points.numel() == 0:
        raise ValueError("Cannot normalize a solid without visible face samples")
    minimum = visible_points.amin(dim=0)
    maximum = visible_points.amax(dim=0)
    extent = maximum - minimum
    longest_extent = extent.amax()
    if longest_extent <= 0:
        raise ValueError("Cannot normalize a solid with a zero-size bounding box")
    center = 0.5 * (minimum + maximum)
    scale = 2.0 / longest_extent
    face_uv[..., :3].sub_(center).mul_(scale)
    edge_uv[..., :3].sub_(center).mul_(scale)


def _random_axis_rotation():
    axis = random.randrange(3)
    quarter_turns = random.randrange(4)
    angle = quarter_turns * torch.pi / 2
    cosine = torch.cos(torch.as_tensor(angle, dtype=torch.float32))
    sine = torch.sin(torch.as_tensor(angle, dtype=torch.float32))
    rotation = torch.eye(3, dtype=torch.float32)
    if axis == 0:
        rotation[1:, 1:] = torch.tensor([[cosine, -sine], [sine, cosine]])
    elif axis == 1:
        rotation[0, 0] = cosine
        rotation[0, 2] = sine
        rotation[2, 0] = -sine
        rotation[2, 2] = cosine
    else:
        rotation[:2, :2] = torch.tensor([[cosine, -sine], [sine, cosine]])
    return rotation


def _rotate_grid(grid, rotation):
    grid[..., :3] = grid[..., :3].reshape(-1, 3).matmul(rotation).reshape(
        grid[..., :3].shape
    )
    grid[..., 3:6] = grid[..., 3:6].reshape(-1, 3).matmul(rotation).reshape(
        grid[..., 3:6].shape
    )


class MFCADPyGDataset(Dataset):
    """Load converted MFCAD samples as PyG Data objects."""

    num_classes = 16

    def __init__(
        self,
        root_dir,
        split="train",
        center_and_scale=True,
        random_rotate=False,
    ):
        split = {"val": "validation"}.get(split, split)
        if split not in ("train", "validation", "test"):
            raise ValueError(f"Unsupported split: {split}")

        self.root = Path(root_dir)
        self.neutral_dir = self.root / "neutral"
        self.center_and_scale = center_and_scale
        self.random_rotate = random_rotate

        with (self.root / "split.json").open("r") as read_file:
            splits = json.load(read_file)
        self.sample_ids = splits[split]

        missing = [
            sample_id
            for sample_id in self.sample_ids
            if not (self.neutral_dir / f"{sample_id}.npz").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} converted samples; first missing sample: {missing[0]}"
            )

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        with np.load(self.neutral_dir / f"{sample_id}.npz", allow_pickle=False) as sample:
            if int(sample["schema_version"]) != 1:
                raise ValueError(f"Unsupported schema for sample {sample_id}")
            face_uv = torch.from_numpy(sample["face_uv"].copy())
            edge_uv = torch.from_numpy(sample["edge_uv"].copy())
            edge_index = torch.from_numpy(sample["edge_index"].copy()).long()
            node_y = torch.from_numpy(sample["node_y"].copy()).long()

        if self.center_and_scale:
            _center_and_scale(face_uv, edge_uv)
        if self.random_rotate:
            rotation = _random_axis_rotation()
            _rotate_grid(face_uv, rotation)
            _rotate_grid(edge_uv, rotation)

        return Data(
            face_uv=face_uv,
            edge_uv=edge_uv,
            edge_index=edge_index,
            y=node_y,
            num_nodes=face_uv.shape[0],
            sample_id=sample_id,
        )
