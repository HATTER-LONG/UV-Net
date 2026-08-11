# Processing your own data

We provide scripts to process your own STEP file data into the DGL bin format that UV-Net consumes, point clouds in NPZ format and render meshes (non-watertight meshes) in STL format.

Example usage:

```
cd /path/to/uv_net
python -m process.solid_to_graph /path/to/input/step_files /path/to/output/bin_graphs
```

Other scripts can be run similarly. For more details, run the script with the `--help` argument.

## Convert the official MFCAD DGL graphs for modern PyTorch/PyG

The official preprocessed MFCAD dataset can be converted losslessly from its
DGL 0.6 `.bin` files into framework-neutral, compressed NPZ files. Run the
converter with the original Python 3.9/DGL environment:

```
DGLBACKEND=pytorch .venv/bin/python -m process.convert_mfcad_dgl_to_npz \
  ./data/mfcad --workers 8
```

The converter preserves DGL edge-ID order and writes raw, unnormalized arrays
to `data/mfcad/neutral/`. Existing samples are skipped, so the command can be
resumed safely. Use `--overwrite` to regenerate them.

The converted samples can be loaded in the modern GPU environment as PyG data:

```python
from datasets.pyg_mfcad import MFCADPyGDataset
from torch_geometric.loader import DataLoader

dataset = MFCADPyGDataset("data/mfcad", split="train")
loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
batch = next(iter(loader))
```

Each PyG batch contains `face_uv`, `edge_uv`, `edge_index`, node labels `y`,
and the standard PyG node-to-graph vector `batch`.
