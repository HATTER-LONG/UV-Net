"""Convert the official MFCAD DGL archives into framework-neutral NPZ files."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("DGLBACKEND", "pytorch")

from dgl.data.utils import load_graphs
import numpy as np
from tqdm import tqdm


SCHEMA_VERSION = 1
NUM_CLASSES = 16


def _load_labels(label_path):
    with label_path.open("r") as read_file:
        labels_data = json.load(read_file)
    return np.asarray(
        [face["segment"]["index"] for face in labels_data["body"]["faces"]],
        dtype=np.int64,
    )


def _validate_arrays(sample_id, face_uv, edge_uv, edge_index, node_y):
    if face_uv.ndim != 4 or face_uv.shape[1:] != (10, 10, 7):
        raise ValueError(f"{sample_id}: invalid face_uv shape {face_uv.shape}")
    if edge_uv.ndim != 3 or edge_uv.shape[1:] != (10, 6):
        raise ValueError(f"{sample_id}: invalid edge_uv shape {edge_uv.shape}")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"{sample_id}: invalid edge_index shape {edge_index.shape}")
    if edge_index.shape[1] != edge_uv.shape[0]:
        raise ValueError(
            f"{sample_id}: {edge_index.shape[1]} graph edges but "
            f"{edge_uv.shape[0]} edge feature grids"
        )
    if node_y.ndim != 1 or node_y.shape[0] != face_uv.shape[0]:
        raise ValueError(
            f"{sample_id}: {face_uv.shape[0]} graph nodes but "
            f"{node_y.shape[0]} face labels"
        )
    if edge_index.size and (
        edge_index.min() < 0 or edge_index.max() >= face_uv.shape[0]
    ):
        raise ValueError(f"{sample_id}: edge_index contains an invalid node index")
    if node_y.size and (node_y.min() < 0 or node_y.max() >= NUM_CLASSES):
        raise ValueError(f"{sample_id}: labels must be in [0, {NUM_CLASSES - 1}]")


def _convert_one(task):
    sample_id, graph_path, label_path, output_path, overwrite, compress = task
    graph_path = Path(graph_path)
    label_path = Path(label_path)
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        return sample_id, "skipped", None

    temporary_path = None
    try:
        graph = load_graphs(str(graph_path))[0][0]
        src, dst = graph.edges(order="eid")

        # Copy raw arrays without normalization or augmentation. Edge features and
        # edge_index use the same DGL edge-ID order.
        face_uv = graph.ndata["x"].detach().cpu().numpy().astype(np.float32, copy=True)
        edge_uv = graph.edata["x"].detach().cpu().numpy().astype(np.float32, copy=True)
        edge_index = np.stack(
            (src.detach().cpu().numpy(), dst.detach().cpu().numpy()), axis=0
        ).astype(np.int64, copy=False)
        node_y = _load_labels(label_path)

        _validate_arrays(sample_id, face_uv, edge_uv, edge_index, node_y)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".npz", dir=output_path.parent, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            save = np.savez_compressed if compress else np.savez
            save(
                temporary_file,
                schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int16),
                sample_id=np.asarray(sample_id),
                face_uv=face_uv,
                edge_uv=edge_uv,
                edge_index=edge_index,
                node_y=node_y,
            )
        os.replace(temporary_path, output_path)
        return sample_id, "converted", None
    except Exception as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        return sample_id, "failed", f"{type(error).__name__}: {error}"


def _load_official_splits(root):
    with (root / "split.json").open("r") as read_file:
        splits = json.load(read_file)
    expected_keys = {"train", "validation", "test"}
    if set(splits) != expected_keys:
        raise ValueError(f"Expected split keys {expected_keys}, found {set(splits)}")
    all_ids = [sample_id for names in splits.values() for sample_id in names]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("The official splits contain duplicate sample IDs")
    return splits


def convert(args):
    root = Path(args.dataset_path).resolve()
    output_dir = (
        Path(args.output).resolve() if args.output else root.joinpath("neutral")
    )
    splits = _load_official_splits(root)

    tasks = []
    missing_inputs = []
    for names in splits.values():
        for sample_id in names:
            graph_path = root / "graph" / f"{sample_id}.bin"
            label_path = root / "labels" / f"{sample_id}_ids.json"
            if not graph_path.is_file() or not label_path.is_file():
                missing_inputs.append(sample_id)
                continue
            tasks.append(
                (
                    sample_id,
                    str(graph_path),
                    str(label_path),
                    str(output_dir / f"{sample_id}.npz"),
                    args.overwrite,
                    not args.no_compress,
                )
            )
    if missing_inputs:
        raise FileNotFoundError(
            f"Missing graph or label files for {len(missing_inputs)} samples; "
            f"first missing sample: {missing_inputs[0]}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.workers > 0:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(
                tqdm(
                    executor.map(_convert_one, tasks, chunksize=16),
                    total=len(tasks),
                    desc="Converting MFCAD",
                )
            )
    else:
        results = [
            _convert_one(task)
            for task in tqdm(tasks, desc="Converting MFCAD")
        ]

    failures = [(sample_id, error) for sample_id, status, error in results if status == "failed"]
    if failures:
        for sample_id, error in failures[:20]:
            print(f"FAILED {sample_id}: {error}")
        raise RuntimeError(f"Failed to convert {len(failures)} samples")

    counts = {
        status: sum(result_status == status for _, result_status, _ in results)
        for status in ("converted", "skipped")
    }
    manifest = {
        "schema": "uvnet-mfcad-neutral",
        "schema_version": SCHEMA_VERSION,
        "num_classes": NUM_CLASSES,
        "format": "npz-compressed" if not args.no_compress else "npz-store",
        "features": {
            "face_uv": "float32 [num_nodes, 10, 10, 7]",
            "edge_uv": "float32 [num_edges, 10, 6]",
            "edge_index": "int64 [2, num_edges], DGL edge-ID order",
            "node_y": "int64 [num_nodes]",
        },
        "split_counts": {key: len(value) for key, value in splits.items()},
        "total_samples": len(tasks),
    }
    with (output_dir / "manifest.json").open("w") as write_file:
        json.dump(manifest, write_file, indent=2, sort_keys=True)
        write_file.write("\n")

    print(
        f"Converted: {counts['converted']}; skipped: {counts['skipped']}; "
        f"total: {len(tasks)}"
    )
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        "Convert official MFCAD DGL .bin graphs to framework-neutral NPZ files"
    )
    parser.add_argument("dataset_path", help="MFCAD root containing graph/, labels/, split.json")
    parser.add_argument("--output", help="Output directory (default: DATASET_PATH/neutral)")
    parser.add_argument("--workers", type=int, default=0, help="Parallel worker processes")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing NPZ samples")
    parser.add_argument(
        "--no_compress", action="store_true", help="Store arrays without ZIP compression"
    )
    convert(parser.parse_args())


if __name__ == "__main__":
    main()
