"""Train and test the modern PyG implementation of UV-Net on MFCAD."""

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys

import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import torch
from torch_geometric.loader import DataLoader

from datasets.pyg_mfcad import MFCADPyGDataset
from uvnet.pyg_models import SegmentationModule


class PredictionReportCallback(Callback):
    """Print readable per-face prediction/ground-truth comparisons during test."""

    def __init__(self, max_samples, errors_only=False):
        self.max_samples = max_samples
        self.errors_only = errors_only
        self.samples_printed = 0
        self.total_faces = 0
        self.total_correct = 0
        self.confidence_sum = 0.0
        self.use_color = sys.stdout.isatty() and "NO_COLOR" not in os.environ

    def _color(self, text, code):
        if not self.use_color:
            return text
        return f"\033[{code}m{text}\033[0m"

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if not isinstance(outputs, dict):
            return

        prediction = outputs["prediction"]
        confidence = outputs["confidence"]
        target = outputs["target"]
        face_to_sample = outputs["face_to_sample"]
        sample_ids = outputs["sample_ids"]

        correct = prediction.eq(target)
        self.total_faces += target.numel()
        self.total_correct += int(correct.sum())
        self.confidence_sum += float(confidence.sum())

        for sample_index, sample_id in enumerate(sample_ids):
            if self.samples_printed >= self.max_samples:
                break
            mask = face_to_sample.eq(sample_index)
            sample_prediction = prediction[mask]
            sample_confidence = confidence[mask]
            sample_target = target[mask]
            sample_correct = sample_prediction.eq(sample_target)
            num_faces = sample_target.numel()
            num_correct = int(sample_correct.sum())
            if self.errors_only and num_correct == num_faces:
                continue

            title = f" CAD sample {self.samples_printed + 1}/{self.max_samples}: {sample_id} "
            print("\n" + self._color(f"┌{'─' * 3}{title}{'─' * 3}", "1;36"))
            print(
                "│ Faces: "
                f"{num_faces}  Correct: {num_correct}  Errors: {num_faces - num_correct}  "
                f"Face accuracy: {100.0 * num_correct / max(num_faces, 1):.2f}%"
            )
            print("├────────┬──────────────┬──────────────┬────────────┬────────")
            print("│ Face   │ Ground truth │ Prediction   │ Confidence │ Status")
            print("├────────┼──────────────┼──────────────┼────────────┼────────")

            for face_index, (truth, predicted, conf, is_correct) in enumerate(
                zip(sample_target, sample_prediction, sample_confidence, sample_correct)
            ):
                if self.errors_only and bool(is_correct):
                    continue
                status = (
                    self._color("✓ match", "32")
                    if bool(is_correct)
                    else self._color("✗ error", "1;31")
                )
                print(
                    f"│ {face_index:>6} │ class {int(truth):>6} │ class {int(predicted):>6} │"
                    f" {100.0 * float(conf):>9.2f}% │ {status}"
                )
            print("└────────┴──────────────┴──────────────┴────────────┴────────")
            self.samples_printed += 1

    def on_test_epoch_end(self, trainer, pl_module):
        if self.total_faces == 0:
            return
        accuracy = 100.0 * self.total_correct / self.total_faces
        mean_confidence = 100.0 * self.confidence_sum / self.total_faces
        print(self._color("\nDetailed prediction report summary", "1;36"))
        print(f"  Samples shown : {self.samples_printed}")
        print(f"  Faces evaluated: {self.total_faces}")
        print(f"  Correct / error: {self.total_correct} / {self.total_faces - self.total_correct}")
        print(f"  Face accuracy : {accuracy:.4f}%")
        print(f"  Mean confidence: {mean_confidence:.4f}%")


def build_loader(dataset, args, shuffle, drop_last):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=torch.cuda.is_available() and args.accelerator != "cpu",
    )


def build_trainer(args, run_dir, training):
    callbacks = []
    if training:
        callbacks.append(
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                dirpath=run_dir,
                filename="best",
                save_last=True,
                save_top_k=1,
            )
        )
    if not training and args.verbose_predictions:
        callbacks.append(
            PredictionReportCallback(
                max_samples=args.prediction_samples,
                errors_only=args.prediction_errors_only,
            )
        )
    logger = TensorBoardLogger(
        save_dir=str(run_dir.parent.parent),
        name=run_dir.parent.name,
        version=run_dir.name,
    )
    trainer_options = {
        "max_epochs": args.max_epochs,
        "accelerator": args.accelerator,
        "devices": args.devices,
        "precision": args.precision,
        "deterministic": args.deterministic,
        "callbacks": callbacks,
        "logger": logger,
        "log_every_n_steps": args.log_every_n_steps,
    }
    if args.limit_train_batches is not None:
        trainer_options["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        trainer_options["limit_val_batches"] = args.limit_val_batches
    if args.limit_test_batches is not None:
        trainer_options["limit_test_batches"] = args.limit_test_batches
    return L.Trainer(**trainer_options)


def main():
    parser = argparse.ArgumentParser("Modern PyG UV-Net face segmentation")
    parser.add_argument("traintest", choices=("train", "test"))
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--experiment_name", default="segmentation-pyg")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--random_rotate", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", default="32-true")
    parser.add_argument(
        "--matmul_precision",
        choices=("highest", "high", "medium"),
        default="high",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--limit_train_batches", type=int)
    parser.add_argument("--limit_val_batches", type=int)
    parser.add_argument("--limit_test_batches", type=int)
    parser.add_argument(
        "--verbose_predictions",
        action="store_true",
        help="Print per-face predictions, labels, confidence, and match status during test",
    )
    parser.add_argument(
        "--prediction_samples",
        type=int,
        default=5,
        help="Maximum number of CAD samples shown by --verbose_predictions (default: 5)",
    )
    parser.add_argument(
        "--prediction_errors_only",
        action="store_true",
        help="With --verbose_predictions, show only misclassified faces",
    )
    args = parser.parse_args()

    if args.prediction_samples < 1:
        parser.error("--prediction_samples must be at least 1")
    if args.prediction_errors_only and not args.verbose_predictions:
        parser.error("--prediction_errors_only requires --verbose_predictions")

    torch.set_float32_matmul_precision(args.matmul_precision)
    L.seed_everything(args.seed, workers=True)
    now = datetime.now()
    run_dir = (
        Path(__file__).parent
        / "results"
        / args.experiment_name
        / now.strftime("%m%d")
        / now.strftime("%H%M%S")
    )
    trainer = build_trainer(args, run_dir, training=args.traintest == "train")

    if args.traintest == "train":
        train_dataset = MFCADPyGDataset(
            args.dataset_path,
            split="train",
            random_rotate=args.random_rotate,
        )
        val_dataset = MFCADPyGDataset(args.dataset_path, split="validation")
        model = SegmentationModule(learning_rate=args.learning_rate)
        trainer.fit(
            model,
            train_dataloaders=build_loader(train_dataset, args, True, True),
            val_dataloaders=build_loader(val_dataset, args, False, False),
        )
        print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
        return

    if args.checkpoint is None:
        parser.error("--checkpoint is required for test")
    test_dataset = MFCADPyGDataset(args.dataset_path, split="test")
    model = SegmentationModule.load_from_checkpoint(args.checkpoint)
    model.enable_test_predictions(args.verbose_predictions)
    results = trainer.test(
        model=model,
        dataloaders=build_loader(test_dataset, args, False, False),
        verbose=False,
    )
    print(f"Segmentation IoU (%) on test set: {results[0]['test_iou'] * 100.0}")


if __name__ == "__main__":
    main()
