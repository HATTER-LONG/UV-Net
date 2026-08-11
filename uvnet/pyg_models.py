"""Modern PyTorch/PyG UV-Net models and Lightning training module."""

import lightning as L
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

from uvnet.pyg_encoders import (
    UVNetCurveEncoder,
    UVNetGraphEncoder,
    UVNetSurfaceEncoder,
)


class _NonLinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, num_classes)
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, inputs):
        hidden = F.relu(self.bn1(self.linear1(inputs)))
        hidden = self.dp1(hidden)
        hidden = F.relu(self.bn2(self.linear2(hidden)))
        hidden = self.dp2(hidden)
        return self.linear3(hidden)


class UVNetSegmenter(nn.Module):
    def __init__(
        self,
        num_classes,
        crv_in_channels=6,
        crv_emb_dim=64,
        srf_emb_dim=64,
        graph_emb_dim=128,
        dropout=0.3,
    ):
        super().__init__()
        self.curv_encoder = UVNetCurveEncoder(crv_in_channels, crv_emb_dim)
        self.surf_encoder = UVNetSurfaceEncoder(7, srf_emb_dim)
        self.graph_encoder = UVNetGraphEncoder(
            srf_emb_dim, crv_emb_dim, graph_emb_dim
        )
        self.seg = _NonLinearClassifier(
            graph_emb_dim + srf_emb_dim, num_classes, dropout
        )

    def forward_features(self, data):
        curve_input = data.edge_uv.permute(0, 2, 1)
        surface_input = data.face_uv.permute(0, 3, 1, 2)
        curve_embedding = self.curv_encoder(curve_input)
        surface_embedding = self.surf_encoder(surface_input)
        node_embedding, graph_embedding = self.graph_encoder(
            data.edge_index,
            data.batch,
            surface_embedding,
            curve_embedding,
        )
        local_global = torch.cat(
            (node_embedding, graph_embedding[data.batch]), dim=1
        )
        return {
            "curve_embedding": curve_embedding,
            "surface_embedding": surface_embedding,
            "node_embedding": node_embedding,
            "graph_embedding": graph_embedding,
            "local_global": local_global,
        }

    def forward(self, data):
        return self.seg(self.forward_features(data)["local_global"])


def convert_dgl_state_dict(state_dict):
    """Map original DGL UVNetSegmenter parameter names to this PyG model."""
    converted = {}
    for key, value in state_dict.items():
        converted[key.replace(".gconv.edge_func.", ".gconv.nn.")] = value
    return converted


class SegmentationModule(L.LightningModule):
    def __init__(self, num_classes=16, crv_in_channels=6, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = UVNetSegmenter(num_classes, crv_in_channels=crv_in_channels)
        self.train_iou = MulticlassJaccardIndex(num_classes=num_classes, average="macro")
        self.val_iou = MulticlassJaccardIndex(num_classes=num_classes, average="macro")
        self.test_iou = MulticlassJaccardIndex(num_classes=num_classes, average="macro")
        self.train_accuracy = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.test_accuracy = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self._emit_test_predictions = False

    def forward(self, batch):
        return self.model(batch)

    def _shared_step(self, batch, stage):
        logits = self.model(batch)
        loss = F.cross_entropy(logits, batch.y, reduction="mean")
        batch_size = batch.y.numel()
        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=stage != "train",
            batch_size=batch_size,
            sync_dist=True,
        )
        iou = getattr(self, f"{stage}_iou")
        accuracy = getattr(self, f"{stage}_accuracy")
        iou(logits, batch.y)
        accuracy(logits, batch.y)
        self.log(
            f"{stage}_iou", iou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log(
            f"{stage}_accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        return loss, logits

    def training_step(self, batch, batch_idx):
        loss, _ = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        loss, logits = self._shared_step(batch, "test")
        if not self._emit_test_predictions:
            return loss

        probabilities = logits.softmax(dim=1)
        confidence, prediction = probabilities.max(dim=1)
        return {
            "prediction": prediction.detach().cpu(),
            "confidence": confidence.detach().cpu(),
            "target": batch.y.detach().cpu(),
            "face_to_sample": batch.batch.detach().cpu(),
            "sample_ids": list(batch.sample_id),
        }

    def enable_test_predictions(self, enabled=True):
        """Expose per-face predictions to test callbacks without a second forward pass."""
        self._emit_test_predictions = enabled

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
