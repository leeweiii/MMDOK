import os
import torch
import json

from .. import builder
import numpy as np
import torch.nn as nn

from sklearn.metrics import average_precision_score, roc_auc_score
from pytorch_lightning.core import LightningModule
from sklearn.metrics import roc_auc_score


class ClassificationModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.mglrl = builder.build_model(cfg)
        self.lr = cfg.lightning.trainer.lr
        self.dm = None

    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.mglrl)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def training_epoch_end(self, training_step_outputs):
        return self.shared_epoch_end(training_step_outputs, "train")

    def validation_epoch_end(self, validation_step_outputs):
        return self.shared_epoch_end(validation_step_outputs, "val")

    def test_epoch_end(self, test_step_outputs):
        return self.shared_epoch_end(test_step_outputs, "test")

    def shared_step(self, batch, split):
        """Similar to traning step"""
        # pred, labels = self.mglrl(batch)
        pred, labels, loss_pred, loss_proto = self.mglrl(batch)

        loss = 0.6 * loss_pred + 0.4 * loss_proto

        # log training progress
        log_iter_loss = True if split == "train" else False
        self.log(
            f"{split}_loss",
            loss,
            on_epoch=True,
            on_step=log_iter_loss,
            logger=True,
            prog_bar=True,
        )
        # pred_np = pred.detach().cpu().numpy()
        # labels_np = labels.detach().cpu().numpy()
        # auroc = roc_auc_score(labels_np, pred_np, average='macro', multi_class='ovr')
        # self.log(f"{split}_auroc", auroc, on_step=True, on_epoch=True, prog_bar=True)
        #
        return_dict = {"loss": loss, "pred": pred, "labels": labels}
        return return_dict

    def shared_epoch_end(self, step_outputs, split):
        logit = torch.cat([x["pred"] for x in step_outputs])
        y = torch.cat([x["labels"] for x in step_outputs])
        prob = torch.sigmoid(logit)

        y = y.detach().cpu().numpy()
        prob = prob.detach().cpu().numpy()

        auroc_list, auprc_list = [], []
        for i in range(y.shape[0]):
            y_cls = y[i, :]
            prob_cls = prob[i, :]

            if np.isnan(prob_cls).any():
                auprc_list.append(0)
                auroc_list.append(0)
            else:
                auprc_list.append(average_precision_score(y_cls, prob_cls))
                auroc_list.append(roc_auc_score(y_cls, prob_cls))

        auprc = np.mean(auprc_list)
        auroc = np.mean(auroc_list)

        self.log(f"{split}_auroc", auroc, on_epoch=True, logger=True, prog_bar=True)
        self.log(f"{split}_auprc", auprc, on_epoch=True, logger=True, prog_bar=True)

        if split == "test":
            results_csv = os.path.join(self.cfg.output_dir, "results.csv")
            results = {"auorc": auroc, "auprc": auprc}
            with open(results_csv, "w") as fp:
                json.dump(results, fp)






