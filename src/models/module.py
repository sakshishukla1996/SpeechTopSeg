from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torcheval.metrics.functional import multiclass_f1_score
from src.utils import get_seg_boundaries, pk, win_diff


class LitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=2)
        self.val_acc = Accuracy(task="multiclass", num_classes=2)
        self.test_acc = Accuracy(task="multiclass", num_classes=2)
        self.prev_batch = None

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch   # (1,sent,embb), (sent)
        b,s,e = x.shape
        x = x.view(-1,1,e)
        # breakpoint()
        logits = self.forward(x)
        logits=logits.squeeze(0)
        # breakpoint()
        y=y.squeeze(0)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        preds[-1]=1
        return loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        f1_score =  multiclass_f1_score(preds, targets, num_classes=2, average="macro")
        if f1_score>0.75:
            pred_boundaries = get_seg_boundaries(classifications=preds)
            target_boundaries = get_seg_boundaries(classifications=targets)
            train_pk, train_count_pk = pk(pred_boundaries, target_boundaries)
            train_windiff, train_count_windiff = win_diff(pred_boundaries, target_boundaries)
            self.log("train/pk", train_pk, on_step=True, on_epoch=False, prog_bar=True)
            self.log("train/windiff", train_windiff, on_step=True, on_epoch=False, prog_bar=True)
        

        self.log("train/f1", f1_score, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        # print("INTO VAL_STEP!")
        # breakpoint()
        loss, preds, targets = self.model_step(batch)
        pred_boundaries = get_seg_boundaries(classifications=preds)
        target_boundaries = get_seg_boundaries(classifications=targets)
        val_pk, val_count_pk = pk(pred_boundaries, target_boundaries)
        val_windiff, val_count_windiff = win_diff(pred_boundaries, target_boundaries)
        self.log("val/pk", val_pk, on_step=True, on_epoch=False, prog_bar=True)
        self.log("val/windiff", val_windiff, on_step=True, on_epoch=False, prog_bar=True)
        self.prev_batch = {"batch": batch, "preds": pred_boundaries, "targets": target_boundaries}
        print(f"Predictions: \n\t{self.prev_batch['targets']} \n\t{self.prev_batch['preds']}")
        print("="*80)
        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        f1_score =  multiclass_f1_score(preds, targets, num_classes=2, average="macro")
        self.log("val/f1", f1_score, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    # def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        # """Perform a single test step on a batch of data from the test set.

        # :param batch: A batch of data (a tuple) containing the input tensor of images and target
        #     labels.
        # :param batch_idx: The index of the current batch.
        # """
        # loss, preds, targets = self.model_step(batch)

        # # update and log metrics
        # self.test_loss(loss)
        # self.test_acc(preds, targets)
        # self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    # def on_test_epoch_end(self) -> None:
    #     """Lightning hook that is called when a test epoch ends."""
    #     pass

    def setup(self, stage: str) -> None:
        """Latest method torch.compile to speedup on a100 machines. This gives about 1.9x to 2.1x boost. 
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/acc",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}