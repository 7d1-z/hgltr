import numpy as np
from collections import OrderedDict
import torch
from loguru import logger


class Evaluator:
    """Evaluator for classification."""

    def __init__(self, many_idx=None, med_idx=None, few_idx=None, fmt=".3f"):
        self.many_idx = many_idx
        self.med_idx = med_idx
        self.few_idx = few_idx
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.correct = 0
        self.correct_top5 = 0
        self.total = 0
        self.y_true = []
        self.y_pred = []
        self.y_conf = []

    def _compute_topk(self, pred: torch.Tensor, labels: torch.LongTensor, k: int = 5):
        topk = torch.topk(pred, k=k, dim=1).indices  # (batch, k)
        match_matrix = topk.eq(labels.unsqueeze(1))  # (batch, k)
        return match_matrix.any(dim=1).float()       # (batch,)

    def process(self, pred: torch.Tensor, labels: torch.LongTensor):
        """
        pred: [batch, num_classes]
        labels: [batch]
        """
        pred = pred.detach()
        labels = labels.detach()

        conf, top1_pred = torch.softmax(pred, dim=1).max(1)

        top1_matches = top1_pred.eq(labels).float()
        top5_matches = self._compute_topk(pred, labels, k=5)

        self.correct += int(top1_matches.sum().item())
        self.correct_top5 += int(top5_matches.sum().item())
        self.total += labels.size(0)

        self.y_true.extend(labels.cpu().numpy())
        self.y_pred.extend(top1_pred.cpu().numpy())
        self.y_conf.extend(conf.cpu().numpy())

    def _compute_per_class_accuracy(self):
        label_arr = np.array(self.y_true)
        pred_arr = np.array(self.y_pred)

        cls_num = label_arr.max() + 1  # class [0, cls_num-1]
        correct = np.zeros(cls_num, dtype=np.int32)
        total = np.zeros(cls_num, dtype=np.int32)

        np.add.at(correct, label_arr, label_arr == pred_arr)
        np.add.at(total, label_arr, 1)

        with np.errstate(divide='ignore', invalid='ignore'):
            cls_acc = np.true_divide(correct, total)
            cls_acc[np.isnan(cls_acc)] = 0.0

        return cls_acc

    def evaluate(self):
        results = OrderedDict()
        acc = 100 * self.correct / self.total
        acc_top5 = 100 * self.correct_top5 / self.total

        results["accuracy"] = acc
        results["top5_accuracy"] = acc_top5
        logger.info(f"total: {self.total}, correct: {self.correct}, "
                    f"accuracy: {acc:{self.fmt}}%, top5_accuracy: {acc_top5:{self.fmt}}%")

        cls_acc = self._compute_per_class_accuracy()

        if len(cls_acc) > 0:
            results["worst_case_acc"] = float(cls_acc.min()) * 100
            logger.info(f"worst case acc: {cls_acc.min():{self.fmt}}%")
        else:
            logger.info("worst case acc: 0.0")

        if self.many_idx is not None:
            many_acc = np.mean(cls_acc[self.many_idx]) if len(self.many_idx) > 0 else 0.0
            results["many_acc"] = many_acc * 100

        if self.med_idx is not None:
            med_acc = np.mean(cls_acc[self.med_idx]) if len(self.med_idx) > 0 else 0.0
            results["med_acc"] = med_acc  * 100

        if self.few_idx is not None:
            few_acc = np.mean(cls_acc[self.few_idx]) if len(self.few_idx) > 0 else 0.0
            results["few_acc"] = few_acc * 100

        if self.many_idx is not None and self.med_idx is not None and self.few_idx is not None:
            logger.info(f"Many: {results['many_acc']:{self.fmt}}%, "
                        f"Med: {results['med_acc']:{self.fmt}}%, "
                        f"Few: {results['few_acc']:{self.fmt}}%")

        return results
