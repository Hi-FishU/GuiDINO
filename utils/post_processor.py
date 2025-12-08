import torch
import torch.nn as nn
from utils.criterion import rescale_bboxes, box_cxcywh_to_xyxy


class DETRPostProcessor(nn.Module):
    """
    Convert DETR outputs to per-image boxes & scores in absolute XYXY (pixels).
    """

    def __init__(self, score_threshold=0.0):
        super().__init__()
        self.score_threshold = score_threshold

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        prob = out_logits.softmax(-1)
        scores, labels = prob[..., :-1].max(-1)  # ignore no-object

        results = []
        for i, (scores_i, labels_i, boxes_i) in enumerate(zip(scores, labels, out_bbox)):
            h, w = target_sizes[i]
            boxes_abs = rescale_bboxes(boxes_i, (h, w))
            if self.score_threshold > 0:
                keep = scores_i > self.score_threshold
                boxes_abs = boxes_abs[keep]
                labels_i = labels_i[keep]
                scores_i = scores_i[keep]
            results.append({
                "scores": scores_i,
                "labels": labels_i,
                "boxes": boxes_abs
            })
        return results
