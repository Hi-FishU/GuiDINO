import torch
from torch import nn
from torchvision.ops.boxes import box_area
from torchvision.ops import generalized_box_iou
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F


def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    b = [(cx - 0.5 * w), (cy - 0.5 * h),
         (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)], dim=-1)


def rescale_bboxes(out_bbox, size):
    # out_bbox in [0,1] cxcywh → to absolute XYXY in pixels
    img_h, img_w = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], device=b.device)
    return b


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 5, cost_giou: float = 2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "All costs can't be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        outputs: dict with 'pred_logits': (B, Q, K+1), 'pred_boxes': (B, Q, 4 in [0,1])
        targets: list of dicts with 'labels': (Ni,), 'boxes': (Ni, 4) ABSOLUTE XYXY in pixels & 'orig_size': (H,W)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].softmax(-1)  # (B, Q, K+1)
        # (B, Q, 4) normalized cxcywh
        out_bbox = outputs["pred_boxes"]

        indices = []
        for b in range(bs):
            tgt_ids = targets[b]["labels"]
            tgt_bbox_abs = targets[b]["boxes"]  # absolute XYXY
            H, W = targets[b]["orig_size"]
            # Normalize target boxes to [0,1] cxcywh for matching
            tgt_bbox = box_xyxy_to_cxcywh(
                tgt_bbox_abs / torch.tensor([W, H, W, H], device=tgt_bbox_abs.device))
            tgt_bbox = tgt_bbox.clamp(0, 1)

            # Classification cost: we take probability of the target class
            cost_class = -out_prob[b][:, tgt_ids]  # (Q, Nt)

            # bbox L1 cost
            cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)

            # giou cost
            out_xyxy = box_cxcywh_to_xyxy(out_bbox[b])
            tgt_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
            cost_giou = -generalized_box_iou(out_xyxy, tgt_xyxy)

            C = self.cost_class * cost_class + self.cost_bbox * \
                cost_bbox + self.cost_giou * cost_giou
            C = C.cpu()

            q_idx, t_idx = linear_sum_assignment(C)
            indices.append((torch.as_tensor(q_idx, dtype=torch.int64),
                           torch.as_tensor(t_idx, dtype=torch.int64)))

        return indices


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """
        weight_dict: e.g., {"loss_ce": 1, "loss_bbox": 5, "loss_giou": 2, "loss_ce_0": 1, ...} for aux losses
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        src_logits = outputs['pred_logits']  # (B, Q, K+1)

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J]
                                     for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(
            1, 2), target_classes, self.empty_weight)
        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]  # normalized cxcywh [0,1]

        target_boxes = torch.cat([
            box_xyxy_to_cxcywh(t['boxes'][i] / torch.tensor([t['orig_size'][1], t['orig_size']
                               [0], t['orig_size'][1], t['orig_size'][0]], device=src_boxes.device))
            for t, (_, i) in zip(targets, indices)
        ]).clamp(0, 1)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)
        )).sum() / num_boxes

        return {'loss_bbox': loss_bbox, 'loss_giou': loss_giou}

    def _get_src_permutation_idx(self, indices):
        # Permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i)
                              for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return (batch_idx, src_idx)

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i)
                              for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return (batch_idx, tgt_idx)

    def forward(self, outputs, targets):
        """
        targets: list of dicts with:
          - 'labels': (N_i,) int64 in [0, num_classes-1]
          - 'boxes': (N_i, 4) ABSOLUTE XYXY in pixels
          - 'orig_size': (H, W)
        """
        # remove aux for matching
        outputs_no_aux = {k: v for k,
                          v in outputs.items() if k != 'aux_outputs'}

        indices = self.matcher(outputs_no_aux, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1.0).item()

        losses = {}
        for loss in self.losses:
            losses.update(getattr(self, f'loss_{loss}')(
                outputs, targets, indices, num_boxes))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = getattr(self, f'loss_{loss}')(
                        aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # apply weights
        weighted = {k: v * self.weight_dict.get(k, 1.0)
                    for k, v in losses.items()}
        return weighted


class SingleObjectAccuracy(nn.Module):
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, outputs, targets):
        """
        outputs: list of dicts with 'scores': (Ni,), 'labels': (Ni,), 'boxes': (Ni, 4) ABSOLUTE XYXY in pixels
        targets: list of dicts with 'labels': (1,), 'boxes': (1, 4) ABSOLUTE XYXY in pixels
        """
        for output, target in zip(outputs, targets):
            if len(output['labels']) == 0:
                pred_label = -1  # No prediction
            else:
                top_idx = output['scores'].argmax()
                pred_label = output['labels'][top_idx].item()
            true_label = target['labels'][0].item()
            if pred_label == true_label:
                self.correct += 1
            self.total += 1

    def compute(self):
        return self.correct / self.total if self.total > 0 else 0.0