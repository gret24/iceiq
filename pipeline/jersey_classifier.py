"""
pipeline/jersey_classifier.py

Closed-set jersey number classifier.
학습된 MobileNetV3-small 모델로 OCR 대신 번호 분류.

팀 분류가 먼저 확정된 후 해당 팀 모델을 호출:
    from pipeline.jersey_classifier import JerseyClassifier
    clf = JerseyClassifier()
    number, conf = clf.predict(crop_bgr, team='aigis')
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

logger = logging.getLogger(__name__)

_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')

_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def _load_model(path: str):
    ckpt = torch.load(path, map_location=_DEVICE, weights_only=False)
    model = models.mobilenet_v3_small()
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, ckpt['n_classes'])
    model.load_state_dict(ckpt['model_state'])
    model = model.to(_DEVICE).eval()
    return model, ckpt['idx_to_num']


def _preprocess(img_bgr: np.ndarray, size: int = 96) -> np.ndarray:
    """상체 중앙 torso 영역만 크롭 후 리사이즈"""
    h, w = img_bgr.shape[:2]
    y1, y2 = int(h * 0.28), int(h * 0.68)
    x1, x2 = int(w * 0.12), int(w * 0.88)
    torso = img_bgr[y1:y2, x1:x2]
    if torso.size == 0:
        torso = img_bgr
    return cv2.resize(torso, (size, size), interpolation=cv2.INTER_CUBIC)


class JerseyClassifier:
    """
    팀별 closed-set jersey number classifier.

    Parameters
    ----------
    model_dir : 모델 파일 디렉토리 (default: ../models)
    min_conf  : 예측 최소 신뢰도 (default: 0.6)
    """

    def __init__(self, model_dir: Optional[str] = None, min_conf: float = 0.6):
        self.min_conf = min_conf
        mdir = model_dir or _MODEL_DIR
        self._models: dict = {}

        for team in ('aigis', 'lopez'):
            path = os.path.join(mdir, f'jersey_classifier_{team}.pt')
            if os.path.exists(path):
                try:
                    model, idx_to_num = _load_model(path)
                    self._models[team] = (model, idx_to_num)
                    logger.info('JerseyClassifier: loaded %s (%d classes)',
                                team, len(idx_to_num))
                except Exception as e:
                    logger.warning('JerseyClassifier: failed to load %s — %s', team, e)
            else:
                logger.warning('JerseyClassifier: model not found — %s', path)

    @property
    def available_teams(self):
        return list(self._models.keys())

    def predict(self, crop_bgr: np.ndarray, team: str) -> Tuple[Optional[int], float]:
        """
        Parameters
        ----------
        crop_bgr : 선수 crop 이미지 (BGR)
        team     : 'aigis' or 'lopez'

        Returns
        -------
        (jersey_number, confidence) or (None, 0.0) if unavailable/low-conf
        """
        if team not in self._models:
            return None, 0.0

        model, idx_to_num = self._models[team]

        try:
            torso = _preprocess(crop_bgr)
            img_rgb = cv2.cvtColor(torso, cv2.COLOR_BGR2RGB)
            x = _TRANSFORM(img_rgb).unsqueeze(0).to(_DEVICE)

            with torch.no_grad():
                out = model(x)
                probs = torch.softmax(out, dim=1)[0]
                idx = probs.argmax().item()
                conf = probs[idx].item()

            if conf < self.min_conf:
                return None, conf

            return idx_to_num[idx], conf

        except Exception as e:
            logger.debug('JerseyClassifier.predict error: %s', e)
            return None, 0.0

    def predict_top_k(self, crop_bgr: np.ndarray, team: str, k: int = 3):
        """상위 k개 후보 반환 — [(number, conf), ...]"""
        if team not in self._models:
            return []
        model, idx_to_num = self._models[team]
        try:
            torso = _preprocess(crop_bgr)
            img_rgb = cv2.cvtColor(torso, cv2.COLOR_BGR2RGB)
            x = _TRANSFORM(img_rgb).unsqueeze(0).to(_DEVICE)
            with torch.no_grad():
                out = model(x)
                probs = torch.softmax(out, dim=1)[0]
            topk = probs.topk(min(k, len(idx_to_num)))
            return [(idx_to_num[i.item()], p.item()) for i, p in zip(topk.indices, topk.values)]
        except Exception as e:
            logger.debug('predict_top_k error: %s', e)
            return []
