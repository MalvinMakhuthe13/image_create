from __future__ import annotations

import cv2

from app.engine.matching.models import DescriptorMatch, ImageFeatures


def match_descriptors(left: ImageFeatures, right: ImageFeatures, *,
                      ratio_threshold: float = 0.75, max_matches: int = 4000) -> list[DescriptorMatch]:
    if not 0.5 <= ratio_threshold < 1.0:
        raise ValueError("ratio_threshold must be in [0.5, 1.0)")
    if max_matches < 8:
        raise ValueError("max_matches must be >= 8")
    if left.descriptors is None or right.descriptors is None:
        return []
    if len(left.descriptors) < 2 or len(right.descriptors) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    forward = matcher.knnMatch(left.descriptors, right.descriptors, k=2)
    reverse = matcher.knnMatch(right.descriptors, left.descriptors, k=2)
    def ratio_pass(rows):
        passed = {}
        for pair in rows:
            if len(pair) == 2 and pair[0].distance < ratio_threshold * pair[1].distance:
                passed[pair[0].queryIdx] = pair[0]
        return passed
    f_good, r_good = ratio_pass(forward), ratio_pass(reverse)
    mutual = [DescriptorMatch(m.queryIdx, m.trainIdx, float(m.distance))
              for m in f_good.values()
              if m.trainIdx in r_good and r_good[m.trainIdx].trainIdx == m.queryIdx]
    mutual.sort(key=lambda item: item.distance)
    return mutual[:max_matches]
