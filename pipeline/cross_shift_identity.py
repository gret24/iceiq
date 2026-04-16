"""
Cross-shift player identity tracking for IceIQ.
Maintains player identity across line changes and shifts.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class CrossShiftIdentityTracker:
    """
    Tracks player identity across shifts using jersey number,
    team color, and physical appearance features.
    """

    def __init__(self, similarity_threshold: float = 0.75):
        self.similarity_threshold = similarity_threshold
        self.player_registry: Dict[int, dict] = {}
        self.shift_history: List[dict] = []
        self.next_global_id = 1

    def register_player(self, track_id: int, features: dict) -> int:
        """Register or re-identify a player, returning global ID."""
        best_match_id = None
        best_score = 0.0

        for global_id, profile in self.player_registry.items():
            score = self._compute_similarity(features, profile["features"])
            if score > best_score:
                best_score = score
                best_match_id = global_id

        if best_score >= self.similarity_threshold and best_match_id is not None:
            self.player_registry[best_match_id]["track_ids"].append(track_id)
            self.player_registry[best_match_id]["features"] = self._merge_features(
                self.player_registry[best_match_id]["features"], features
            )
            return best_match_id
        else:
            global_id = self.next_global_id
            self.next_global_id += 1
            self.player_registry[global_id] = {
                "track_ids": [track_id],
                "features": features,
            }
            return global_id

    def _compute_similarity(self, f1: dict, f2: dict) -> float:
        """Compute similarity score between two feature sets."""
        score = 0.0
        weight_sum = 0.0

        # Jersey number match (high weight)
        if f1.get("jersey_number") and f2.get("jersey_number"):
            w = 0.5
            score += w * (1.0 if f1["jersey_number"] == f2["jersey_number"] else 0.0)
            weight_sum += w

        # Team match
        if f1.get("team") and f2.get("team"):
            w = 0.3
            score += w * (1.0 if f1["team"] == f2["team"] else 0.0)
            weight_sum += w

        # Appearance embedding cosine similarity
        if f1.get("embedding") is not None and f2.get("embedding") is not None:
            e1 = np.array(f1["embedding"])
            e2 = np.array(f2["embedding"])
            norm1, norm2 = np.linalg.norm(e1), np.linalg.norm(e2)
            if norm1 > 0 and norm2 > 0:
                w = 0.2
                cos_sim = float(np.dot(e1, e2) / (norm1 * norm2))
                score += w * max(0.0, cos_sim)
                weight_sum += w

        return score / weight_sum if weight_sum > 0 else 0.0

    def _merge_features(self, existing: dict, new_features: dict) -> dict:
        """Merge new features into existing profile using exponential moving average."""
        merged = dict(existing)
        alpha = 0.3  # weight for new features

        if new_features.get("embedding") is not None and existing.get("embedding") is not None:
            e_old = np.array(existing["embedding"])
            e_new = np.array(new_features["embedding"])
            merged["embedding"] = ((1 - alpha) * e_old + alpha * e_new).tolist()

        if new_features.get("jersey_number") and not existing.get("jersey_number"):
            merged["jersey_number"] = new_features["jersey_number"]

        return merged

    def get_global_id(self, track_id: int) -> Optional[int]:
        """Get global player ID for a given tracker ID."""
        for global_id, profile in self.player_registry.items():
            if track_id in profile["track_ids"]:
                return global_id
        return None

    def get_stats(self) -> dict:
        return {
            "total_players": len(self.player_registry),
            "total_track_ids": sum(len(p["track_ids"]) for p in self.player_registry.values()),
        }
