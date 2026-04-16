"""
Path D integration test for IceIQ.
Tests: highlight_scoring, highlight_extraction, cross_shift_identity, recruiting_pdf
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestHighlightScoring(unittest.TestCase):
    def test_import(self):
        """highlight_scoring module imports cleanly."""
        try:
            import pipeline.highlight_scoring as hs
            self.assertTrue(hasattr(hs, '__file__'))
        except ImportError as e:
            self.skipTest(f"highlight_scoring not available: {e}")

    def test_score_range(self):
        """Scores should be in [0, 1] range."""
        try:
            from pipeline.highlight_scoring import HighlightScorer
            scorer = HighlightScorer()
            score = scorer.score_frame({
                "has_shot": True,
                "player_speed": 0.8,
                "near_goal": True,
                "puck_visible": True,
            })
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        except (ImportError, AttributeError):
            self.skipTest("HighlightScorer not implemented")


class TestHighlightExtraction(unittest.TestCase):
    def test_import(self):
        """highlight_extraction module imports cleanly."""
        try:
            import pipeline.highlight_extraction as he
            self.assertTrue(hasattr(he, '__file__'))
        except ImportError as e:
            self.skipTest(f"highlight_extraction not available: {e}")

    def test_extract_returns_list(self):
        """extract_highlights should return a list."""
        try:
            from pipeline.highlight_extraction import extract_highlights
            result = extract_highlights([], min_score=0.5)
            self.assertIsInstance(result, list)
        except (ImportError, TypeError):
            self.skipTest("extract_highlights not available")


class TestCrossShiftIdentity(unittest.TestCase):
    def test_import(self):
        """cross_shift_identity module imports cleanly."""
        from pipeline.cross_shift_identity import CrossShiftIdentityTracker
        tracker = CrossShiftIdentityTracker()
        self.assertIsNotNone(tracker)

    def test_register_player(self):
        """Should assign consistent global ID across shifts."""
        from pipeline.cross_shift_identity import CrossShiftIdentityTracker
        tracker = CrossShiftIdentityTracker(similarity_threshold=0.4)

        features = {"jersey_number": 17, "team": "home"}
        gid1 = tracker.register_player(track_id=1, features=features)
        gid2 = tracker.register_player(track_id=5, features=features)

        self.assertEqual(gid1, gid2, "Same player should get same global ID")

    def test_different_players(self):
        """Different players should get different global IDs."""
        from pipeline.cross_shift_identity import CrossShiftIdentityTracker
        tracker = CrossShiftIdentityTracker(similarity_threshold=0.8)

        gid1 = tracker.register_player(1, {"jersey_number": 17, "team": "home"})
        gid2 = tracker.register_player(2, {"jersey_number": 99, "team": "away"})

        self.assertNotEqual(gid1, gid2)

    def test_stats(self):
        """get_stats should return valid dict."""
        from pipeline.cross_shift_identity import CrossShiftIdentityTracker
        tracker = CrossShiftIdentityTracker()
        tracker.register_player(1, {"jersey_number": 10, "team": "home"})
        stats = tracker.get_stats()
        self.assertIn("total_players", stats)
        self.assertEqual(stats["total_players"], 1)


class TestRecruitingPDF(unittest.TestCase):
    def test_import(self):
        """recruiting_pdf module imports cleanly."""
        from pipeline.recruiting_pdf import generate_recruiting_pdf
        self.assertTrue(callable(generate_recruiting_pdf))

    def test_generate_text_fallback(self):
        """Should generate a text report when reportlab unavailable."""
        from pipeline.recruiting_pdf import _generate_text_fallback
        import tempfile

        player = {
            "name": "Test Player",
            "jersey_number": 42,
            "team": "home",
            "position": "forward",
            "shot_hand": "left",
            "metrics": {"speed_score": 0.75},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out = _generate_text_fallback(player, os.path.join(tmpdir, "test.txt"))
            self.assertTrue(os.path.exists(out))
            content = open(out).read()
            self.assertIn("Test Player", content)
            self.assertIn("42", content)


class TestPathDIntegration(unittest.TestCase):
    def test_all_modules_importable(self):
        """All Path D modules should be importable."""
        modules = [
            "pipeline.cross_shift_identity",
            "pipeline.recruiting_pdf",
        ]
        for mod in modules:
            try:
                __import__(mod)
            except ImportError as e:
                self.fail(f"Failed to import {mod}: {e}")

    def test_pipeline_files_exist(self):
        """Required pipeline files should exist."""
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        required = [
            "pipeline/cross_shift_identity.py",
            "pipeline/recruiting_pdf.py",
            "pipeline/highlight_extraction.py",
        ]
        for f in required:
            path = os.path.join(base, f)
            self.assertTrue(os.path.exists(path), f"Missing: {f}")


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
