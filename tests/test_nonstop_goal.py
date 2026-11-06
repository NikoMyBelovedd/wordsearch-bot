"""Headless runs don't pace unless asked (owner 11/5): play and nonstop have no daily number."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from wsbot import cli
from wsbot.goal import Goal, Progress


class NonstopGoalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def goal(self, mode, per_day=None):
        with mock.patch.object(cli, "ROOT", self.root):
            return cli.make_goal(mode, per_day, "ios")

    def test_play_and_nonstop_have_no_pace(self):
        for mode in ("play", "nonstop"):
            g = self.goal(mode)
            self.assertEqual(g.plan_id, "nonstop")
            self.assertIsNone(g.per_day)
            self.assertIsNone(g.total)
            self.assertIsNone(g.schedule)  # no pacer: no play window, no breaks, no idling
            self.assertFalse(g.quota_reached_today())
            self.assertFalse(g.finished())

    def test_custom_still_paces(self):
        g = self.goal("custom", 600)
        self.assertEqual(g.per_day, 600)
        self.assertIsNotNone(g.schedule)

    def test_levels_still_stop_a_nonstop_run(self):
        g = Goal.nonstop(Progress(self.root / "p.json"))
        g.session_target = 2
        g.record_level()
        g.record_level()
        self.assertTrue(g.finished())


if __name__ == "__main__":
    unittest.main()
