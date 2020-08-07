import time
from unittest import TestCase

from run import _timerfunc


class TestRun(TestCase):

    def test_timerfunc(self):
        def sleep(sec):
            time.sleep(sec)
        _, got = _timerfunc(sleep, [1])
        self.assertEqual(int(got), 1)
