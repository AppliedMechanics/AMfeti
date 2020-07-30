"""Basic integration test for examples"""

from unittest import TestCase


class ExampleTestBase(TestCase):
    def setUp(self):
        self.run_parallel_tests = False