import os
import sys
import unittest

sys.path.insert(0, os.environ.get("PTDBENCH_WORKSPACE", "/workspace"))

from toy_swe.slugify import slugify


class SlugifyTests(unittest.TestCase):
    def test_basic_spaces(self):
        self.assertEqual(slugify("Hello World"), "hello-world")

    def test_punctuation_collapses(self):
        self.assertEqual(slugify("  Hello__World!!  "), "hello-world")

    def test_multiple_separators(self):
        self.assertEqual(slugify("Already--slug"), "already-slug")

    def test_mixed_tokens(self):
        self.assertEqual(slugify("Mix_of Things"), "mix-of-things")

    def test_empty_input(self):
        self.assertEqual(slugify("   "), "")


if __name__ == "__main__":
    unittest.main()
