# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 15:18:00 2021

@author: eugen
"""

from context import scai

import unittest

class TestExamples(unittest.TestCase):

    def test_upper(self):
        scai.scai_Example2()
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()