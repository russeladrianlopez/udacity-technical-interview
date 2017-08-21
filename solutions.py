#!/usr/bin/env python

import unittest  # testing purposes

"""
Question 1: Given two strings s and t, determine whether some anagram of
t is a substring of s. For example: if s = "udacity" and t = "ad", then
the function returns True. Your function definition should look like:
question1(s, t) and return a boolean True or False.
"""

# Assess if s1 and s2 are anagram of each other
# return True if strings are anagram of each other else False
def is_anagram(s1, s2):
    s1 = list(s1)
    s2 = list(s2)

    s1.sort()  # QuickSort O(n*log(n))
    s2.sort()  # QuickSort O(n*log(n))
    return s1 == s2


# Passdown the possible substring of s and t to is_anagram function.
# return False if s or t doesnt not exist,
# True if is_anagram() returns True, else False
def question1(s, t):
    if s and t:
        substring_length = len(t)
        string_length = len(s)

        for i in range(string_length - substring_length + 1):
            if is_anagram(s[i: i + substring_length], t):
                return True
    return False


class TestQuestion1(unittest.TestCase):
    print '\nTESTCASES FOR QUESTION #1: \n'

    def test_isanagram(self):
        self.assertTrue(question1('udacity', 'ad'))
        self.assertTrue(question1('udacity', 'da'))

    def test_notanagram(self):
        self.assertFalse(question1('udacity', 'xyz'))
        self.assertFalse(question1('udacity', 'act'))
        self.assertFalse(question1('uda', 'udacity'))

    def test_emptystring(self):
        self.assertFalse(question1('', 'ab'))
        self.assertFalse(question1('udacity', ''))
        with self.assertRaises(TypeError):
            question1('udacity', 1)
        with self.assertRaises(TypeError):
            (question1(True, 'ab'))


suite = unittest.TestLoader().loadTestsFromTestCase(TestQuestion1)
unittest.TextTestRunner(verbosity=2).run(suite)

"""
Question 2: Given a string a, find the longest palindromic substring
contained in a. Your function definition should look like question2(a),
and return a string.
"""

"""
Question 3: Given an undirected graph G, find the minimum spanning tree
within G. A minimum spanning tree connects all vertices in a graph with the
smallest possible total weight of edges.

Your function should take in and return an adjacency list structured like this:
{'A': [('B', 2)],
 'B': [('A', 2), ('C', 5)],
 'C': [('B', 5)]}

Vertices are represented as unique strings.
The function definition should be question3(G)
"""

"""
Question 4: Find the least common ancestor between two nodes on a binary
search tree. The least common ancestor is the farthest node from the root that
is an ancestor of both nodes. For example, the root is a common ancestor of
all nodes on the tree, but if both nodes are descendents of the root's left
child, then that left child might be the lowest common ancestor. You can assume
that both nodes are in the tree, and the tree itself adheres to all BST
properties. The function definition should look like question4(T, r, n1, n2),
where T is the tree represented as a matrix, where the index of the list is
equal to the integer stored in that node and a 1 represents a child node, r is
a non-negative integer representing the root, and n1 and n2 are non-negative
integers representing the two nodes in no particular order. For example, one
test case might be ...

question4([[0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [1, 0, 0, 0, 1],
           [0, 0, 0, 0, 0]],
          3,
          1,
          4)

and the answer would be 3.
"""

"""
Question 5: Find the element in a singly linked list that's m elements
from the end. For example, if a linked list has 5 elements, the 3rd element
from the end is the 3rd element. The function definition should look like
question5(ll, m), where ll is the first node of a linked list and m is the
"mth number from the end". You should copy/paste the Node class below to use
as a representation of a node in the linked list. Return the value of the
node at that position.
"""

