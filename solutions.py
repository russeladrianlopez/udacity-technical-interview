#!/usr/bin/env python

import unittest  # testing purposes
import operator  # for sorting the weight in question #3

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

def question2(s):
    # Assess if s is a string or empty
    if not s or not isinstance(s, basestring):
        return 'Input not a string or empty'

    palindromes = []
    # lowercase and remove whitespaces on the string
    s = s.lower()
    s = s.replace(" ", "")
    for i in range(len(s)):
        for j in range(0, i):
            substring = s[j:i + 1]
            if substring == substring[::-1]:
                palindromes.append(substring)

    if palindromes:
        longest = max(palindromes, key=len)
        return 'Longest palindrome: %s' % (longest)
    return 'No Palindrome found'


class TestQuestion2(unittest.TestCase):
    print '\nTESTCASES FOR QUESTION #2: \n'

    def test_haspalindrome(self):
        self.assertEqual(question2('aa'), 'Longest palindrome: aa')
        self.assertEqual(question2('aasomething'), 'Longest palindrome: aa')
        self.assertEqual(question2('race car'), 'Longest palindrome: racecar')
        self.assertEqual(question2('aabb bbccc'), 'Longest palindrome: bbbb')

    def test_nopalindrome(self):
        self.assertEqual(question2('udacity'), 'No Palindrome found')
        self.assertEqual(question2('abcdefgh'), 'No Palindrome found')

    def test_notstring(self):
        self.assertEqual(question2(''), 'Input not a string or empty')
        self.assertEqual(question2(1), 'Input not a string or empty')
        self.assertEqual(question2(True), 'Input not a string or empty')
        self.assertEqual(question2(None), 'Input not a string or empty')


suite = unittest.TestLoader().loadTestsFromTestCase(TestQuestion2)
unittest.TextTestRunner(verbosity=2).run(suite)

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

# Implement a Krusal algorithm
parent = {}  # records parental relationship between the vertices
rank = {}  # flatten the tree for better performance, depth of the vertex


# initialize disjoint sets. each set contains one vertex(v). rank is
# used to keep the tree flat as much as possible for better search.
def create_set(v):
    parent[v] = v
    rank[v] = 0


# find the root to which this vertex belongs
def find(vertex):
    if parent[vertex] == vertex:
        return parent[vertex]
    else:
        return find(parent[vertex])


# merge the sets represented by these two given root nodes
def union(vertice1, vertice2):
    root1 = find(vertice1)
    root2 = find(vertice2)
    if root1 != root2:
        if rank[root1] > rank[root2]:
            parent[root2] = root1
        else:
            parent[root1] = root2
    else:
        parent[root1] = root2
        rank[root2] += 1


def kruskal(vertices, edges):
    minSpanningTree = set()
    for vertex in vertices:
        create_set(vertex)

    # sort edges by according to weights
    edges = sorted(edges, key=operator.itemgetter(2))

    for edge in edges:
        vertex1, vertex2, weight = edge
        if find(vertex1) != find(vertex2):
            union(vertex1, vertex2)
            minSpanningTree.add(edge)

    return minSpanningTree


def question3(G):
    if G:
        vertices = []
        edges = []

        # pre process given input graph and extract all vertices and edges
        try:
            for vertex in G.keys():
                # append vertices
                vertices.append(vertex)
                # extract the edges of each vertex
                for edge in G[vertex]:
                    direction, weight = edge
                    edges.append((vertex, direction, weight))
        except AttributeError:
            return 'Input is not a Graph Dictionary'
    else:
        return "Empty Graph"

    # perform Kruskal algorithm
    tree = kruskal(vertices, edges)
    print tree
    # post process results into the required output format
    output = {}
    for vertex in tree:
        origin, direction, weight = vertex

        if origin in output:
            output[origin].append((direction, weight))
        else:
            output[origin] = [(direction, weight)]

        if direction in output:
            output[direction].append((origin, weight))
        else:
            output[direction] = [(origin, weight)]
    return output

class TestQuestion3(unittest.TestCase):
    print '\nTESTCASES FOR QUESTION #3: \n'

    def test_hasgraph(self):
        self.assertEqual(question3({'A': [('B', 4), ('F', 2)],
                                    'B': [('C', 6), ('F', 5)],
                                    'C': [('B', 6), ('D', 3), ('F', 1)],
                                    'D': [('C', 3), ('E', 2)],
                                    'E': [('D', 2), ('F', 4)],
                                    'F': [('B', 5), ('C', 1), ('E', 4)]
                                    }), {'A': [('B', 4), ('F', 2)],
                                         'C': [('F', 1), ('D', 3)],
                                         'B': [('A', 4)], 'E': [('D', 2)],
                                         'D': [('E', 2), ('C', 3)],
                                         'F': [('C', 1), ('A', 2)]
                                         })
        self.assertEqual(question3({'A': [('B', 2)],
                                    'B': [('A', 2), ('C', 5)],
                                    'C': [('B', 5)]
                                    }), {'A': [('B', 2)],
                                         'C': [('B', 5)],
                                         'B': [('C', 5), ('A', 2)]
                                         })
        self.assertEqual(question3({'A': [('B', 1), ('C', 1)],
                                    'B': [('A', 1), ('C', 1)],
                                    'C': [('A', 1), ('B', 1)],
                                    }), {'A': [('C', 1), ('B', 1)],
                                         'C': [('A', 1)],
                                         'B': [('A', 1)]
                                         })

    def test_emptygraph(self):
        self.assertEqual(question3({}), 'Empty Graph')
        self.assertEqual(question3(None), 'Empty Graph')

    def test_isgraph(self):
        self.assertEqual(question3('String'),
                         'Input is not a Graph Dictionary')
        self.assertEqual(question3(True), 'Input is not a Graph Dictionary')
        self.assertEqual(question3(123), 'Input is not a Graph Dictionary')


suite = unittest.TestLoader().loadTestsFromTestCase(TestQuestion3)
unittest.TextTestRunner(verbosity=2).run(suite)

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


# Node class initialize
class Q4Node(object):

    def __init__(self, value):
        self.value = value
        self.right = None
        self.left = None


# insert a child node on the right
def push_right(node, new_value):
    new_node = Q4Node(new_value)
    node.right = new_node
    return new_node


# insert a child node on the left
def push_left(node, new_value):
    new_node = Q4Node(new_value)
    node.left = new_node
    return new_node


# find LCA of n1 and n2.
def getLCA(root, n1, n2):
    # Base Case
    if root is None:
        return "Root doesnt exist"

    if not isinstance(n1, int) or not isinstance(n2, int):
        return "Node input not integer"
    # moves the root pointer to left
    # if both nodes are greater value than the root
    if(root.value > n1 and root.value > n2):
        return getLCA(root.left, n1, n2)
    # moves the root pointer to right
    # if both nodes are lesser value than the root
    elif(root.value < n1 and root.value < n2):
        return getLCA(root.right, n1, n2)

    # If root on the middle of both nodes
    return root.value


def question4(T, r, n1, n2):
    # Validate inputs
    # Create Tree based on Matrix
    root = Q4Node(r)
    node_value = 0
    node_list = []

    # setting up the root
    for index in T[r]:
        if index:
            if(node_value > r):
                node_list.append(push_right(root, node_value))
            else:
                node_list.append(push_left(root, node_value))
        node_value += 1

    # setting up the child nodes
    node = node_list.pop(0)
    while node_list != []:
        node_value = 0
        for index in T[node.value]:
            if index:
                if(node_value > node.value):
                    node_list.append(push_right(node, node_value))
                else:
                    node_list.append(push_left(node, node_value))
            node_value += 1
        node = node_list.pop(0)

    return getLCA(root, n1, n2)


class TestQuestion4(unittest.TestCase):
    print '\nTESTCASES FOR QUESTION #4: \n'

    def test_getLCA(self):
        self.assertEqual(question4([[0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [1, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 0]],
                                   3,
                                   1,
                                   4), 3)
        self.assertEqual(question4([[0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [1, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 0]],
                                   3,
                                   1,
                                   0), 0)
        self.assertEqual(question4([[0, 0, 0, 0, 0, 0],
                                    [1, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1],
                                    [0, 0, 1, 1, 0, 0],
                                    [0, 0, 0, 0, 0, 0]],
                                   1,
                                   2,
                                   5), 4)

    def test_matrixinput(self):
        with self.assertRaises(IndexError):
            question4([], 3, 1, 4)

    def test_nodeinput(self):
        with self.assertRaises(TypeError):
            question4([[0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [1, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]],
                      'stringroot',
                      1,
                      4)
        self.assertEqual(question4([[0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [1, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 0]],
                                   3,
                                   'stringnode1',
                                   4), 'Node input not integer')
        self.assertEqual(question4([[0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [1, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 0]],
                                   3,
                                   1,
                                   'stringnode2'), 'Node input not integer')


suite = unittest.TestLoader().loadTestsFromTestCase(TestQuestion4)
unittest.TextTestRunner(verbosity=2).run(suite)

"""
Question 5: Find the element in a singly linked list that's m elements
from the end. For example, if a linked list has 5 elements, the 3rd element
from the end is the 3rd element. The function definition should look like
question5(ll, m), where ll is the first node of a linked list and m is the
"mth number from the end". You should copy/paste the Node class below to use
as a representation of a node in the linked list. Return the value of the
node at that position.
"""


# Node class initialize
class Node(object):

    def __init__(self, data):
        self.data = data
        self.next = None


# Linked list class
class LinkedList(object):

    def __init__(self, head=None):
        self.head = head

    def push(self, new_data):
        current = self.head
        if self.head:
            while current.next:
                current = current.next
            current.next = new_data
        else:
            self.head = new_data

    def elemfromEnd(self, m):
        pointer = self.head  # moves in respect with the brake
        brake = self.head  # is setup to indicate the stop

        # setup brake m nodes apart from pointer
        count = 0
        if brake is not None:
            while count < m:
                if brake is None:
                    return "Input '%s' exceeds the length of the list" % (m)
                brake = brake.next
                count += 1

        # move both pointer and brake until brake goes beyond the list length
        while brake is not None:
            pointer = pointer.next
            brake = brake.next

        return pointer.data


def question5(ll, m):
    if ll is not None and isinstance(ll, LinkedList):
        return ll.elemfromEnd(m)
    return "Input '%s' is not a Linkedlist object" % (ll)


class TestQuestion5(unittest.TestCase):
    print '\nTESTCASES FOR QUESTION #5: \n'

    def setUp(self):
        # setup Linked List nodes
        node1 = Node(1)
        node2 = Node(2)
        node3 = Node(3)
        node4 = Node(4)
        node5 = Node(5)

        self.ll = LinkedList(node1)
        self.ll.push(node2)
        self.ll.push(node3)
        self.ll.push(node4)
        self.ll.push(node5)

    def test_elemFromEnd(self):
        self.assertEqual(question5(self.ll, 3), 3)
        self.assertEqual(question5(self.ll, 2), 4)
        self.assertEqual(question5(self.ll, 5), 1)

    def test_notLinkedList(self):
        self.assertEqual(question5('string', 3),
                         "Input 'string' is not a Linkedlist object")
        self.assertEqual(question5(1, 3),
                         "Input '1' is not a Linkedlist object")
        self.assertEqual(question5(None, 3),
                         "Input 'None' is not a Linkedlist object")

    def test_m_inputistoomuch(self):
        self.assertEqual(question5(self.ll, 30),
                         "Input '30' exceeds the length of the list")


suite = unittest.TestLoader().loadTestsFromTestCase(TestQuestion5)
unittest.TextTestRunner(verbosity=2).run(suite)

# Combine all test
'''
test_cases = (TestQuestion1, TestQuestion2, TestQuestion3,
              TestQuestion4, TestQuestion5)


def load_tests(loader, tests_cases, pattern):
    suite = unittest.TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    return suite


suite = load_tests(unittest.TestLoader(), test_cases, None)
unittest.TextTestRunner(verbosity=2).run(suite)
'''