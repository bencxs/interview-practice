
# coding: utf-8

# In[ ]:

'''Question 1'''
# Given two strings s and t, determine whether some anagram of t is a substring of s.
# For example: if s = "udacity" and t = "ad", then the function returns True.
# Your function definition should look like: question1(s, t) and return a boolean True or False.


# In[105]:

# Input: String s, String t
# Output: True or False for a valid substring of s
#
# Test cases:
# - t does not contain characters found in s
# - characters of t should only be used once in s
# - t or s has special characters (numbers or symbols)
# - input t or s as None
# - input s ="udacity", t = "ad" (valid case)
# - s = 'udacity', t = 'uy' - Invalid anagram. Not in dictionary
# - s = 'udacity', t = 'a' - Single character input in t
#
# Brainstorming:
# - Search each character in t wrt s
# - Compare characters in t and s if there are any overlaps
#
# Runtime:
# - O(n)


# In[10]:

# References: 
# https://pymotw.com/2/collections/counter.html
# http://stackoverflow.com/questions/8270092/python-remove-all-whitespace-in-a-string
# http://pythex.org/
# http://www.velvetcache.org/2010/03/01/looking-up-words-in-a-dictionary-using-python
# http://stackoverflow.com/questions/3788870/how-to-check-if-a-word-is-an-english-word-with-python
from nltk.corpus import wordnet
from collections import Counter
import re

def question1(s, t):
    
    # Check for None input
    if s == None or t == None:
        return False
    
    # Check for single-character in input t
    if len(t) <= 1:
        return False
    
    # Check for special characters (numbers, symbols).
    # Regex searches for any character except alpha ([a-zA-Z]) and whitespace (\s).
    if re.search('[^a-zA-Z*\s*]', s) or re.search('[^a-zA-Z*\s*]', t):
        return False

    # Anagram checker
    # Remove all whitespaces in strings and set all characters to lowercase
    s = s.replace(" ", "").lower()
    t = t.replace(" ", "").lower()

    c = Counter(s)
    d = Counter(t)
    
    ##print c, d
    
    # Finds intersection of Counter c and d (taking positive minimums) and compares if it still appears in d.
    # Checks for a valid anagram against WordNet from NLTK
    if c & d == d and wordnet.synsets(t):
        return True
    else:
        return False


# In[11]:

print "============== Question 1 ================"
# t does not contain characters found in s
# Should be False
print "Test Case 1 -", question1("udacity", "ar")

# characters of t should only be used once in s
# Should be False
print "Test Case 2 -", question1("udacity", "aa")

# t or s has special characters (numbers or symbols)
# Should be False
print "Test Case 3 -", question1("uda%city3", "a3")

# input t or s as None
# Should be False
print "Test Case 4 -", question1("udacity", None)

# s = 'udacity', t = 'a' - Single character input in t
# Should be False
print "Test Case 5 -", question1("udacity", "a")

# s = 'udacity', t = 'uy' - Invalid anagram. Not in dictionary
# Should be False
print "Test Case 6 -", question1("udacity", "uy")

# input s ='udacity', t = 'ad' (valid case)
# Should be True
print "Test Case 7 -", question1("udacity", "ad")


# In[ ]:

'''Question 2'''
# Given a string a, find the longest palindromic substring contained in a. 
# Your function definition should look like question2(a), and return a string.


# In[ ]:

# Input: String a
# Output: String - Longest palindromic substring of a
#
# Test cases:
# - input is None
# - input has no palindromes
# - input has a palindrome
# - input has a mix of symbols, numbers and whitespaces, with a valid palindrome
# - input is a sentence palindrome, where punctuation, capitalization, and spaces are usually ignored
# - input has an even palindrome
# - input has two or more palindromes, but of equal length
#
# Brainstorming:
# - Iterate over string a
# - For each character in string a, search one character to its adjacent left and right
# - If palindrome is found, store in memory and search two characters adjacent, three, etc until no more palindrome
#
# Runtime:
# - O(n^2)


# In[11]:

# Reference:
# http://stackoverflow.com/questions/17331290/how-to-check-for-palindrome-using-python-logic
# http://stackoverflow.com/questions/16343849/python-returning-longest-strings-from-list
# http://pythex.org/
import re

def question2(a):
    
    # Check for None input
    if a == None:
        return False
    
    # Remove punctuations [,.!':;?-], whitespaces (\s) and sets all characters to lowercase.
    a = re.sub("\s*[,.!':;?-]*", "", a).lower()
    
    # Helper to check for even and/or odd palindromes
    def palinCheck(window, r, k, i, a, min_valid, odd_ind):
        ##print window, r, k, i, a
        # While the indices are within bounds of the string
        while (i - k + 1) >= 0 and (i + k - 1) <= len(a):  
            # Checks the reverse of the subset of the string with [::-1]
            # Pass in min_valid = 2 for even palindromes and min_valid = 3 for odd
            if str(window) == str(window[::-1]) and len(window) >= min_valid:
                r.append(window)
                k += 1 # Increments index to search for adjacent characters to the right and left
                # Increase the window spread. odd_ind = 1 for odd palindromes
                window = a[i - k: i + k + odd_ind]
                palinCheck(window, r, k, i, a, min_valid, odd_ind) # Recursion until all possible palindromes are found
            return r

    r = [] # Stores any found palindrome
    k = 1 # Counter for adjacent character search
    
    # Starts with index 1, Ends with index -1
    # Even palindrome checker
    for i in range(1, len(a) - 1):
        # Min palindrome to search for is 2 characters in length
        window = a[i - k: i + k]
        palinCheck(window, r, k, i, a, min_valid=2, odd_ind=0)
        
    # Odd palindrome checker
    for i in range(1, len(a) - 1):
        # Min palindrome to search for is 3 characters in length
        window = a[i - k: i + k + 1]
        palinCheck(window, r, k, i, a, min_valid=3, odd_ind=1)
        
    # If there are no palindromes found, return False
    # Else, return the longest palindrome(s)
    if len(r) == 0:
        return False
    else:
        lp = max(len(x) for x in r)
        longest_palin = [x for x in r if len(x) == lp]
        return longest_palin

#print question2(a)


# In[12]:

print "============== Question 2 ================"
# input is None
# Should be False
print "Test Case 1 -", question2(None)

# input has no palindromes
# Should be False
print "Test Case 2 -", question2("circumstances")

# input has a palindrome
# Should be "reviver"
print "Test Case 3 -", question2("reviver")

# input has a mix of symbols, numbers and whitespaces, with a valid palindrome
# Should be "reviver"
print "Test Case 4 -", question2("s2f nt@!ofhrevivertglr%hn,n8s")

# input is a sentence palindrome, where punctuation, capitalization, and spaces are usually ignored
# Should be "wasitacaroracatisaw"
print "Test Case 5 -", question2("Was it a car or a cat I saw?")

# input has an even palindrome
# Should be "liveontimeemitnoevil"
print "Test Case 6 -", question2("Live on time, emit no evil")

# input has two or more palindromes, but of equal max length
# Should be "rotor" and "kayak"
print "Test Case 7 -", question2("rotorkayak")


# In[ ]:

'''Question 3'''
# Given an undirected graph G, find the minimum spanning tree within G. 
# A minimum spanning tree connects all vertices in a graph with the smallest possible total weight of edges. 
# Your function should take in and return an adjacency list structured like this:
'''
{'A': [('B', 2)],
 'B': [('A', 2), ('C', 5)], 
 'C': [('B', 5)]}
'''
# Vertices are represented as unique strings. The function definition should be question3(G)


# In[ ]:

# Input: Adjacency list of graph G
# Output: Adjacency list - min spanning tree
#
# Test cases:
# - input is None
# - input has a min spanning tree
# - graph is disconnected
#
# Brainstorming:
# - Choose an arbitary vertex, v 
# - Then, choose an edge that has smallest weight and grow the tree
# - Repeat until minimum spanning tree is obtained
#
# Runtime:
# - 


# In[176]:

# Reference:
# http://www.stoimen.com/blog/2012/11/19/computer-algorithms-prims-minimum-spanning-tree/
# http://stackoverflow.com/questions/3282823/get-key-with-the-least-value-from-a-dictionary
import numpy as np

'''G = {'A': [('B', 1), ('C', 4), ('D', 3)],
     'B': [('A', 1), ('D', 2)], 
     'C': [('A', 4), ('D', 5)],
     'D': [('C', 5), ('B', 2), ('A', 3)]}'''

'''
Graph Visualization:

A--1--B
| \   |
4  3  2
|   \ |
C--5--D
'''

# Undirected graph
G = {'A': [('B', 3), ('E', 1)],
     'B': [('A', 3), ('C', 9), ('D', 2), ('E', 2)], 
     'C': [('B', 9), ('D', 3), ('E', 7)],
     'D': [('B', 2), ('C', 3)],
     'E': [('A', 1), ('B', 2), ('C', 7)]}

'''
G Graph Visualization

A--3--B--9--C
|   / |    /|
1  2  2   3 |
| /   |  /  |
 E     D    |
 |          |
 +----7-----+

'''

# Disconnected graph
D = {'A': [('B', 1)],
     'B': [('A', 1)], 
     'C': [('D', 5)],
     'D': [('C', 5)]}

def question3(graph):
    # Reject None input
    if graph == None:
        return False
    
    # Initialize
    Q = {} # Priority queue
    P = {} # Parent

    # Select a vertex arbitrarily as the root
    root = np.random.choice(graph.keys())
    ##root = 'D'
    ##print "Root vertex:", root

    # Set priority of each member in Q to approx infinity
    for v in graph:
        Q[v] = 1e9
    # Set priority of starting vertex to 0
    Q[root] = 0
    
    # Set parent of starting vertex to null
    P[root] = None
    
    ##print Q
    ##print P
    
    # Prim's algorithm
    
    while Q:
        # Get minimum from Q. u=(key, priority value)
        u = min(Q.items(), key=lambda x: x[1])
        ##print "u: ", u
        # Initialize list to store neighboring vertices
        temp = []
        # Check all neighbor vertices to u
        for v in graph[u[0]]:
            ##print "v: ", v
            # If the vertex is found in Q and its weight is less than the priority...
            # v=(key, weight value)
            if v[0] in Q and v[1] < Q[v[0]]:
                temp.append(v)
                # Add u as the parent vertex of v
                ##P[u[0]] = [v]
                P[u[0]] = temp
                # And add the weight value as the new priority
                Q[v[0]] = v[1]
                ##print "P: ", P
        # Remove u from Q
        Q.pop(u[0])
        ##print "Q new: ", Q
    return P
        
##print question3(G)


# In[34]:

print "============== Question 3 ================"
# input is None
# Should be False
print "Test Case 1 -", question3(None)

# input has a min spanning tree
print "Test Case 2 -", question3(G)

# input graph is disconnected
print "Test Case 3 -", question3(D)


# In[ ]:

'''Question 4'''
#Find the least common ancestor between two nodes on a binary search tree. 
#The least common ancestor is the farthest node from the root that is an ancestor of both nodes. 
#For example, the root is a common ancestor of all nodes on the tree, 
#but if both nodes are descendents of the root's left child, then that left child might be the lowest common ancestor. 
#You can assume that both nodes are in the tree, and the tree itself adheres to all BST properties. 
#The function definition should look like question4(T, r, n1, n2), 
#where T is the tree represented as a matrix, 
#where the index of the list is equal to the integer stored in that node and a 1 represents a child node, 
#r is a non-negative integer representing the root, 
#and n1 and n2 are non-negative integers representing the two nodes in no particular order. 
#For example, one test case might be

'''
question4([[0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [1, 0, 0, 0, 1],
           [0, 0, 0, 0, 0]],
          3,
          1,
          4)
'''
#and the answer would be 3.

### M[i][j] = 1, where i is an ancestor to j
'''
Tree representation
    3
   / \
  0   4
 /
1 
'''


# In[ ]:

# Input: Matrix of BST, root node, node 1, node 2
# Output: (Integer) least common ancestor of both nodes
#
# Test cases:
# - input is None
# - input is not a valid matrix
# - input has a least common ancestor
# - input has no LCA
# - input has LCA other than at root level
#
# Brainstorming:
# - Create BST from matrix, via insertion
# - Search for LCA using single traversal
# - 
#
# Runtime:
# - O(n)


# In[28]:

# Reference:
# http://blog.rdtr.net/post/algorithm/algorithm_tree_lowest_common_ancestor_of_a_binary_tree/
# http://www.ritambhara.in/build-binary-tree-from-ancestor-matrics/
# http://yucoding.blogspot.my/2016/04/leetcode-question-lowest-common.html
# http://www.geeksforgeeks.org/lowest-common-ancestor-binary-tree-set-1/

M = [[0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1],
    [0, 0, 0, 0, 0]]

'''
M tree visualization
    3
   / \
  0   4
   \
    1 
'''

# Invalid tree with values other than 0 or 1
J = [[0, 1, 0, 0, -1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, -1],
    [1, 0, 0, 0, 0],
    [0, 0, 0, 0, 2]]

K = [[0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0]]

'''
K tree visualization
     4
   /   \
  2     5
 / \   / \
0   1 3   6
'''

'''
# Lowest Common Ancestor
def question4(T, r, n1, n2):
    # Handle None input
    if T == None or r == None or n1 == None or n2 == None:
        return "None input entered."
    
    # Construct dict of {node: parent}
    node_parent_map = {}
    for i in range(len(T)):
        for j in range(len(T)):
            # Handle invalid matrix
            if T[i][j] < 0 or T[i][j] > 1:
                return "Invalid matrix entered."
            # Add key:value pair if a node-parent relationship exists
            # T[i][j] == 1, where i is an ancestor to j
            elif T[i][j] == 1:
                node_parent_map[j] = i

    # Initialize path from node to root
    path = []
    # Initialize current node n1
    current = n1
    # Traverse and add nodes to path until we reach the root
    while current in node_parent_map:
        # Add node to path
        path.append(current)
        # Switch to next node in path
        current = node_parent_map[current]
    # Add root as the last node in path
    if current == r:
        path.append(current)
    # Handle no LCA case
    else:
        return "No LCA found with specified root."
        
    # Find first common node for n1 and n2 in path
    # Initialize node n2
    current = n2
    # Keep running check for node n2 to exist in path
    while current not in path:
        # Traverse up the tree to the node's parent
        current = node_parent_map[current]
    # When the node n2 exists in path, return it since it is the LCA
    return current
'''

# Lowest Common Ancestor
def question4(T, r, n1, n2):
    # Handle None input
    if T == None or r == None or n1 == None or n2 == None:
        return "None input entered."
    
    # Construct dict of {node: parent}
    node_parent_map = {}
    for i in range(len(T)):
        for j in range(len(T)):
            # Handle invalid matrix
            if T[i][j] < 0 or T[i][j] > 1:
                return "Invalid matrix entered."
            # Add key:value pair if a node-parent relationship exists
            # T[i][j] == 1, where i is an ancestor to j
            elif T[i][j] == 1:
                node_parent_map[j] = i
    print node_parent_map

    # Traverse tree from root node
    # Get child nodes from root
    temp = {}
    for node, parent in node_parent_map.iteritems():
        if parent == r:
            temp[node] = parent
    print temp
    for node, parent in temp.iteritems():
        #if n1 <= node and n2 <= node:
        
        print "node", node
        print "1", n1 <= node
        print "2", n2 <= node
        # then search correct key
            
print question4(K, 4, 0, 1)


# In[54]:

print "============== Question 4 ================"
# input is None
# Should be "None input entered."
print "Test Case 1 -", question4(None, 3, 1, 4)

# input is an invalid matrix
# Should be "Invalid matrix entered."
print "Test Case 2 -", question4(J, 3, 1, 4)

# input has a Least Common Ancestor
# Should be "3"
print "Test Case 3 -", question4(M, 3, 1, 4)

# input has no Least Common Ancestor
# Should be "No LCA found with specified root."
print "Test Case 4 -", question4(M, 0, 1, 4)

# input has LCA other than at root level
# Should be "5"
print "Test Case 5 -", question4(K, 4, 3, 6)


# In[ ]:

'''Question 5'''
#Find the element in a singly linked list that's m elements from the end. 
#For example, if a linked list has 5 elements, the 3rd element from the end is the 3rd element. 
#The function definition should look like question5(ll, m), 
#where ll is the first node of a linked list and m is the "mth number from the end". 
#You should copy/paste the Node class below to use as a representation of a node in the linked list. 
#Return the value of the node at that position.

'''
class Node(object):
  def __init__(self, data):
    self.data = data
    self.next = None
'''


# In[ ]:

# Input: Singly linked list and integer m (m items from the end of the list)
# Output: (Integer/Float/Char) Data element m
#
# Test cases:
# - input is None
# - input is a valid singly linked list
# - input m is not within list
# - 
# - 
#
# Brainstorming:
# - Create linked list class
# - Add elements to linked list
# - Search elements in linked list forward
# - Calculate length of list
# - Use this calculation to find the reverse position from the end of the list
#
# Runtime:
# - O(n)


# In[166]:

# Reference:
# https://www.codefellows.org/blog/implementing-a-singly-linked-list-in-python/
# https://classroom.udacity.com/nanodegrees/nd009/parts/00913454013/modules/773670769775460/lessons/7117335401/concepts/78875247320923#

# Indivisual node class for linked list
class Node(object):
    def __init__(self, data):
        self.data = data
        self.next = None

# Linked list class
class LinkedList(object):
    def __init__(self, head=None):
        self.head = head
    
    def get_size(self):
        # Initialize head as starting node
        # and length = 0
        current = self.head
        length = 0
        # Traverse through next nodes until the end and add 1 to the length each time
        while current:
            length += 1
            current = current.next
        return length
   
    def append(self, new_node):
        current = self.head
        # If head node is present...
        if self.head:
            while current.next:
                # Cycle through the next nodes
                current = current.next
            # And append new node to end of list
            current.next = new_node
        else:
            # Add node as the head
            self.head = new_node
    
    def get_position(self, position):
        counter = 1
        current = self.head
        # Handle invalid position
        if position < 1:
            return None
        while current and counter <= position:
            # Returns node if position matches with counter
            if counter == position:
                return current
            # Cycle through next nodes
            current = current.next
            counter += 1
        # If position of node is not found return None
        return None
    
    def get_position_reverse(self, position, size):
        counter = 1
        current = self.head
        # Calculates the position integer from total length of list
        # Then, search through list forward until position is found
        position = size - position
        if position < 1:
            return None
        while current and counter <= position:
            if counter == position:
                return current
            current = current.next
            counter += 1
        return None
        
# Set up nodes
n1 = Node(1)
n2 = Node(2)
n3 = Node(3)
n4 = Node(4)
n5 = Node(5)

# Set up Linked List
# 1 -> 2 -> 3 -> 4 -> 5
linkl = LinkedList(n1)
linkl.append(n2)
linkl.append(n3)
linkl.append(n4)
linkl.append(n5)

# Test out linked list
# Should be 3
print "Test Position:", linkl.get_position(3).data
# Should be 1
print "Test Position:", linkl.head.data

# Initialize empty list for Test Case 4
link2 = LinkedList()

def question5(ll, m):
    # Handle None input
    if ll == None or m == None:
        return "None input entered."
    # Handle empty list
    if ll.get_size() == 0:
        return "Linked list is empty."
    # Get length of linked list
    size = ll.get_size()
    # Handle None output
    if ll.get_position_reverse(m, size) == None:
        return "No position found."
    else:
        # Get node position from end of list
        node_val = ll.get_position_reverse(m, size).data
        return node_val


# In[169]:

print "============== Question 5 ================"
# input is None
# Should be "None input entered."
print "Test Case 1 -", question5(None, None)

# input is a valid singly linked list
# Should be 4
# linkl: 1 -> 2 -> 3 -> 4 -> 5
# 1 node from the end = node 4
print "Test Case 2 -", question5(linkl, 1)

# input m is not within list
# Should be "No position found."
print "Test Case 3 -", question5(linkl, 8)

# input ll is an empty list
# Should be "Linked list is empty."
print "Test Case 4 -", question5(link2, 1)


# In[ ]:



