# instruction: Find all possible combinations of k numbers that add up to a number n, given that only numbers from 1 to 9 can be used and each combination should be a unique set of numbers.

# Ensure that numbers within the set are sorted in ascending order.

class Solution:
    # @param {integer} k
    # @param {integer} n
    # @return {integer[][]}
    def __init__(self):
        self.res = list()
    def combinationSum3(self, k, n):
        l = list()
        self.getCombinationHelper(l, k, n, 1)
        return self.res

    def getCombinationHelper(self, l, k, n, number):
        if k == 1:
            if n < number or n > 9:
                return
            l.append(n)
            self.res.append(l)
        else:
            for i in xrange(number, min(n/k + 1, 10)):
                sublist = list(l)
                sublist.append(i)
                self.getCombinationHelper(sublist, k - 1, n - i, i + 1)

'''
Given an array of integers, 
find if the array contains any duplicates. 
Your function should return true if any value appears at least twice in the array, and it should return false if every element is distinct.
'''
class Solution:
    # @param {integer[]} nums
    # @return {boolean}
    def containsDuplicate(self, nums):
        set1 = set()
        for num in nums:
            if num in set1:
                return True
            set1.add(num)
        return False


# implement trie
class TrieNode:
    # Initialize your data structure here.
    def __init__(self, val = None):
        self.val = val
        self.next = dict()
        self.isEnding = False


class Trie:

    def __init__(self):
        self.root = TrieNode()

    # @param {string} word
    # @return {void}
    # Inserts a word into the trie.
    def insert(self, word):
        r = self.root
        for characters in word:
            if characters not in r.next:
                node = TrieNode(characters)
                r.next[characters] = node
            r = r.next[characters]
        r.isEnding = True



    # @param {string} word
    # @return {boolean}
    # Returns if the word is in the trie.
    def search(self, word):
        r = self.root
        for characters in word:
            if characters not in r.next:
                return False
            r = r.next[characters]
        return r.isEnding


    # @param {string} prefix
    # @return {boolean}
    # Returns if there is any word in the trie
    # that starts with the given prefix.
    def startsWith(self, prefix):
        r = self.root
        for characters in prefix:
            if characters not in r.next:
                return False
            r = r.next[characters]
        return True

# Your Trie object will be instantiated and called as such:
# trie = Trie()
# trie.insert("somestring")
# trie.search("key")