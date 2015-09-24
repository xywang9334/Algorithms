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



#Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.
#
#For example,
#"A man, a plan, a canal: Panama" is a palindrome.
#"race a car" is not a palindrome.

class Solution(object):
    def isPalindrome(self, s):
        """
            :type s: str
            :rtype: bool
            """
        if s == "":
            return True
        string = ''.join(e for e in s if e.isalnum()).lower()
        return string == string[::-1]

#
#Given a non-negative number represented as an array of digits, plus one to the number.
#
#The digits are stored such that the most significant digit is at the head of the list.


class Solution(object):
    def plusOne(self, digits):
        """
            :type digits: List[int]
            :rtype: List[int]
            """
        length = len(digits)
        result = (digits[length - 1] + 1) % 10
        carry = (digits[length - 1] + 1) / 10
        digits[length - 1] = result
        for i in reversed(xrange(length - 1)):
            digits[i] = digits[i] + carry
            if digits[i] >= 10:
                digits[i] = digits[i] - 10
                carry = 1
            else:
                carry = 0
        if carry == 1:
            digits.insert(0, 1)
        return digits

#Given a column title as appear in an Excel sheet, return its corresponding column number.
#
#For example:
#    
#    A -> 1
#    B -> 2
#    C -> 3
#    ...
#    Z -> 26
#    AA -> 27
#    AB -> 28

class Solution(object):
    def titleToNumber(self, s):
        """
            :type s: str
            :rtype: int
            """
        length = len(s)
        re = 0
        i = 0
        for char in s:
            num = ord(char) - ord('A') + 1
            re += num * 26 ** (length - i - 1)
            i += 1
        return re


#Given a string s consists of upper/lower-case alphabets and empty space characters ' ', return the length of last word in the string.
#
#If the last word does not exist, return 0.
#
#Note: A word is defined as a character sequence consists of non-space characters only.
class Solution(object):
    def lengthOfLastWord(self, s):
        """
            :type s: str
            :rtype: int
            """
        l = s.split()
        length = len(l)
        if length > 0:
            return len(l[length - 1])
        else:
            return 0


#Write a program to check whether a given number is an ugly number.
#
#Ugly numbers are positive numbers whose prime factors only include 2, 3, 5. For example, 6, 8 are ugly while 14 is not ugly since it includes another prime factor 7.
#
#Note that 1 is typically treated as an ugly number.

class Solution(object):
    def isUgly(self, num):
        """
            :type num: int
            :rtype: bool
            """
        if num == 0:
            return False
        while num % 5 == 0 or num % 2 == 0 or num % 3 == 0:
            if num % 5 == 0:
                num /= 5
            if num % 2 == 0:
                num /= 2
            if num % 3 == 0:
                num /= 3
        if num == 1:
            return True
        else:
            return False


#Given a non-negative integer num, repeatedly add all its digits until the result has only one digit.
#
#For example:
#
#Given num = 38, the process is like: 3 + 8 = 11, 1 + 1 = 2. Since 2 has only one digit, return it.

class Solution(object):
    def addDigits(self, num):
        """
            :type num: int
            :rtype: int
            """
        while num / 10 != 0:
            num = self.get_digits(num)
        return num
    
    
    def get_digits(self, num):
        temp = num
        sum = 0
        while temp != 0:
            sum += temp % 10
            temp /= 10
        return sum

#Given a binary tree, return all root-to-leaf paths.
#
#For example, given the following binary tree:
#
#      1
#    /   \
#   2     3
#    \
#    5
#All root-to-leaf paths are:
#
#["1->2->5", "1->3"]


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def binaryTreePaths(self, root):
        """
            :type root: TreeNode
            :rtype: List[str]
            """
        l = list()
        ls = list()
        if not root:
            return l
        self.call_with_string(l, root, ls)
        return ls
    
    def call_with_string(self, l, root, ls):
        l.append(root.val)
        if (not root.left) and (not root.right):
            s = self.make_string(l)
            ls.append(s)
            l.pop()
            return
        if root.left and (not root.right):
            self.call_with_string(l, root.left, ls)
            l.pop()
        elif root.right and (not root.left):
            self.call_with_string(l, root.right, ls)
            l.pop()
        else:
            self.call_with_string(l, root.left, ls)
            self.call_with_string(l, root.right, ls)
            l.pop()

    def make_string(self, l):
        s = ""
        if not l:
            return s
        s += str(l[0])
            length = len(l)
            for i in xrange(1, length):
                s += "->"
                s+= str(l[i])
        return s


""" check to see if two strings are anagrams """
class Solution(object):
    def isAnagram(self, s, t):
        """
            :type s: str
            :type t: str
            :rtype: bool
            """
        d = dict()
        for char in s:
            d[char] = d.get(char, 0) + 1
        for char in t:
            d[char] = d.get(char, 0) - 1
            if d.get(char) < 0:
                return False
        for key, value in d.items():
            if value != 0:
                return False
        return True


#Write a function to delete a node (except the tail) in a singly linked list, given only access to that node.
#
#Supposed the linked list is 1 -> 2 -> 3 -> 4 and you are given the third node with value 3, the linked list should become 1 -> 2 -> 4 after calling your function.

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution(object):
    def deleteNode(self, node):
        """
            :type node: ListNode
            :rtype: void Do not return anything, modify node in-place instead.
            """
        node.val = node.next.val
        node.next = node.next.next


# Find the common ancestor of a binary tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
            :type root: TreeNode
            :type p: TreeNode
            :type q: TreeNode
            :rtype: TreeNode
            """
        if (not self.covers(root, p)) or (not self.covers(root, q)):
            return None
        return self.helper_func(root, p, q)
    
    def covers(self, a, b):
        if not a:
            return False
        if a == b:
            return True
        return (self.covers(a.left, b)) or (self.covers(a.right, b))
    
    def helper_func(self, root, p, q):
        if not root:
            return None
        if root == p or root == q:
            return root
        p_on_left = self.covers(root.left, p)
        q_on_left = self.covers(root.left, q)
        if p_on_left != q_on_left:
            return root
        else:
            if p_on_left:
                return self.helper_func(root.left, p, q)
            else:
                return self.helper_func(root.right, p, q)

# if the tree is a binary search tree
class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
            :type root: TreeNode
            :type p: TreeNode
            :type q: TreeNode
            :rtype: TreeNode
            """
        if not root:
            return None
        if p==root or q==root:
            return root
        if p.val>q.val:
            p,q=q,p
        if p.val<root.val and q.val>root.val:
            return root
        if p.val>root.val:
            return self.lowestCommonAncestor(root.right, p, q)
        return self.lowestCommonAncestor(root.left, p, q)

# determine if a linkedlist is a palindrome in O(N) time and O(1) space
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def isPalindrome(self, head):
        """
            :type head: ListNode
            :rtype: bool
            """
        if not head:
            return True
    
        slow = head
        fast = head.next
        pre = None
        
        # move and change the list to inverse linked list
        while fast and fast.next:
            tmp = slow.next
            fast = fast.next.next
            slow.next = pre
            pre = slow
            slow = tmp

        left = slow
        if not fast:
            left = pre
            right = slow.next
            slow.next = pre
        while right and left:
            if right.val == left.val:
                right = right.next
                left = left.next
            else:
                return False
        return True





# implement a queue using list
class Queue(object):
    def __init__(self):
        """
            initialize your data structure here.
            """
        self.list1 = list()
    
    
    def push(self, x):
        """
            :type x: int
            :rtype: nothing
            """
        self.list1.append(x)
    
    
    def pop(self):
        """
            :rtype: nothing
            """
        self.list1.pop()
    
    
    def peek(self):
        """
            :rtype: int
            """
        if self.list1 != []:
            return self.list1[0]
        else:
            return None


    def empty(self):
        """
            :rtype: bool
        """
        return self.list1 == []


# determine if a number is a power of 2
class Solution(object):
    def isPowerOfTwo(self, n):
        """
            :type n: int
            :rtype: bool
            """
        if n == 0:
            return False
        while n % 2 == 0:
            n /= 2
        if n == 1:
            return True
        return False


#Given a sorted integer array without duplicates, return the summary of its ranges.
#
#For example, given [0,1,2,4,5,7], return ["0->2","4->5","7"].

class Solution(object):
    def summaryRanges(self, nums):
        """
            :type nums: List[int]
            :rtype: List[str]
            """
        l = list()
        if not nums:
            return l
        if len(nums) == 1:
            l.append(str(nums[0]))
            return l
        counter = nums[0] + 1
        start = counter
        for num in nums[1:]:
            if num == counter:
                counter += 1
            else:
                if counter == start:
                    l.append(str(counter - 1))
                else:
                    l.append(str(start - 1) + "->" + str(counter - 1))
                counter = num + 1
                start = num + 1
            if num == nums[len(nums) - 1]:
                if counter == start:
                    l.append(str(counter - 1))
                else:
                    l.append(str(start - 1) + "->" + str(counter - 1))
        return l


# invert a binary tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def invertTree(self, root):
        """
            :type root: TreeNode
            :rtype: TreeNode
            """
        if root == None:
            return root
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root


# implement stack using queue
class Stack(object):
    def __init__(self):
        """
            initialize your data structure here.
            """
        self.l = list()
        self.temp = list()
    
    
    def push(self, x):
        """
            :type x: int
            :rtype: nothing
            """
        self.l.append(x)
    
    
    def pop(self):
        """
            :rtype: nothing
            """
        for num in self.l:
            self.temp.append(num)
        self.l = []
        for num in self.temp[:-1]:
            self.l.append(num)
        self.temp = []
    
    
    def top(self):
        """
            :rtype: int
            """
        for num in self.l[::-1]:
            self.temp.append(num)
        val = self.temp[0]
        self.temp = []
        return val
    
    
    
    def empty(self):
        """
            :rtype: bool
            """
        return self.l == []


# Find the total area covered by two rectilinear rectangles in a 2D plane.
class Solution(object):
    def computeArea(self, A, B, C, D, E, F, G, H):
        """
            :type A: int
            :type B: int
            :type C: int
            :type D: int
            :type E: int
            :type F: int
            :type G: int
            :type H: int
            :rtype: int
            """
        
        area_total = abs((C - A) * (D - B)) + abs((G - E) * (F - H))
        overlap = max((min(C, G) - max(A, E)), 0) * max((min(D, H) - max(B, F)), 0)
        return area_total - overlap




# returns true if a list contains duplicates
# Otherwise, returns false

class Solution(object):
    def containsDuplicate(self, nums):
        """
            :type nums: List[int]
            :rtype: bool
            """
        d = dict()
        for num in nums:
            d[num] = d.get(num, 0) + 1
            if d[num] > 1:
                return True
        return False




# Count the number of trailing zero in n!

class Solution(object):
    def trailingZeroes(self, n):
        """
            :type n: int
            :rtype: int
            """
        
        # count the number of 5s in the factorization
        if n == 0:
            return 0
        return n / 5 + self.trailingZeroes(n / 5)

#Given an array of integers and an integer k,
#find out whether there are two distinct indices i and j in the array such that nums[i] = nums[j] and the difference between i and j is at most k.
class Solution(object):
    def containsNearbyDuplicate(self, nums, k):
        """
            :type nums: List[int]
            :type k: int
            :rtype: bool
            """
        d = dict()
        for k1, v in list(enumerate(nums)):
            if v in d and k1 - d[v] <= k:
                return True
            d[v] = k1
        return False

# reverse an integer
class Solution(object):
    def reverse(self, x):
        """
            :type x: int
            :rtype: int
            """
        is_negative = False
        if x < 0:
            x = -x
            is_negative = True
        n = 0
        length = len(str(x))
        while x != 0:
            n += (x % 10) * (10 ** (length - 1))
            length -= 1
            x /= 10
        
        # check overflows
        if n > 2147483647:
            return 0
        if is_negative:
            n = -n
        return n

# implement atoi

class Solution(object):
    def myAtoi(self, string):
        """
            :type str: str
            :rtype: int
            """
        
        string = string.strip()
        integer = ""
        first_occur = False
        for char in string:
            if char not in set(['+','-']) and (not str(char).isdigit()) and not first_occur:
                break
            if (not str(char).isdigit()) and first_occur:
                break
            if char == '+' and not first_occur:
                first_occur = True
            elif char == '-' and not first_occur:
                first_occur = True
                integer += "-"
            elif str(char).isdigit():
                integer += char
                first_occur = True
    
        if integer == "" or integer == "-" or integer == "+":
            return 0

        i = int(integer)
        if i > 2147483647:
            return 2147483647
        elif i < -2147483648:
            return -2147483648
        else:
            return i

# know if digit is a palindrome
class Solution(object):
    def isPalindrome(self, x):
        """
            :type x: int
            :rtype: bool
            """
        if x < 0:
            return False
        length = len(str(x))
        i = length
        if length % 2:
            length += 1
        while i > 0:
            if x % 10 != x / (10 ** (i - 1)):
                return False
            x -= (x % 10) * (10 ** (i - 1))
            i -= 2
            x /= 10
        return True

# reverse a linkedlist
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def reverseList(self, head):
        """
            :type head: ListNode
            :rtype: ListNode
            """
        pointer = head
        pre = None
        new_head = None
        while pointer != None:
            temp = pointer.next
            if temp == None:
                new_head = pointer
            pointer.next = pre
            pre = pointer
            pointer = temp
        return new_head


#
#Given two strings s and t, determine if they are isomorphic.
#
#Two strings are isomorphic if the characters in s can be replaced to get t.
#
#All occurrences of a character must be replaced with another character while preserving the order of characters. No two characters may map to the same character but a character may map to itself.
#
#For example,
#Given "egg", "add", return true.
#
#Given "foo", "bar", return false.
#
#Given "paper", "title", return true.

class Solution(object):
    def isIsomorphic(self, s, t):
        """
            :type s: str
            :type t: str
            :rtype: bool
            """
        d_t = dict()
        d_s = dict()
        if len(s) != len(t):
            return False
        for x in xrange(len(s)):
            if s[x] not in d_s:
                d_s[s[x]] = [x]
            if t[x] not in d_t:
                d_t[t[x]] = [x]
            if s[x] in d_s:
                d_s[s[x]] += [x]
            if t[x] in d_t:
                d_t[t[x]] += [x]

    return sorted(d_t.values()) == sorted(d_s.values())

# Count the number of prime numbers less than a non-negative number, n.
class Solution(object):
    def countPrimes(self, n):
        """
            :type n: int
            :rtype: int
            """
        if n == 0 or n == 1:
            return 0
        
        l = [True] * n
        l[0] = l[1] = False
        for i in xrange(2, n):
            if l[i]:
                for x in xrange(2, n / i + 1):
                    if x * i < n:
                        l[x * i] = False
        return sum(l)

# remove element by value in a linkedlist
class Solution(object):
    def removeElements(self, head, val):
        """
            :type head: ListNode
            :type val: int
            :rtype: ListNode
            """
        head2 = head
        pre = head
        pointer = head
        
        while pointer != None:
            if pointer.val == val and pointer == head2:
                pre = pointer
                pointer = pointer.next
                head2 = head2.next
            elif pointer.val == val:
                pre.next = pointer.next
                pointer = pointer.next
            else:
                pre = pointer
                pointer = pointer.next

    return head2


# if a number is a happy number

class Solution(object):
    def isHappy(self, n):
        """
            :type n: int
            :rtype: bool
            """
        
        sum = n
        d = []
        
        while sum != 1:
            string = str(n)
            sum = 0
            for char in string:
                sum += int(char) ** 2
            n = sum
            if n in d:
                return False
            d.append(n)
        
        if n == 1:
            return True
        return False

# dp
#You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security system connected and it will automatically contact the police if two adjacent houses were broken into on the same night.
#
#Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.

class Solution(object):
    def rob(self, nums):
        """
            :type nums: List[int]
            :rtype: int
            """
        n = len(nums)
        if n == 2:
            return max(nums[0], nums[1])
        if n == 1:
            return nums[0]
        if n == 0:
            return 0
    
    
        sum = [0] * n
        
        i = 2
        
        sum[0] = nums[0]
        sum[1] = max(nums[0], nums[1])
        
        
        while i < n:
            sum[i] = max(sum[i - 1], sum[i - 2] + nums[i])
            i += 1
        
    return sum[n - 1]


# min stack, return the minimum element in O(1) time
class MinStack(object):
    def __init__(self):
        """
            initialize your data structure here.
            """
        self.l = list()
        self.min_l = list()
        self.l_length = 0
        self.min_l_length = 0
    
    
    def push(self, x):
        """
            :type x: int
            :rtype: nothing
            """
        self.l.append(x)
        self.l_length += 1
        if not self.min_l:
            self.min_l.append(x)
            self.min_l_length += 1
        elif x <= self.min_l[self.min_l_length - 1]:
            self.min_l.append(x)
            self.min_l_length += 1


    def pop(self):
        """
        :rtype: nothing
        """
        k = self.l.pop()
        self.l_length -= 1
        if k == self.min_l[self.min_l_length - 1]:
           self.min_l.pop()
           self.min_l_length -= 1


    def top(self):
        """
        :rtype: int
        """
        return self.l[self.l_length - 1]
        
        
    def getMin(self):
        """
            :rtype: int
            """
        return self.min_l[self.min_l_length - 1]


# Compute the h-index for a scholar
class Solution(object):
    def hIndex(self, citations):
        """
            :type citations: List[int]
            :rtype: int
            """
        
        citations.sort(reverse = True)
        length = len(citations)
        h_index = 0
        while h_index < length:
            if h_index < citations[h_index]:
                h_index += 1
            else:
                break
        return h_index

# roman to integer
class Solution(object):
    def romanToInt(self, s):
        """
            :type s: str
            :rtype: int
            """
        d = {}
        d['I'] = 1
        d['V'] = 5
        d['X'] = 10
        d['L'] = 50
        d['C'] = 100
        d['D'] = 500
        d['M'] = 1000
        
        length = len(s)
        sum = 0
        for i in xrange(length):
            if i < length - 1 and d.get(s[i]) < d.get(s[i + 1]):
                sum -= d.get(s[i])
            else:
                sum += d.get(s[i])
        return sum

# merge two sorted lists in place
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
            :type nums1: List[int]
            :type m: int
            :type nums2: List[int]
            :type n: int
            :rtype: void Do not return anything, modify nums1 in-place instead.
            """
        j = 0
        for i in xrange(m, m + n):
            nums1[i] = nums2[j]
            j += 1
        nums1.sort()

# calculate the minimum     roof to leave     depth
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def minDepth(self, root):
        """
            :type root: TreeNode
            :rtype: int
            """
        if root == None:
            return 0
        
        if not root.left or not root.right:
            return max(self.minDepth(root.left), self.minDepth(root.right)) + 1
        else:
        return min(self.minDepth(root.left), self.minDepth(root.right)) + 1

# Find the majority element in an array
class Solution(object):
    def majorityElement(self, nums):
        """
            :type nums: List[int]
            :rtype: int
            """
        d = {}
        length = len(nums)
        for num in nums:
            d[num] = d.get(num, 0) + 1
            if d[num] > length / 2:
                return num
        return 0

# Find the longest common prefix in a list of strings
class Solution(object):
    def longestCommonPrefix(self, strs):
        """
            :type strs: List[str]
            :rtype: str
            """
        if strs == []:
            return ""
        l = map(len, strs)
        length = min(l)
        prefix = ""
        for i in xrange(length):
            temp = strs[0][i]
            for s in strs:
                if s[i] == temp:
                    continue
                else:
                    return prefix
            prefix += temp
        return prefix

# find the maximum depth for a binary tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def maxDepth(self, root):
        """
            :type root: TreeNode
            :rtype: int
            """
        if not root:
            return 0
        if not root.left and not root.right:
            return 1
        
        left = 1 + self.maxDepth(root.left)
        right = 1 + self.maxDepth(root.right)
        
        return max(left, right)

# check if two binary trees are the same
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isSameTree(self, p, q):
        """
            :type p: TreeNode
            :type q: TreeNode
            :rtype: bool
            """
        
        if not p and not q:
            return True
        elif p and not q:
            return False
        elif q and not p:
            return False
        
        return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)


# determine if a binary tree is balanced
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isBalanced(self, root):
        """
            :type root: TreeNode
            :rtype: bool
            """
        if root == None:
            return True
        
        left = self.max_depth(root.left)
        right = self.max_depth(root.right)
        if abs(left - right) > 1:
            return False
        return self.isBalanced(root.left) and self.isBalanced(root.right)
    
    def max_depth(self, root):
        if not root:
            return 0
        if not root.left and not root.right:
            return 1
        return max(self.max_depth(root.left) + 1, self.max_depth(root.right) + 1)


# Determine if a binary tree is a mirror of itself
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isSymmetric(self, root):
        """
            :type root: TreeNode
            :rtype: bool
            """
        if not root:
            return True
        return self.sym(root.left, root.right)
    
    def sym(self, p, q):
        if not p and not q:
            return True
        elif p and not q:
            return False
        elif not p and q:
            return False
        elif p.val != q.val:
            return False
        return self.sym(p.left, q.right) and self.sym(q.left, p.right)

# level order traversal, iterative way
from Queue import Queue

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def levelOrder(self, root):
        """
            :type root: TreeNode
            :rtype: List[List[int]]
            """
        l = list()
        if not root:
            return l
        q = Queue()
        q.put(root)
        while not q.empty():
            l1 = list()
            l2 = list()
            while not q.empty():
                node = q.get()
                l1.append(node)
                l2.append(node.val)
            
            for nodes in l1:
                if nodes.left != None:
                    q.put(nodes.left)
                if nodes.right != None:
                    q.put(nodes.right)
            l.append(l2)
        return l

# binary tree view from right
from Queue import Queue
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def rightSideView(self, root):
        """
            :type root: TreeNode
            :rtype: List[int]
            """
        l = list()
        if not root:
            return l
        q = Queue()
        q.put(root)
        while not q.empty():
            l2 = list()
            while not q.empty():
                l2.append(q.get())
            for nodes in l2:
                if nodes.left:
                    q.put(nodes.left)
                if nodes.right:
                    q.put(nodes.right)
            l.append(l2[len(l2) - 1].val)
        return l



# BST iterator
# Definition for a  binary tree node
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class BSTIterator(object):
    def __init__(self, root):
        """
            :type root: TreeNode
            """
        self.l = list()
        pointer = root
        while pointer != None:
            self.l.append(pointer)
            pointer = pointer.left


    def hasNext(self):
    """
        :rtype: bool
        """
            return self.l != []
        
        
        
    def next(self):
        """
            :rtype: int
            """
        curr = self.l.pop()
        if curr.right:
            curr = curr.right
            while curr:
                self.l.append(curr)
                curr = curr.left
        return curr.val


# Your BSTIterator will be called like this:
# i, v = BSTIterator(root), []
# while i.hasNext(): v.append(i.next())


# determine if a binary tree has a path that sums up to sum
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def hasPathSum(self, root, sum):
        """
            :type root: TreeNode
            :type sum: int
            :rtype: bool
            """
        if not root:
            return False
        if root.val == sum and not root.left and not root.right:
            return True
        return self.hasPathSum(root.left, sum - root.val) or  self.hasPathSum(root.right, sum - root.val)


# binary tree pre order traversal
class Solution(object):
    def preorderTraversal(self, root):
        """
            :type root: TreeNode
            :rtype: List[int]
            """
        stack = list()
        answer = list()
        if not root:
            return answer
        stack.append(root)
        while stack != []:
            v = stack.pop()
            if v.right:
                stack.append(v.right)
            if v.left:
                stack.append(v.left)
            answer.append(v.val)
        
        return answer


# binary tree inorder traversal
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def inorderTraversal(self, root):
        """
            :type root: TreeNode
            :rtype: List[int]
            """
        stack = list()
        answer = list()
        s = set()
        if not root:
            return answer
        stack.append(root)
        while stack != []:
            node = stack[-1]
            if node.left and node.left not in s:
                stack.append(node.left)
                s.add(node.left)
            else:
                node = stack.pop()
                if node.right:
                    stack.append(node.right)
                    s.add(node.right)
                answer.append(node.val)
        return answer

# bottom up level order traversal
from Queue import Queue
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def levelOrderBottom(self, root):
        """
            :type root: TreeNode
            :rtype: List[List[int]]
            """
        queue = Queue()
        answer = list()
        if not root:
            return answer
        queue.put(root)
        while not queue.empty():
            
            l = list()
            l1 = list()
            while not queue.empty():
                element = queue.get()
                l.append(element)
                l1.append(element.val)
            for element in l:
                if element.left:
                    queue.put(element.left)
                if element.right:
                    queue.put(element.right)
            answer.append(l1)
        
        answer.reverse()
                return answer

# Given n, how many structurally unique BST's (binary search trees) that store values 1...n?
class Solution(object):
    def numTrees(self, n):
        """
            :type n: int
            :rtype: int
            """
        l = list()
        l.append(1)
        l.append(1)
        
        for i in xrange(2, n + 1):
            sum = 0
            for j in xrange(i):
                sum += l[j] * l[i - j - 1]
            l.append(sum)
        return l[-1]


# return roof to leave path sums up to sum
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def pathSum(self, root, sum):
        """
            :type root: TreeNode
            :type sum: int
            :rtype: List[List[int]]
            """
        l = list()
        if not root:
            return l
        if root and not root.left and not root.right:
            if root.val == sum:
                l.append([root.val])
                return l
            else:
                return l
        list_left = self.pathSum(root.left, sum - root.val)
        if list_left:
            for items in list_left:
                items.insert(0, root.val)
                l.append(items)
        list_right = self.pathSum(root.right, sum - root.val)
        if list_right:
            for items in list_right:
                items.insert(0, root.val)
                l.append(items)
        return l

# Given a 2d grid map of '1's (land) and '0's (water), count the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically.
# You may assume all four edges of the grid are all surrounded by water.


class Solution(object):
    
    def numIslands(self, grid):
        """
            :type grid: List[List[str]]
            :rtype: int
        """
        if grid == []:
            return 0
        
        number = 0
        lengthx = len(grid)
        lengthy = len(grid[0])
        visited = [[False for x in xrange(lengthy)] for x in xrange(lengthx)]
        for i in xrange(lengthx):
            for j in xrange(lengthy):
                if visited[i][j] == False and grid[i][j] == "1":
                    number += 1
                    self.helper_func(grid, i, j, lengthx, lengthy, visited)
        
        return number

    def helper_func(self, grid, i, j, lengthx, lengthy, visited):
        if i >= 0 and i < lengthx and j >= 0 and j < lengthy and not visited[i][j] and grid[i][j] == "1":
            visited[i][j] = True
            self.helper_func(grid, i + 1, j, lengthx, lengthy, visited)
            self.helper_func(grid, i, j + 1, lengthx, lengthy, visited)
            self.helper_func(grid, i - 1, j, lengthx, lengthy, visited)
            self.helper_func(grid, i, j - 1, lengthx, lengthy, visited)

#Given a binary tree
#    
#    struct TreeLinkNode {
#        TreeLinkNode *left;
#            TreeLinkNode *right;
#                TreeLinkNode *next;
#}
#Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.
#
#Initially, all next pointers are set to NULL.

class Solution(object):
    def connect(self, root):
        """
            :type root: TreeLinkNode
            :rtype: nothing
            """
        if not root:
            return
        if not root.left or not root.right:
            return
        if root.left and root.right:
            root.left.next = root.right
        if root.next:
            root.right.next = root.next.left
        self.connect(root.left)
        self.connect(root.right)



# binary tree recover from inorder and postorder
class Solution(object):
    def buildTree(self, inorder, postorder):
        """
            :type inorder: List[int]
            :type postorder: List[int]
            :rtype: TreeNode
            """
        if inorder == [] or postorder == []:
            return None
        
        root = TreeNode(postorder.pop())
        index = inorder.index(root.val)
        root.right = self.buildTree(inorder[index + 1:], postorder)
        root.left = self.buildTree(inorder[:index], postorder)
        
        return root


# recover tree inorder and preorder
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def buildTree(self, preorder, inorder):
        """
            :type inorder: List[int]
            :type postorder: List[int]
            :rtype: TreeNode
            """
        if inorder == [] or preorder == []:
            return None
        
        root = TreeNode(preorder.pop(0))
        index = inorder.index(root.val)
        root.right = self.buildTree(preorder[index:], inorder[index + 1:])
        root.left = self.buildTree(preorder, inorder[:index])
        
    return root


# Given a binary tree, flatten it to a linked list in-place.
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def flatten(self, root):
        """
            :type root: TreeNode
            :rtype: void Do not return anything, modify root in-place instead.
            """
        if not root:
            return
        self.flatten_helper(root)
    
    def flatten_helper(self, root):
        if not root.left and not root.right:
            return root
        left, right = root.left, root.right
        remaining = root
        if left:
            root.left = None
            root.right = left
            remaining = self.flatten_helper(left)
        if right:
            remaining.right = right
            remaining = self.flatten_helper(right)
        
        return remaining

# implement prefix tree
class TrieNode(object):
    def __init__(self):
        """
            Initialize your data structure here.
            """
        self.children = {}
        self.exist = False


class Trie(object):
    
    def __init__(self):
        self.root = TrieNode()
        self.root.exist = True
    
    
    def insert(self, word):
        """
            Inserts a word into the trie.
            :type word: str
            :rtype: void
            """
        pointer = self.root
        for i in word:
            if i in pointer.children:
                pointer = pointer.children[i]
            else:
                pointer.children[i] = TrieNode()
                pointer = pointer.children[i]
        pointer.exist = True
    
    
    def search(self, word):
        """
            Returns if the word is in the trie.
            :type word: str
            :rtype: bool
            """
        pointer = self.root
        for i in word:
            if i in pointer.children:
                pointer = pointer.children[i]
            else:
                return False
        return pointer.exist
    
    def startsWith(self, prefix):
        """
            Returns if there is any word in the trie
            that starts with the given prefix.
            :type prefix: str
            :rtype: bool
            """
        pointer = self.root
        for i in prefix:
            if i not in pointer.children:
                return False
            pointer = pointer.children[i]
        return True


# Your Trie object will be instantiated and called as such:
# trie = Trie()
# trie.insert("somestring")
# trie.search("key")

#Design a data structure that supports the following two operations:
#
#void addWord(word)
#bool search(word)
#search(word) can search a literal word or a regular expression string containing only letters a-z or .. A . means it can represent any one letter.




class TrieNode():
    def __init__(self):
        self.d = {}
            self.exist = False

class WordDictionary(object):
    
    def __init__(self):
        """
            initialize your data structure here.
            """
        self.root = TrieNode()
        self.root.exist = True
    
    
    def addWord(self, word):
        """
            Adds a word into the data structure.
            :type word: str
            :rtype: void
            """
        pointer = self.root
        for i in word:
            if i in pointer.d:
                pointer = pointer.d[i]
            else:
                pointer.d[i] = TrieNode()
                pointer = pointer.d[i]
        pointer.exist = True
    
    
    def search(self, word):
        """
            Returns if the word is in the data structure. A word could
            contain the dot character '.' to represent any one letter.
            :type word: str
            :rtype: bool
            """
        pointer = self.root
        start = 0
        end = len(word)
        return self.helper_func(word, pointer, start, end)
    
    def helper_func(self, word, pointer, start, end):
        if start == end:
            return pointer.exist
        if word[start] != '.':
            if word[start] in pointer.d:
                return self.helper_func(word, pointer.d[word[start]], start + 1, end)
            else:
                return False
        # apply dfs to every element in the dictionary
        else:
            for key, value in pointer.d.items():
                if self.helper_func(word, value, start + 1, end):
                    return True
        return False



# time out for some reason
#Given a positive integer n, find the least number of perfect square numbers (for example, 1, 4, 9, 16, ...) which sum to n.
#
#For example, given n = 12, return 3 because 12 = 4 + 4 + 4; given n = 13, return 2 because 13 = 4 + 9.
class Solution(object):
    def numSquares(self, n):
        """
            :type n: int
            :rtype: int
            """
        l = list()
        l.append(0)
        for i in xrange(1, n + 1):
            min_num = sys.maxint
            j = 1
            while j * j <= i:
                min_num = min(min_num, l[i - j * j] + 1)
                j += 1
            l.append(min_num)
        
        return l[n]

#
#Find the contiguous subarray within an array (containing at least one number) which has the largest sum.
#
#For example, given the array [−2,1,−3,4,−1,2,1,−5,4],
#the contiguous subarray [4,−1,2,1] has the largest sum = 6.


class Solution(object):
    def maxSubArray(self, nums):
        """
            :type nums: List[int]
            :rtype: int
            """
        length = len(nums)
        i = 1
        sum = nums[0]
        greatest_sum = sum
        while i < length:
            sum = max(nums[i], nums[i] + sum)
            greatest_sum = max(greatest_sum, sum)
            i += 1
        return greatest_sum




#There are a total of n courses you have to take, labeled from 0 to n - 1.
#
#Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]
#
#Given the total number of courses and a list of prerequisite pairs, is it possible for you to finish all courses?


class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
            :type numCourses: int
            :type prerequisites: List[List[int]]
            :rtype: bool
            """
        if prerequisites == []:
            return True
        d = dict()
        visited = [False for i in xrange(numCourses)]
        for l in prerequisites:
            pre = l[1]
            course = l[0]
            if pre in d:
                d[pre].append(course)
            else:
                d[pre] = [course]
        for i in xrange(numCourses):
            pre = [False for j in xrange(numCourses)]
            if self.cycle(i, d, visited, pre):
                return False
        return True
    
    
    def cycle(self, courseNum, d, visited, pre):
        if visited[courseNum]:
            return False
        if courseNum not in d:
            return False
        
        if pre[courseNum]:
            return True
        pre[courseNum] = True
        
        
        for i in d[courseNum]:
            if self.cycle(i, d, visited, pre):
                return True
        
            visited[courseNum] = True
            
                return False


#Follow up for N-Queens problem.
#
#Now, instead outputting board configurations, return the total number of distinct solutions.

class Solution(object):
    def totalNQueens(self, n):
        """
            :type n: int
            :rtype: int
            """
        if n < 1:
            return 0
        position = [-1 for i in xrange(n)]
        start = 0
        number = 0
        return self.helper_func(n, position, start, number)

    def helper_func(self, n, position, start, number):
        result = 0
        if start >= n:
            return 1

        for i in xrange(n):
            available = True
            for j in xrange(number):
                if (position[j] == i) or (abs(position[j] - i) == abs(j - start)):
                    available = False
                    break
        if available:
            position[start] = i
                result += self.helper_func(n, position, start + 1, number + 1)
    return result



#Given a 2D board and a word, find if the word exists in the grid.
#
#The word can be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once.

class Solution(object):
    def exist(self, board, word):
        """
            :type board: List[List[str]]
            :type word: str
            :rtype: bool
            """
        length = len(board)
        
        if length == 0:
            return len(word) == 0
        
        l = len(board[0])
        
        visited = [[False for i in xrange(l)] for i in xrange(length)]
        
        for i in xrange(length):
            for j in xrange(l):
                if self.dfs(i, j, length, word, board, l, visited):
                    return True
            
    return False


    def dfs(self, i, j, length, word, board, l, visited):
        if len(word) == 0:
            return True
        
        if i < 0 or i >= length or j < 0 or j >= l:
            return False
    
        if visited[i][j]:
            return False
        
        
        if board[i][j] != word[0]:
            return False
    
        visited[i][j] = True
        
        result = self.dfs(i - 1, j, length, word[1:], board, l, visited)
        if result:
            return True
        result = result or self.dfs(i + 1, j, length, word[1:], board, l, visited)
        if result:
        return True
        result = result or self.dfs(i, j + 1, length, word[1:], board, l, visited)
        if result:
            return True
        result = result or self.dfs(i, j - 1, length, word[1:], board, l, visited)
        
        
        visited[i][j] = False
        
        return result



# Given a string S, find the longest palindromic substring in S. You may assume that the maximum length of S is 1000, and there exists one unique longest palindromic substring.
# when find a single character, search left and right to find if there exists a palindrome substring
class Solution(object):
    def longestPalindrome(self, s):
        """
            :type s: str
            :rtype: str
            """
        length = len(s)
        max_length = 0
        result = str()
        for i in xrange(length):
            if self.is_palindrome(s, i - max_length - 1, i):
                result = s[i - max_length - 1: i + 1]
                max_length += 2
            elif self.is_palindrome(s, i - max_length, i):
                result = s[i - max_length : i + 1]
                max_length += 1
        
    return result

    def is_palindrome(self, s, start, end):
        if start < 0:
            return False
        
        while start < end:
            if s[start] == s[end]:
                start += 1
                end -= 1
            else:
                return False
        return True



# check if the parenthesis is valid
class Solution(object):
    def isValid(self, s):
        """
            :type s: str
            :rtype: bool
            """
        stack = list()
        for char in s:
            if char in ['(', '{', '[']:
                stack.append(char)
            else:
                if len(stack) != 0:
                    c = stack.pop()
                    a = str(c) + char
                    if a not in ["()", "{}", "[]"]:
                        return False
            
                else:
                    return False

        return len(stack) == 0


# find all subsets of a given set

class Solution(object):
    def subsets(self, nums):
        """
            :type nums: List[int]
            :rtype: List[List[int]]
            """
        l = list()
        answer = list()
        answer.append(l)
        nums.sort()
        if not nums:
            return answer
        length = len(nums)
        for i in xrange(1, length + 1):
            # return value, subset list, total length, start position
            self.get_subset(answer, l, nums, i, 0)
        
        return answer
    
    def get_subset(self, answer, l, nums, i, position):
        if len(l) == i:
            answer.append(copy.deepcopy(l))
            return
        
        length = len(nums)
        for k in xrange(position, length):
            l.append(nums[k])
            self.get_subset(answer, l, nums, i, k + 1)
            l.pop()
        
        return