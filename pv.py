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
#        TreeLinkNode *right;
#        TreeLinkNode *next;
#   }
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

# n queens print the board
class Solution(object):
    def solveNQueens(self, n):
        """
            :type n: int
            :rtype: List[List[str]]
            """
        result = list()
        path = list()
        nums = [-1] * n
        self.helper_func(result, nums, 0, path)
        return result
    
    def helper_func(self, position, nums, index, path):
        length = len(nums)
        if length == index:
            position.append(path)
            return
        for i in xrange(length):
            nums[index] = i
            if self.isValid(nums, index):
                temp = '.' * length
                self.helper_func(position, nums, index + 1, path + [temp[:i] + 'Q' + temp[i + 1:]])


def isValid(self, nums, index):
    for i in xrange(index):
        if abs(index - i) == abs(nums[index] - nums[i]) or nums[index] == nums[i]:
            return False
        return True

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


#Given a collection of candidate numbers (C) and a target number (T), find all unique combinations in C where the candidate numbers sums to T.
#
#Each number in C may only be used once in the combination.
class Solution(object):
    def combinationSum2(self, candidates, target):
        """
            :type candidates: List[int]
            :type target: int
            :rtype: List[List[int]]
            """
        
        candidates.sort()
        answer = list()
        single_answer = list()
        self.find_answer(answer, single_answer, candidates, target)
        
        return answer
    
    def find_answer(self, answer, single_answer, candidates, target):
        if target == 0:
            answer.append(single_answer[:])
            return
    
        length = len(candidates)
        
        for i in xrange(length):
            if candidates[i] > target:
                return
            if i == 0 or candidates[i] != candidates[i - 1]:
                self.find_answer(answer, single_answer + [candidates[i]], candidates[i + 1:], target - candidates[i])

        return

# Given two integers n and k, return all possible combinations of k numbers out of 1 ... n.

class Solution(object):
    def combine(self, n, k):
        """
            :type n: int
            :type k: int
            :rtype: List[List[int]]
            """
        l = list()
        answer = list()
        self.helper_func(1, n, k, l, answer)
        return answer
    
    def helper_func(self, start, n, k, l, answer):
        if k == 0:
            answer.append(l[:])
            return
        
        for i in xrange(start, n + 1):
            l.append(i)
            self.helper_func(i + 1, n, k - 1, l, answer)
            l.pop()


# implement LRU
class LRUCache(object):
    
    def __init__(self, capacity):
        """
            :type capacity: int
            """
        self.d = collections.OrderedDict()
        self.rest = capacity
    
    
    def get(self, key):
        """
            :rtype: int
            """
        if key not in self.d:
            return -1
        element = self.d[key]
        del self.d[key]
        self.d[key] = element
        return element
    
    
    def set(self, key, value):
        """
            :type key: int
            :type value: int
            :rtype: nothing
            """
        if key in self.d:
            del self.d[key]
            self.d[key] = value
            return
        if self.rest > 0:
            self.d[key] = value
            self.rest -= 1
        else:
            self.d.popitem(last=False)
            self.d[key] = value

#A strobogrammatic number is a number that looks the same when rotated 180 degrees (looked at upside down).
#
#Write a function to determine if a number is strobogrammatic. The number is represented as a string.
#
#For example, the numbers "69", "88", and "818" are all strobogrammatic.
class Solution(object):
    def isStrobogrammatic(self, num):
        """
            :type num: str
            :rtype: bool
            """
        strobogramPair = dict()
        strobogramPair['6'] = '9'
        strobogramPair['9'] = '6'
        strobogramPair['8'] = '8'
        strobogramPair['0'] = '0'
        strobogramPair['1'] = '1'
        
        length = len(str(num))
        string = str(num)
        for i in xrange(length):
            if string[i] not in ['6', '9', '8', '0', '1']:
                return False
            if string[length - i - 1] not in ['6', '9', '8', '0', '1']:
                return False
            if strobogramPair[string[i]] != string[length - i - 1]:
                return False
        return True

#An abbreviation of a word follows the form <first letter><number><last letter>. Below are some examples of word abbreviations:
#
#a) it                      --> it    (no abbreviation)
#
#1
#b) d|o|g                   --> d1g
#    
#    1    1  1
#        1---5----0----5--8
#c) i|nternationalizatio|n  --> i18n
#
#1
#    1---5----0
#d) l|ocalizatio|n          --> l10n
#Assume you have a dictionary and given a word, find whether its abbreviation is unique in the dictionary.
#A word's abbreviation is unique if no other word from the dictionary has the same abbreviation.

class ValidWordAbbr(object):
    def __init__(self, dictionary):
        """
            initialize your data structure here.
            :type dictionary: List[str]
            """
        self.d = dict()
        self.wordList = dictionary[:]
        for word in dictionary:
            length = len(word)
            abbr = word[0] + str(length) + word[-1]
            if abbr in self.d:
                self.d[abbr].append(word)
            else:
                self.d[abbr] = [word]


    def isUnique(self, word):
        """
            check if a word is unique.
            :type word: str
            :rtype: bool
        """
        length = len(word)
        abbr = word[0] + str(length) + word[-1]
        if abbr not in self.d:
            return True
        else:
            if len(self.d[abbr]) != 1:
                return False
            if word not in self.wordList:
                return False
            return True



# Your ValidWordAbbr object will be instantiated and called as such:
# vwa = ValidWordAbbr(dictionary)
# vwa.isUnique("word")
# vwa.isUnique("anotherWord")

#
#There is a fence with n posts, each post can be painted with one of the k colors.
#
#You have to paint all the posts such that no more than two adjacent fence posts have the same color.
#
#Return the total number of ways you can paint the fence.
#

class Solution(object):
    def numWays(self, n, k):
        """
            :type n: int
            :type k: int
            :rtype: int
            """
        if n == 0 or k == 0:
            return 0
        if n == 1:
            return k
        same = k
        different = k * (k - 1)
        for i in xrange(3, n + 1):
            same, different = different, (same + different) * (k - 1)
        return same + different


# runs slowly
# find the closest value in binary tree
class Solution(object):
    def closestValue(self, root, target):
        """
            :type root: TreeNode
            :type target: float
            :rtype: int
            """
        d = dict()
        smallest = abs(root.val - target)
        smallest = self.helper_func(root, target, d, smallest)
        return d[smallest].val
    
    
    
    def helper_func(self, root, target, d, smallest):
        value = abs(root.val - target)
        d[value] = root
        if value < smallest:
            smallest = value
        if root.left:
            smallest = self.helper_func(root.left, target, d, smallest)
        if root.right:
            smallest = self.helper_func(root.right, target, d, smallest)
        return smallest


# if the permutation can be a palindrome
class Solution(object):
    def canPermutePalindrome(self, s):
        """
            :type s: str
            :rtype: bool
            """
        d = dict()
        for char in s:
            d[char] = d.get(char, 0) + 1
        count_odd = 0
        for key, value in d.items():
            if value % 2:
                count_odd += 1
            if count_odd >= 2:
                return False
        return True

# if word can be broken up into words in dictionary
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
            :type s: str
            :type wordDict: Set[str]
            :rtype: bool
            """
        if not s:
            return True
        length = len(s)
        words = [False for i in xrange(length)]
        for i in reversed(xrange(length)):
            if s[i:] in wordDict:
                words[i] = True
                continue
            for j in xrange(i + 1, length):
                if s[i:j] in wordDict and words[j]:
                    words[i] = True
                    break
        return words[0]

#Given an array of n integers nums and a target, find the number of index triplets i, j, k with 0 <= i < j < k < n that satisfy the condition nums[i] + nums[j] + nums[k] < target.
#
#For example, given nums = [-2, 0, 1, 3], and target = 2.
#
#Return 2.

class Solution(object):
    def threeSumSmaller(self, nums, target):
        """
            :type nums: List[int]
            :type target: int
            :rtype: int
            """
        nums.sort()
        length = len(nums)
        if length < 3:
            return 0
        count = 0
        for i in xrange(length):
            start = i + 1
            end = length - 1
            while start < end:
                if nums[i] + nums[start] + nums[end] < target:
                    count += end - start
                    start += 1
                else:
                    end -= 1

        return count



# Given an unsorted array nums, reorder it in-place such that nums[0] <= nums[1] >= nums[2] <= nums[3]....
class Solution(object):
    def wiggleSort(self, nums):
        """
            :type nums: List[int]
            :rtype: void Do not return anything, modify nums in-place instead.
            """
        length = len(nums)
        if length == 0 or length == 1:
            return
        for i in xrange(1, length, 2):
            if nums[i - 1] > nums[i]:
                nums[i - 1], nums[i] = nums[i], nums[i - 1]
            
            if i + 1 < length and nums[i] < nums[i + 1]:
                nums[i + 1], nums[i] = nums[i], nums[i + 1]




# kth smallest element in a binary search tree
class Solution(object):
    def kthSmallest(self, root, k):
        """
            :type root: TreeNode
            :type k: int
            :rtype: int
            """
        l = list()
        pointer = root
        while l or pointer:
            if pointer:
                l.append(pointer)
                pointer = pointer.left
            else:
                pointer = l.pop()
                if k > 1:
                    k -= 1
                else:
                    return pointer.val
                pointer = pointer.right



#Given a board with m by n cells, each cell has an initial state live (1) or dead (0). Each cell interacts with its eight neighbors (horizontal, vertical, diagonal) using the following four rules (taken from the above Wikipedia article):
#
#Any live cell with fewer than two live neighbors dies, as if caused by under-population.
#Any live cell with two or three live neighbors lives on to the next generation.
#Any live cell with more than three live neighbors dies, as if by over-population..
#Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.
#Write a function to compute the next state (after one update) of the board given its current state.

class Solution(object):
    def gameOfLife(self, board):
        """
            :type board: List[List[int]]
            :rtype: void Do not return anything, modify board in-place instead.
            """
        m = len(board)
        n = len(board[0])
        for i in xrange(m):
            for j in xrange(n):
                alive = 0
                for p in xrange(-1, 2):
                    for q in xrange(-1, 2):
                        if self.checkAlive(board, p, q, i, j):
                            alive += 1
                alive -= board[i][j]
                if alive < 2 and board[i][j]:
                    board[i][j] |= 2
                elif (alive == 2 or alive == 3) and board[i][j]:
                    continue
                elif alive > 3 and board[i][j]:
                    board[i][j] |= 2
                elif alive == 3 and not board[i][j]:
                    board[i][j] |= 2
    
        for i in xrange(m):
            for j in xrange(n):
                if board[i][j] == 3:
                    board[i][j] = 0
                elif board[i][j] == 2:
                    board[i][j] = 1


    def checkAlive(self, board, p, q, i, j):
        m = len(board)
        n = len(board[0])
        if i + p >= 0 and i + p < m and q + j >= 0 and q + j < n and board[i + p][j + q] & 1:
            return True


#Given an Iterator class interface with methods: next() and hasNext(), design and implement a PeekingIterator that support the peek() operation -- it essentially peek() at the element that will be returned by the next call to next().
#
#Here is an example. Assume that the iterator is initialized to the beginning of the list: [1, 2, 3].
#
#Call next() gets you 1, the first element in the list.
#
#Now you call peek() and it returns 2, the next element. Calling next() after that still return 2.
#
#You call next() the final time and it returns 3, the last element. Calling hasNext() after that should return false.


# Below is the interface for Iterator, which is already defined for you.
#
# class Iterator(object):
#     def __init__(self, nums):
#         """
#         Initializes an iterator object to the beginning of a list.
#         :type nums: List[int]
#         """
#
#     def hasNext(self):
#         """
#         Returns true if the iteration has more elements.
#         :rtype: bool
#         """
#
#     def next(self):
#         """
#         Returns the next element in the iteration.
#         :rtype: int
#         """

class PeekingIterator(object):
    def __init__(self, iterator):
        """
            Initialize your data structure here.
            :type iterator: Iterator
            """
        self.l = list()
        self.iter = iterator
    
    def peek(self):
        """
            Returns the next element in the iteration without advancing the iterator.
            :rtype: int
            """
        if not self.l:
            self.l.append(self.iter.next())
        return self.l[0]
    
    
    
    def next(self):
        """
            :rtype: int
            """
        if self.l:
            return self.l.pop()
        else:
            return self.iter.next()


    def hasNext(self):
    """
        :rtype: bool
        """
            if self.l:
            return True
                return self.iter.hasNext()


# Your PeekingIterator object will be instantiated and called as such:
# iter = PeekingIterator(Iterator(nums))
# while iter.hasNext():
#     val = iter.peek()   # Get the next element but not advance the iterator.
#     iter.next()         # Should return the same value as [val].



#A peak element is an element that is greater than its neighbors.
#
#Given an input array where num[i] ≠ num[i+1], find a peak element and return its index.
#
#The array may contain multiple peaks, in that case return the index to any one of the peaks is fine.
#
#You may imagine that num[-1] = num[n] = -∞.
#
#For example, in array [1, 2, 3, 1], 3 is a peak element and your function should return the index number 2.

# in o(nlogn) time, divide and conquer or binary search, can also find the maximum element in an array

class Solution(object):
    def findPeakElement(self, nums):
        """
            :type nums: List[int]
            :rtype: int
            """
        start = 0
        end = len(nums) - 1
        return self.helper_func(nums, start, end)
    
    def helper_func(self, nums, start, end):
        middle = (start + end) / 2
        if middle < len(nums) - 1 and nums[middle] < nums[middle + 1]:
            return self.helper_func(nums, middle + 1, end)
        elif middle > 0 and nums[middle] < nums[middle - 1]:
            return self.helper_func(nums, start, middle - 1)
        return middle


#Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
#
#For example, given n = 3, a solution set is:
#
#"((()))", "(()())", "(())()", "()(())", "()()()"

class Solution(object):
    def generateParenthesis(self, n):
        """
            :type n: int
            :rtype: List[str]
            """
        result = str()
        value = list()
        self.helper_func(0, 0, n, result, value)
        return value
    
    def helper_func(self, count_open, count_close, total, result, value):
        if count_open < count_close:
            return
        if count_close == total and count_open == total:
            value.append(result)
            return
        
        if count_open == total:
            for i in xrange(total - count_close):
                result += ')'
            value.append(result)
            return
        
        self.helper_func(count_open + 1, count_close, total, result + '(', value)
        self.helper_func(count_open, count_close + 1, total, result + ')', value)


#Given a sorted integer array where the range of elements are [lower, upper] inclusive, return its missing ranges.
#
#For example, given [0, 1, 3, 50, 75], lower = 0 and upper = 99, return ["2", "4->49", "51->74", "76->99"].


class Solution(object):
    def findMissingRanges(self, nums, lower, upper):
        """
            :type nums: List[int]
            :type lower: int
            :type upper: int
            :rtype: List[str]
            """
        l = list()
        
        
        
        if upper == lower and nums == []:
            l.append(str(upper))
            return l
        if nums == []:
            l.append(str(lower) + "->" + str(upper))
            return l
        
        temp = self.makeStart(lower, nums[0])
        if temp:
            l.append(temp)
        nums.append(upper + 1)
        length = len(nums)
        
        for i in xrange(length - 1):
            result = self.makeString(nums[i], nums[i + 1])
            if result:
                l.append(result)
        return l
    
    def makeString(self, lower, upper):
        if upper - lower == 1 or upper == lower:
            return
        if upper - lower == 2:
            return str(lower + 1)
        return str(lower + 1) + "->" + str(upper - 1)
    
    def makeStart(self, lower, upper):
        lower = min(lower, upper)
        upper = max(lower, upper)
        if lower == upper:
            return
        if lower == upper - 1:
            return str(lower)
        else:
            return str(lower) + "->" + str(upper - 1)



# wildcard matching, slower solution
class Solution(object):
    def isMatch(self, s, p):
        """
            :type s: str
            :type p: str
            :rtype: bool
            """
        length_p = len(p)
        length_s = len(s)
        exist = [[False for i in xrange(length_p + 1)] for j in xrange(length_s + 1)]
        exist[length_s][length_p] = True
        for i in reversed(xrange(length_p)):
            if p[i] == '*':
                exist[length_s][i] = True
            else:
                break
    
        for i in reversed(xrange(length_s - 1)):
            for j in reversed(xrange(length_p - 1)):
                if s[i] == p[j] or p[j] == '?':
                    exist[i][j] = exist[i + 1][j + 1]
                elif p[j] == '*':
                    exist[i][j] = exist[i][j + 1] or exist[i + 1][j]

    return exist[0][0]

# c++ faster solution
#class Solution {
#public:
#    bool isMatch(string s, string p) {
#        int  slen = s.size(), plen = p.size(), i, j, iStar=-1, jStar=-1;
#        
#        for(i=0,j=0 ; i<slen; ++i, ++j)
#        {
#            if(p[j]=='*')
#            { //meet a new '*', update traceback i/j info
#                iStar = i;
#                jStar = j;
#                --i;
#            }
#            else
#            {
#                if(p[j]!=s[i] && p[j]!='?')
#                {  // mismatch happens
#                    if(iStar >=0)
#                    { // met a '*' before, then do traceback
#                        i = iStar++;
#                        j = jStar;
#                    }
#                    else return false; // otherwise fail
#            }
#    }
#        }
#        while(p[j]=='*') ++j;
#        return j==plen;
#    }
#};

#class Solution {
#private:
#    bool helper(const string &s, const string &p, int si, int pi, int &recLevel)
#    {
#        int sSize = s.size(), pSize = p.size(), i, curLevel = recLevel;
#        bool first=true;
#        while(si<sSize && (p[pi]==s[si] || p[pi]=='?')) {++pi; ++si;} //match as many as possible
#        if(pi == pSize) return si == sSize; // if p reaches the end, return
#        if(p[pi]=='*')
#        { // if a star is met
#            while(p[++pi]=='*'); //skip all the following stars
#            if(pi>=pSize) return true; // if the rest of p are all star, return true
#            for(i=si; i<sSize; ++i)
#            {   // then do recursion
#                if(p[pi]!= '?' && p[pi]!=s[i]) continue;
#                if(first) {++recLevel; first = false;}
#                if(helper(s, p, i, pi, recLevel)) return true;
#                if(recLevel>curLevel+1) return false; // if the currently processed star is not the last one, return
#            }
#        }
#            return false;
#    }
#    public:
#        bool isMatch(string s, string p) {
#            int recLevel = 0;
#            return helper(s, p, 0, 0, recLevel);
#    }
#};

#Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:
#
#Integers in each row are sorted in ascending from left to right.
#Integers in each column are sorted in ascending from top to bottom.
#For example,
#
#Consider the following matrix:
#
#[
# [1,   4,  7, 11, 15],
# [2,   5,  8, 12, 19],
# [3,   6,  9, 16, 22],
# [10, 13, 14, 17, 24],
# [18, 21, 23, 26, 30]
# ]
#Given target = 5, return true.
#
#Given target = 20, return false.

class Solution(object):
    def searchMatrix(self, matrix, target):
        """
            :type matrix: List[List[int]]
            :type target: int
            :rtype: bool
            """
        m = len(matrix)
        n = len(matrix[0])
        for i in xrange(m):
            if matrix[i][0] > target:
                return False
            else:
                start = 0
                end = n - 1
                if matrix[i][start] == target or matrix[i][end] == target:
                    return True
                while start < end - 1:
                    middle = (start + end) / 2
                    if matrix[i][middle] == target:
                        return True
                    elif matrix[i][middle] < target:
                        start = middle
                    elif matrix[i][middle] > target:
                        end = middle


    return False

# Design an algorithm to encode a list of strings to a string. The encoded string is then sent over the network and is decoded back to the original list of strings.

class Codec:
    
    def encode(self, strs):
        """Encodes a list of strings to a single string.
            
            :type strs: List[str]
            :rtype: str
            """
        result = ''.join('%d:'%len(strings) + strings for strings in strs)
        return result
    
    
    def decode(self, s):
        """Decodes a single string to a list of strings.
            
            :type s: str
            :rtype: List[str]
            """
        length = len(s)
        i = 0
        result = list()
        while i < length:
            value = s.find(':', i)
            i = int(s[i:value]) + 1 + value
            result.append(s[value + 1: i])
        return result


#Implement an iterator to flatten a 2d vector.
#
#For example,
#Given 2d vector =
#
#[
# [1,2],
# [3],
# [4,5,6]
# ]
#By calling next repeatedly until hasNext returns false, the order of elements returned by next should be: [1,2,3,4,5,6].


class Vector2D(object):
    
    def __init__(self, vec2d):
        """
            Initialize your data structure here.
            :type vec2d: List[List[int]]
            """
        
        self.vector = list()
        for vec1 in vec2d:
            if vec1 == []:
                continue
            self.vector.append(vec1)
        self.m = len(self.vector)
        self.n = 0
        if self.m != 0:
            self.n = len(self.vector[0])
        self.current_x = 0
        self.current_y = 0
    
    
    def next(self):
        """
            :rtype: int
            """
        self.n = len(self.vector[self.current_x])
        value = self.vector[self.current_x][self.current_y]
        if self.current_y == self.n - 1:
            self.current_x += 1
            if self.current_x < self.m:
                self.n = len(self.vector[self.current_x])
            self.current_y = 0
        else:
            self.current_y += 1
        return value
    
    
    
    
    def hasNext(self):
        """
            :rtype: bool
            """
        return self.current_x < self.m and self.current_y < self.n


# Your Vector2D object will be instantiated and called as such:
# i, v = Vector2D(vec2d), []
# while i.hasNext(): v.append(i.next())

#A strobogrammatic number is a number that looks the same when rotated 180 degrees (looked at upside down).
#
#Find all strobogrammatic numbers that are of length = n.
#
#For example,
#Given n = 2, return ["11","69","88","96"].

# too slow
# answer online

#def findStrobogrammatic(self, n):
#    self.ans = []
#    self.findStr(n, "", "")
#    return self.ans
#
#def findStr(self, n, str1, str2):
#    if n == 0:
#        self.ans.append(str1 + str2)
#        return
#    if n == 1:
#        self.findStr(0, str1 + '0', str2)
#        self.findStr(0, str1 + '1', str2)
#        self.findStr(0, str1 + '8', str2)
#    else:
#        if str1 and str1[0] != '0':
#            self.findStr(n-2, str1 + '0', '0' + str2)
#        self.findStr(n-2, str1 + '1', '1' + str2)
#        self.findStr(n-2, str1 + '6', '9' + str2)
#        self.findStr(n-2, str1 + '8', '8' + str2)
#        self.findStr(n-2, str1 + '9', '6' + str2)

class Solution(object):
    def findStrobogrammatic(self, n):
        """
            :type n: int
            :rtype: List[str]
            """
        d = dict()
        d['1'] = '1'
        d['6'] = '9'
        d['9'] = '6'
        d['0'] = '0'
        d['8'] = '8'
        
        l = list()
        result = list()
        self.helper_func(l, d, n, result)
        return l
    
    def helper_func(self, l, d, end, result):
        if len(result) == end:
            if result[0] == '0':
                return
            test = "".join(char for char in result)
            if test not in l:
                l.append(test)
            return
        if end - len(result) == 1:
            if result and result[0] == '0':
                return
            length = len(result)
            result1 = result[:]
            result1.insert(length / 2, '0')
            l.append(''.join(char for char in result1))
            result3 = result[:]
            result3.insert(length / 2, '1')
            l.append("".join(char for char in result3))
            result2 = result[:]
            result2.insert(length / 2, '8')
            l.append("".join(char for char in result2))
            
            return
        
        
        for key, value in d.items():
            result.insert(0, key)
            result.append(value)
            self.helper_func(l, d, end, result)
            result.pop(0)
            result.pop()

#Given two 1d vectors, implement an iterator to return their elements alternately.
#
#For example, given two 1d vectors:
#
#v1 = [1, 2]
#v2 = [3, 4, 5, 6]
#By calling next repeatedly until hasNext returns false, the order of elements returned by next should be: [1, 3, 2, 4, 5, 6].

class ZigzagIterator(object):
    
    def __init__(self, v1, v2):
        """
            Initialize your data structure here.
            :type v1: List[int]
            :type v2: List[int]
            """
        self.v1 = v1
        self.v2 = v2
        self.lengthV1 = len(v1)
        self.lengthV2 = len(v2)
        self.longer = len(v1) > len(v2)
        self.counter = 0
    
    
    def next(self):
        """
            :rtype: int
            """
        value = 0
        if self.counter / 2 < min(self.lengthV1, self.lengthV2):
            if self.counter % 2 == 0:
                value = self.v1[self.counter / 2]
            else:
                value = self.v2[self.counter / 2]
            self.counter += 1
        else:
            count = self.counter - min(self.lengthV1, self.lengthV2)
            self.counter += 1
            if self.longer:
                return self.v1[count]
            else:
                return self.v2[count]
        return value
    
    
    def hasNext(self):
        """
            :rtype: bool
            """
        return self.counter < self.lengthV1 + self.lengthV2


# Your ZigzagIterator object will be instantiated and called as such:
# i, v = ZigzagIterator(v1, v2), []
# while i.hasNext(): v.append(i.next())

#Given n nodes labeled from 0 to n - 1 and a list of undirected edges (each edge is a pair of nodes), write a function to check whether these edges make up a valid tree.
#
#For example:
#
#Given n = 5 and edges = [[0, 1], [0, 2], [0, 3], [1, 4]], return true.
#
#Given n = 5 and edges = [[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]], return false.

class Solution(object):
    def validTree(self, n, edges):
        """
            :type n: int
            :type edges: List[List[int]]
            :rtype: bool
            """
        l = [-1 for i in range(n)]
        s = set()
        for i,j in edges:
            x = self.find(l, i)
            y = self.find(l, j)
            if x == y:
                return False
            self.union(l, x, y)
        return len(edges) == n - 1
    
    
    def find(self, l, i):
        if l[i] == -1:
            return i
        return self.find(l, l[i])
    
    def union(self, parent, i, j):
        xset = self.find(parent, i)
        yset = self.find(parent, j)
        parent[xset] = yset

#Given two integers representing the numerator and denominator of a fraction, return the fraction in string format.
#
#If the fractional part is repeating, enclose the repeating part in parentheses.

class Solution(object):
    def fractionToDecimal(self, numerator, denominator):
        """
            :type numerator: int
            :type denominator: int
            :rtype: str
            """
        d = dict()
        if denominator == 0:
            return str(numerator)
        l = list()
        if (numerator < 0 and denominator > 0) or (numerator > 0 and denominator < 0):
            l.append("-")
        numerator = abs(numerator)
        denominator = abs(denominator)
        l.append(str(numerator / denominator))
        if numerator % denominator == 0:
            return ''.join(a for a in l)
    
    
        l.append('.')
        numerator = numerator % denominator
        
        while numerator % denominator:
            numerator = numerator * 10
            r = numerator / denominator
            remain = numerator % denominator
            test = r * 10 + remain
            if test in d:
                l.insert(d[test], '(')
                l.append(')')
                break
            l.append(str(r))
            d[test] = len(l) - 1
            numerator = remain
        return ''.join(a for a in l)

# Nim Game
class Solution(object):
    def canWinNim(self, n):
        """
            :type n: int
            :rtype: bool
            """
        return n % 4 != 0

#Given a list of words and two words word1 and word2, return the shortest distance between these two words in the list.
#
#For example,
#Assume that words = ["practice", "makes", "perfect", "coding", "makes"].
#
#Given word1 = “coding”, word2 = “practice”, return 3.
#Given word1 = "makes", word2 = "coding", return 1.

class Solution(object):
    def shortestDistance(self, words, word1, word2):
        """
            :type words: List[str]
            :type word1: str
            :type word2: str
            :rtype: int
            """
        length = len(words)
        index1 = length
        index2 = length
        answer = sys.maxint
        for i in xrange(length):
            if words[i] == word1:
                index1 = i
                answer = min(answer, abs(index1 - index2))
            if words[i] == word2:
                index2 = i
                answer = min(answer, abs(index1 - index2))
        return answer

#Say you have an array for which the ith element is the price of a given stock on day i.
#
#If you were only permitted to complete at most one transaction (ie, buy one and sell one share of the stock), design an algorithm to find the maximum profit.

class Solution(object):
    def maxProfit(self, prices):
        """
            :type prices: List[int]
            :rtype: int
            """
        length = len(prices)
        maxValue = 0
        minValue = sys.maxint
        for i in xrange(length):
            minValue = min(minValue, prices[i])
            maxValue = max(maxValue, prices[i] - minValue)
        
        return maxValue

#Say you have an array for which the ith element is the price of a given stock on day i.
#
#Design an algorithm to find the maximum profit. You may complete as many transactions as you like (ie, buy one and sell one share of the stock multiple times). However, you may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).

class Solution(object):
    def maxProfit(self, prices):
        """
            :type prices: List[int]
            :rtype: int
            """
        value = 0
        length = len(prices)
        for i in xrange(length - 1):
            value += max(0, prices[i + 1] - prices[i])
        return value

# Given a digit string, return all possible letter combinations that the number could represent.


class Solution(object):
    def letterCombinations(self, digits):
        """
            :type digits: str
            :rtype: List[str]
            """
        if digits == "":
            return list()
        d = dict()
        d['2'] = "abc"
        d['3'] = "def"
        d['4'] = "ghi"
        d['5'] = "jkl"
        d['6'] = "mno"
        d['7'] = "pqrs"
        d['8'] = "tuv"
        d['9'] = "wxyz"
        result = []
        digits = digits.replace('1', '')
        digits = digits.replace('0', '')
        self.helper_func(d, 0, digits, [], result)
        return result
    
    def helper_func(self, d, start, digits, combination, result):
        if start == len(digits):
            result.append(''.join(char for char in combination))
            return
        for i in d[digits[start]]:
            combination.append(i)
            self.helper_func(d, start + 1, digits, combination, result)
            combination.pop()


#Design and implement a TwoSum class. It should support the following operations: add and find.
#
#add - Add the number to an internal data structure.
#find - Find if there exists any pair of numbers which sum is equal to the value.
class TwoSum(object):
    
    def __init__(self):
        """
            initialize your data structure here
            """
        self.d = {}
    
    
    def add(self, number):
        """
            Add the number to an internal data structure.
            :rtype: nothing
            """
        self.d[number] = self.d.get(number, 0) + 1
    
    
    def find(self, value):
        """
            Find if there exists any pair of numbers which sum is equal to the value.
            :type value: int
            :rtype: bool
            """
        for key in self.d:
            if value - key in self.d and (value - key != key or self.d[key] > 1):
                return True
        return False


# Your TwoSum object will be instantiated and called as such:
# twoSum = TwoSum()
# twoSum.add(number)
# twoSum.find(value)

#Given a sorted array, remove the duplicates in place such that each element appear only once and return the new length.
#
#Do not allocate extra space for another array, you must do this in place with constant memory.

class Solution(object):
    def removeDuplicates(self, nums):
        """
            :type nums: List[int]
            :rtype: int
            """
        length = len(nums)
        pos = 0
        for i in xrange(length):
            if i == 0 or nums[i] != nums[pos - 1]:
                nums[pos] = nums[i]
                pos += 1
        return pos

#A message containing letters from A-Z is being encoded to numbers using the following mapping:
#
#'A' -> 1
#'B' -> 2
#...
#'Z' -> 26
#Given an encoded message containing digits, determine the total number of ways to decode it.
#
#For example,
#Given encoded message "12", it could be decoded as "AB" (1 2) or "L" (12).
#
#The number of ways decoding "12" is 2.

class Solution(object):
    def numDecodings(self, s):
        """
            :type s: str
            :rtype: int
            """
        length = len(s)
        if length == 0:
            return 0
        l = [0] * (length + 1)
        l[0] = 1
        l[1] = 1
        if s[0] == '0':
            l[1] = 0
        for i in xrange(2, length + 1):
            if s[i - 1] != '0':
                l[i] = l[i - 1]
            ten = int(s[i - 2])
            unit = int(s[i - 1])
            ten = 10 * ten + unit
            if ten >= 10 and ten <= 26:
                l[i] += l[i - 2]

    return l[length]

# spiral matrix
class Solution(object):
    def generateMatrix(self, n):
        """
            :type n: int
            :rtype: List[List[int]]
            """
        l = [[0 for i in xrange(n)] for i in xrange(n)]
        count = 0
        i = 1
        while i <= n * n:
            for j in xrange(count, n - count):
                l[count][j] = i
                i += 1
            for j in xrange(count + 1, n - count):
                l[j][n - count - 1] = i
                i += 1
            for j in reversed(xrange(count, n - count -1)):
                l[n - count - 1][j] = i
                i += 1
            for j in reversed(xrange(count + 1, n - count - 1)):
                l[j][count] = i
                i += 1
            count += 1
        return l


# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
            :type l1: ListNode
            :type l2: ListNode
            :rtype: ListNode
            """
        if not l1:
            return l2
        if not l2:
            return l1
        l_head = ListNode(0)
        pointer = l_head
        while l1 and l2:
            if l1.val < l2.val:
                pointer.next = ListNode(l1.val)
                pointer = pointer.next
                l1 = l1.next
            else:
                pointer.next = ListNode(l2.val)
                pointer = pointer.next
                l2 = l2.next
        if l1:
            while l1:
                pointer.next = ListNode(l1.val)
                l1 = l1.next
                pointer = pointer.next
        if l2:
            while l2:
                pointer.next = ListNode(l2.val)
                l2 = l2.next
                pointer = pointer.next
        return l_head.next

#Given an array of integers, find two numbers such that they add up to a specific target number.
#
#The function twoSum should return indices of the two numbers such that they add up to the target, where index1 must be less than index2. Please note that your returned answers (both index1 and index2) are not zero-based.

class Solution(object):
    def twoSum(self, nums, target):
        """
            :type nums: List[int]
            :type target: int
            :rtype: List[int]
            """
        d = dict()
        length = len(nums)
        result = list()
        for i in xrange(length):
            if nums[i] in d:
                result.append(d[nums[i]][1] + 1)
                result.append(i + 1)
                break
            d[target - nums[i]] = [nums[i], i]
        return result

#Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.
#
#Note:
#Elements in a triplet (a,b,c) must be in non-descending order. (ie, a ≤ b ≤ c)
#The solution set must not contain duplicate triplets.

class Solution(object):
    def threeSum(self, nums):
        """
            :type nums: List[int]
            :rtype: List[List[int]]
            """
        length = len(nums)
        nums.sort()
        result = list()
        for i in xrange(length):
            start = nums[i]
            if i > 0 and start == nums[i - 1]:
                continue
            left = i + 1
            right = length - 1
            while left < right:
                temp = start + nums[left] + nums[right]
                if temp == 0:
                    result.append([start, nums[left], nums[right]])
                    left += 1
                    right -= 1
                    while left < right:
                        if nums[left] == nums[left - 1]:
                            left += 1
                        else:
                            break
                    while right > left:
                        if nums[right] == nums[right + 1]:
                            right -= 1
                        else:
                            break
                elif temp < 0:
                    left += 1
                else:
                    right -= 1
        return result

# Given an array S of n integers, are there elements a, b, c, and d in S such that a + b + c + d = target? Find all unique quadruplets in the array which gives the sum of target.

class Solution(object):
    def fourSum(self, nums, target):
        """
            :type nums: List[int]
            :type target: int
            :rtype: List[List[int]]
            """
        length = len(nums)
        nums.sort()
        result = list()
        for i in xrange(length):
            if i > 0 and nums[i - 1] == nums[i]:
                continue
            start = nums[i]
            for j in xrange(i + 1, length):
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue
                middle1 = nums[j]
                middle2 = j + 1
                end = length - 1
                while middle2 < end:
                    temp = start + middle1 + nums[middle2] + nums[end]
                    if temp == target:
                        result.append([start, middle1, nums[middle2], nums[end]])
                        middle2 += 1
                        end -= 1
                        while nums[middle2 - 1] == nums[middle2] and middle2 < end:
                            middle2 += 1
                        while nums[end + 1] == nums[end] and middle2 < end:
                            end -= 1
                    elif temp < target:
                        middle2 += 1
                    else:
                        end -= 1
        return result


#Given a set of candidate numbers (C) and a target number (T), find all unique combinations in C where the candidate numbers sums to T.
#
#The same repeated number may be chosen from C unlimited number of times.
class Solution(object):
    def combinationSum(self, candidates, target):
        """
            :type candidates: List[int]
            :type target: int
            :rtype: List[List[int]]
            """
        candidates.sort()
        answer = list()
        single_answer = list()
        self.find_answer(answer, single_answer, candidates, target)
        
        return answer
    
    def find_answer(self, answer, single_answer, candidates, target):
        if target == 0:
            answer.append(single_answer[:])
            return
        
        length = len(candidates)
        
        for i in xrange(length):
            if candidates[i] > target:
                return
            self.find_answer(answer, single_answer + [candidates[i]], candidates[i:], target - candidates[i])
        
    return

#Suppose a sorted array is rotated at some pivot unknown to you beforehand.
#
#(i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
#
#Find the minimum element.
#
#You may assume no duplicate exists in the array.

class Solution(object):
    def findMin(self, nums):
        """
            :type nums: List[int]
            :rtype: int
            """
        length = len(nums)
        start = 0
        end = length - 1
        result = nums[0]
        return self.helper_func(nums, start, end, result)
    
    def helper_func(self, nums, start, end, result):
        if start > end:
            return result
        if nums[start] < result:
            result = nums[start]
        if nums[end] < result:
            result = nums[end]
        middle = (start + end) / 2
        if nums[middle] < result:
            result = nums[middle]
        
        if nums[middle] > nums[start]:
            if result > nums[start] and result < nums[middle]:
                return self.helper_func(nums, start + 1, end, result)
            else:
                return self.helper_func(nums, middle + 1, end, result)
        if nums[end] > nums[middle]:
            if result < nums[end] and result > nums[middle]:
                return self.helper_func(nums, middle + 1, end, result)
            else:
                return self.helper_func(nums, start + 1, middle, result)
        else:
            p = self.helper_func(nums, start + 1, middle, result)
            q = self.helper_func(nums, middle + 1, end, result)
            return min(p, q)


# The isBadVersion API is already defined for you.
# @param version, an integer
# @return a bool
# def isBadVersion(version):

class Solution(object):
    def firstBadVersion(self, n):
        """
            :type n: int
            :rtype: int
            """
        start = 1
        end = n
        while start + 1 < end:
            middle = (start + end) / 2
            if isBadVersion(middle):
                end = middle
            else:
                start = middle
        if isBadVersion(start):
            return start
        return end


# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
            :type head1, head1: ListNode
            :rtype: ListNode
            """
        lengthA = 0
        lengthB = 0
        pointerA = headA
        pointerB = headB
        while pointerA:
            pointerA = pointerA.next
            lengthA += 1
        while pointerB:
            pointerB = pointerB.next
            lengthB += 1
        if lengthA == 0 or lengthB == 0:
            return None
        while lengthA < lengthB:
            headB = headB.next
            lengthB -= 1
        while lengthB < lengthA:
            headA = headA.next
            lengthA -= 1
        while headA and headB:
            if headA == headB:
                return headA
            headA = headA.next
            headB = headB.next
        return None

#You are playing the following Flip Game with your friend: Given a string that contains only these two characters: + and -, you and your friend take turns to flip two consecutive "++" into "--". The game ends when a person can no longer make a move and therefore the other person will be the winner.
#
#Write a function to compute all possible states of the string after one valid move.



class Solution(object):
    def generatePossibleNextMoves(self, s):
        """
            :type s: str
            :rtype: List[str]
            """
        if not s:
            return list()
        l = list()
        index = 0
        self.helper_func(s, index, l)
        return l
    
    def helper_func(self, s, index, l):
        if index == len(s) - 1:
            return
        if index < len(s) - 1 and s[index] == s[index + 1]:
            if s[index] == '+':
                l.append(s[:index] + "--" + s[index + 2:])
        self.helper_func(s, index + 1, l)

# swap nodes in pair
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def swapPairs(self, head):
        """
            :type head: ListNode
            :rtype: ListNode
            """
        if not head:
            return None
        pointer = head
        nextPointer = pointer.next
        fakeHead = ListNode(-1)
        fakeHead.next = head
        head = fakeHead
        while head and head.next and head.next.next:
            temp1 = head.next
            temp2 = temp1.next
            toNext = temp2.next
            head.next = temp2
            temp2.next = temp1
            temp1.next = toNext
            head = temp1
        return fakeHead.next


# Definition for binary tree with next pointer.
# class TreeLinkNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None

class Solution(object):
    def connect(self, root):
        """
            :type root: TreeLinkNode
            :rtype: nothing
            """
        if not root:
            return
        l = list()
        l.append(root)
        while l:
            length = len(l)
            for i in xrange(length - 1):
                l[i].next = l[i + 1]
            l[length - 1].next = None
            l1 = list()
            while l:
                node = l.pop(0)
                if node.left:
                    l1.append(node.left)
                if node.right:
                    l1.append(node.right)
            l = l1[:]
#
#Given a positive integer, return its corresponding column title as appear in an Excel sheet.
#
#For example:
#    
#    1 -> A
#    2 -> B
#    3 -> C
#    ...
#    26 -> Z
#    27 -> AA
#    28 -> AB

class Solution(object):
    def convertToTitle(self, n):
        """
            :type n: int
            :rtype: str
            """
        result = list()
        while n:
            divide = n % 26
            if divide == 0:
                divide = 26
            result.insert(0, chr(divide + 64))
            if divide == 26:
                n -= 1
            n /= 26
        return ''.join(c for c in result)


#Find the kth largest element in an unsorted array. Note that it is the kth largest element in the sorted order, not the kth distinct element.
#
#For example,
#Given [3,2,1,5,6,4] and k = 2, return 5.
class Solution(object):
    def findKthLargest(self, nums, k):
        """
            :type nums: List[int]
            :type k: int
            :rtype: int
            """
        nums.sort(reverse = True)
        return nums[k - 1]


# Given an unsorted array of integers, find the length of the longest consecutive elements sequence. O(n) time
class Solution(object):
    def longestConsecutive(self, nums):
        """
            :type nums: List[int]
            :rtype: int
            """
        s1 = set()
        result = 1
        for num in nums:
            s1.add(num)
        for num in nums:
            val = num
            while val in s1:
                s1.remove(val)
                val -= 1
            minimum = val + 1
            val = num
            while val + 1 in s1:
                s1.remove(val + 1)
                val += 1
            maximum = val - minimum + 1
            result = max(maximum, result)
        return result


# returns the permutation of a set
class Solution(object):
    def permute(self, nums):
        """
            :type nums: List[int]
            :rtype: List[List[int]]
            """
        l = list()
        result = list()
        self.helper_func(l, nums, result)
        return result
    
    def helper_func(self, l, nums, result):
        if len(l) == len(nums):
            result.append(l[:])
            return
        for i in nums:
            if i not in l:
                l.append(i)
                self.helper_func(l, nums, result)
                l.pop()
#Given a collection of numbers that might contain duplicates, return all possible unique permutations.
#
#For example,
#[1,1,2] have the following unique permutations:
#[1,1,2], [1,2,1], and [2,1,1].

class Solution(object):
    def permuteUnique(self, nums):
        """
            :type nums: List[int]
            :rtype: List[List[int]]
            """
        length = len(nums)
        visit = [False for i in xrange(length)]
        l = list()
        result = list()
        nums.sort()
        self.helper_func(visit, l, result, nums)
        return result
    
    def helper_func(self, visit, l, result, nums):
        if len(l) == len(nums):
            result.append(l[:])
            return
        length = len(nums)
        for i in xrange(length):
            if i > 0 and nums[i] == nums[i - 1] and not visit[i - 1]:
                continue
            if not visit[i]:
                visit[i] = True
                l.append(nums[i])
                self.helper_func (visit, l, result, nums)
                l.pop()
                visit[i] = False

# find the lowest common ancestor of a binary tree
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
        if not root:
            return None
        if not p:
            return root
        if not q:
            return root
        if p == root or q == root:
            return root
        onLeft = self.lowestCommonAncestor(root.left, p ,q)
        onRight = self.lowestCommonAncestor(root.right, p, q)
        if onLeft and onRight:
            return root
        else:
            if onLeft and not onRight:
                return onLeft
            else:
                return onRight

# rotate an image 90 degree counterclockwise
class Solution(object):
    def rotate(self, matrix):
        """
            :type matrix: List[List[int]]
            :rtype: void Do not return anything, modify matrix in-place instead.
            """
        if not matrix:
            return
        length = len(matrix)
        for i in xrange(length / 2):
            for j in xrange((length + 1) / 2):
                temp = matrix[i][j]
                matrix[i][j] = matrix[length - j - 1][i]
                matrix[length - j - 1][i] = matrix[length - i - 1][length - j - 1]
                matrix[length - i - 1][length - j - 1] = matrix[j][length - i - 1]
                matrix[j][length - i - 1] = temp

#Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:
#
#Integers in each row are sorted from left to right.
#The first integer of each row is greater than the last integer of the previous row.

class Solution(object):
    def searchMatrix(self, matrix, target):
        """
            :type matrix: List[List[int]]
            :type target: int
            :rtype: bool
            """
        if not matrix:
            return False
        lengthHorizontal = len(matrix)
        lengthVertical = len(matrix[0])
        start = 0
        end = lengthHorizontal - 1
        row = 0
        while start <= end:
            middle = (start + end) / 2
            if matrix[middle][0] <= target and matrix[middle][lengthVertical - 1] >= target:
                row = middle
                break
            elif matrix[middle][0] < target:
                start = middle + 1
            else:
                end = middle - 1
        start = 0
        end = lengthVertical - 1
        while start <= end:
            middle = (start + end) / 2
            if matrix[row][end] == target:
                return True
            if matrix[row][start] == target:
                return True
            if matrix[row][middle] == target:
                return True
            elif matrix[row][middle] < target:
                start = middle + 1
            else:
                end = middle - 1
        return False


# binary tree longest consecutive number
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def __init__(self):
        self.longest = 1
    def longestConsecutive(self, root):
        """
            :type root: TreeNode
            :rtype: int
            """
        if not root:
            return 0
        self.helper_func(root, root.val, 0)
        return self.longest
    
    def helper_func(self, root, expected, temp):
        if not root:
            if temp > self.longest:
                self.longest = temp
            return
        if root.val == expected:
            temp += 1
            if temp > self.longest:
                self.longest = temp
        else:
            temp = 1
        
        self.helper_func(root.left, root.val + 1, temp)
        self.helper_func(root.right, root.val + 1, temp)


#You are playing the following Bulls and Cows game with your friend: You write a 4-digit secret number and ask your friend to guess it, each time your friend guesses a number, you give a hint, the hint tells your friend how many digits are in the correct positions (called "bulls") and how many digits are in the wrong positions (called "cows"), your friend will use those hints to find out the secret number.
#
#For example:
#
#Secret number:  1807
#Friend's guess: 7810
#Hint: 1 bull and 3 cows. (The bull is 8, the cows are 0, 1 and 7.)
class Solution(object):
    def getHint(self, secret, guess):
        """
            :type secret: str
            :type guess: str
            :rtype: str
            """
        bull = 0
        cow = 0
        d1 = dict()
        length = len(secret)
        for i in xrange(length):
            if secret[i] == guess[i]:
                bull += 1
            else:
                d1[secret[i]] = d1.get(secret[i], 0) + 1
                d1[guess[i]] = d1.get(guess[i], 0) - 1
                if d1[secret[i]] <= 0:
                    cow += 1
                if d1[guess[i]] >= 0:
                    cow += 1
        return str(bull) + "A" + str(cow) + "B"
#
#Given an input string, reverse the string word by word.
#
#For example,
#Given s = "the sky is blue",
#return "blue is sky the".
class Solution(object):
    def reverseWords(self, s):
        """
            :type s: str
            :rtype: str
            """
        l = s.split(" ")
        if l == []:
            return ""
        length = len(l)
        for i in xrange(length):
            word = l.pop(0)
            if word == "":
                continue
            l.append(word[::-1])
        
        
        value = ' '.join(c for c in l)
        return value[::-1]

# search in a rotated list, duplicates allowed
class Solution(object):
    def search(self, nums, target):
        """
            :type nums: List[int]
            :type target: int
            :rtype: bool
            """
        start = 0
        end = len(nums) - 1
        while start <= end:
            middle = (start + end) / 2
            print nums[middle], nums[start], nums[end]
            if nums[middle] == target or nums[start] == target or nums[end] == target:
                return True
            if nums[middle] < nums[end] or nums[middle] < nums[start]:
                if target > nums[middle] and target < nums[end]:
                    start = middle + 1
                else:
                    end = middle - 1
            elif nums[middle] > nums[start] or nums[middle] > nums[end]:
                if  target < nums[middle] and target > nums[start]:
                    end = middle - 1
                else:
                    start = middle + 1
            else:
                end -= 1

    return False

#Given an unsorted array of integers, find the length of longest increasing subsequence.
class Solution(object):
    def lengthOfLIS(self, nums):
        """
            :type nums: List[int]
            :rtype: int
            """
        if not nums:
            return 0
        length = len(nums)
        dp = [1] * length
        for i in xrange(1, length):
            for j in xrange(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)


#Given a list of non negative integers, arrange them such that they form the largest number.
#
#For example, given [3, 30, 34, 5, 9], the largest formed number is 9534330.
class Solution(object):
    def largestNumber(self, nums):
        """
            :type nums: List[int]
            :rtype: str
            """
        if not nums:
            return "0"
        num = map(str, nums)
        num.sort(cmp = self.compare)
        return str(int(''.join(num)))
    
    def compare(self, a, b):
        if int(a + b) > int (b + a):
            return -1
        else:
            return 1
