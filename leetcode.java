/* older version */
/* instruction: Given an array of integers, every element appears twice except for one. Find that single one. */
import java.util.HashMap;
public class Solution {
    public int singleNumber(int[] A) {
        HashMap<Integer, Integer> hm = new HashMap<Integer, Integer>();
        for(int n = 0; n < A.length; n ++){
            int value = 0;
            int key = 0;
            key = A[n];
            if(hm.containsKey(key)){
                value = hm.get(key);
                value ++;
            }
            hm.put(key, value);
        }
        int ret = 0;
        for(int n = 0; n < A.length; n ++){
            if(hm.get(A[n]) == 0)
                ret = A[n];
        }
        return ret;
    }
}

/**
 * Definition for binary tree
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */

/* instruction: Given a binary tree, find its maximum depth.
 The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.
 */

/************ Tree **********************/
public class Solution {
    public int maxDepth(TreeNode root) {
        if(root == null)
            return 0;
        int max_right = maxDepth(root.right);
        int max_left = maxDepth(root.left);
        if(max_right > max_left)
            return max_right + 1;
        else
            return max_left + 1;
    }
}

/* instruction: invert binary tree */
public class Solution {
    public TreeNode invertTree(TreeNode root) {
        
        if(root == null)
            return null;
        
        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;
        
        invertTree(root.left);
        invertTree(root.right);
        
        
        return root;
        
        
    }
}

/* instruction: Given a binary tree, determine if it is a valid binary search tree (BST). */
public class Solution {
    public boolean isValidBST(TreeNode root) {
        if(root == null)
            return true;
        if(root.left == null && root.right == null)
            return true;
        int left, right;
        left = Integer.MIN_VALUE;
        right = Integer.MAX_VALUE;
        return binaryRecursion(root, left, right);
    }
    public boolean binaryRecursion(TreeNode root, int left, int right)
    {
        if(root == null)
            return true;
        if(root.val < left || root.val > right)
            return false;
        if(root.val == Integer.MIN_VALUE && root.left != null)
            return false;
        if(root.val == Integer.MAX_VALUE && root.right != null)
            return false;
        boolean p, q;
        p = binaryRecursion(root.left, left, root.val - 1);
        q = binaryRecursion(root.right, root.val + 1, right);
        return p && q;
    }
}

/* instruction: determine a binary tree has a path that sum up to a number */
public boolean hasPathSum(TreeNode root, int sum) {
    if(root == null)
        return false;
    boolean success = false;

    if(root.left == null && root.right == null)
        return (root.val == sum);
    TreeNode head = root;
    if (head.left != null){
        success = hasPathSum(head.left, sum - head.val);
        if(success)
            return success;
    }
    if(head.right != null){
        success = hasPathSum(head.right, sum - head.val);
        if(success)
            return success;
    }
    return success;
}

/* instruction: Given a binary tree and a sum, find all root-to-leaf paths where each path's sum equals the given sum. */
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
public class Solution {
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        List<List<Integer>> list = new ArrayList<List<Integer>>();
        if(root == null)
            return list;
        List<Integer> temp = new LinkedList<Integer>();
        getPath(root, sum, list, temp);
        return list;
    }
    
    public void getPath(TreeNode root, int sum, List<List<Integer>> list, List<Integer> temp)
    {
        temp.add(root.val);
        if(root.left == null && root.right == null && sum == root.val)
        {
            LinkedList<Integer> add = new LinkedList<Integer>();
            add.addAll(temp);
            list.add(add);
            temp.remove(temp.size() - 1);
            return;
        }
        if(root.left != null)
            getPath(root.left, sum - root.val, list, temp);
        if(root.right != null)
            getPath(root.right, sum - root.val, list,temp);
        temp.remove(temp.size() - 1);
    }
}

/* instruction: Given a binary tree, flatten it to a linked list in-place. */
public class Solution {
    List<Integer> list = new LinkedList<Integer>();
    public void flatten(TreeNode root) {
        if(root == null)
            return;
        getPreOrderTraversal(root);
        int size = list.size();
        TreeNode pointer = root;
        pointer.left = null;
        for(int i = 1; i < size; i ++)
        {
            pointer.right = new TreeNode(list.get(i));
            pointer = pointer.right;
        }
    }
    public void getPreOrderTraversal(TreeNode root)
    {
        list.add(root.val);
        if(root.left != null)
            getPreOrderTraversal(root.left);
        if(root.right != null)
            getPreOrderTraversal(root.right);
    }
}

/* instructure: Implement an iterator over a binary search tree (BST). Your iterator will be initialized with the root node of a BST.
 
 Calling next() will return the next smallest number in the BST.
 
 Note: next() and hasNext() should run in average O(1) time and uses O(h) memory, where h is the height of the tree.
 */
import java.util.Stack;
public class BSTIterator {
    Stack<TreeNode> stack;
    public BSTIterator(TreeNode root) {
        stack = new Stack<TreeNode>();
        inStack(root);
    }
    public void inStack(TreeNode root)
    {
        if(root == null)
            return;
        stack.push(root);
        inStack(root.left);
    }
    /** @return whether we have a next smallest number */
    public boolean hasNext() {
        return !stack.isEmpty();
    }
    /** @return the next smallest number */
    public int next() {
        TreeNode root = stack.pop();
        inStack(root.right);
        return root.val;
    }
}

/* instruction: Given a binary tree, return the zigzag level order traversal of its nodes' values. (ie, from left to right, then right to left for the next level and alternate between).*/
import java.util.ArrayList;
import java.util.List;
public class Solution {
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<TreeNode> queue = new ArrayList<TreeNode>();
        List<List<TreeNode>> traversal = new ArrayList<List<TreeNode>>();
        if(root == null)
            return new ArrayList<List<Integer>>();
        queue.add(root);
        while(queue.size() != 0)
        {
            traversal.add(queue);
            queue = new ArrayList<TreeNode>();
            int size = traversal.size();
            List<TreeNode> temp = traversal.get(size - 1);
            int tempsize = temp.size();
            for(int i = 0; i < tempsize; i ++)
            {
                TreeNode node = temp.get(i);
                if(node.left != null)
                    queue.add(node.left);
                if(node.right != null)
                    queue.add(node.right);
            }
        }
        List<List<Integer>> values = new ArrayList<List<Integer>>();
        int size = traversal.size();
        for(int i = 0; i < size; i ++)
        {
            List<TreeNode> nodes = traversal.get(i);
            List<Integer> value = new ArrayList<Integer>();
            int length = nodes.size();
            if(i % 2 == 0)
            {
                for(int j = 0; j < length; j ++)
                {
                    TreeNode node = nodes.get(j);
                    value.add(node.val);
                }
            }
            else
            {
                for(int j = length - 1; j >= 0; j --)
                {
                    TreeNode node = nodes.get(j);
                    value.add(node.val);
                }
            }
            values.add(value);
        }
        return values;
    }
}

/* instruction: Given two binary trees, write a function to check if they are equal or not. */
public class Solution {
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if((p == null) && (q == null))
            return true;
        if(((p == null) && (q != null)) || ((p != null) && (q == null)))
            return false;
        if(p.val != q.val)
            return false;
        if(!isSameTree(p.left, q.left))
            return false;
        if(!isSameTree(p.right, q.right))
            return false;
        return true;
    }
}

/* instruction: Given a singly linked list where elements are sorted in ascending order, convert it to a height balanced BST. */
//singly linked list is defined as LisNode a = ...
public class Solution {
    public TreeNode sortedListToBST(ListNode head)
    {
        if(head == null)
            return null;
        if(head.next == null)
        {
            TreeNode root = new TreeNode(head.val);
            return root;
        }
        ListNode fast = head, slow = head, previous = head;
        while(fast != null && fast.next != null)
        {
            fast = fast.next.next;
            previous = slow;
            slow = slow.next;
        }
        TreeNode root = new TreeNode(slow.val);
        ListNode middle = slow.next;
        previous.next = null;
        root.left = sortedListToBST(head);
        root.right = sortedListToBST(middle);
        return root;
    }
}

/* instruction: reverse a linked list */
public class Solution {
    public ListNode reverseList(ListNode head) {
        if(head == null)
            return null;
        ListNode pointer = head;
        ListNode previous = null;
        ListNode next;
        while(pointer!= null)
        {
            next = pointer.next;
            pointer.next = previous;
            previous = pointer;
            pointer = next;
        }
        head = previous;
        return head;
    }
}


/* instruction: delete a single node in a singly linked list, given only that node */

public class Solution {
    public void deleteNode(ListNode node) {
        ListNode pointer = node;
        ListNode pre = node;
        while(node.next != null)
        {
            node.val = node.next.val;
            pre = node;
            node = node.next;
        }
        pre.next = null;
    }
}

/* instruction: Given a binary tree, return the preorder traversal of its nodes' values. */
import java.util.Stack;
public class Solution {
    public ArrayList<Integer> preorderTraversal(TreeNode root) {
        ArrayList<Integer> ret = new ArrayList<Integer>();
        if(root == null)
            return ret;
        Stack<TreeNode> temp = new Stack<TreeNode>();
        temp.push(root);
        TreeNode curr;
        while(!temp.empty()){
            curr = temp.peek();
            ret.add(curr.val);
            temp.pop();
            if(curr.right != null)
                temp.push(curr.right);
            if(curr.left != null)
                temp.push(curr.left);
        }
        return ret;
    }
}

/* instruction: preorder traversal using recursion */
public class Solution {
    /**
     * @param root: The root of binary tree.
     * @return: Preorder in ArrayList which contains node values.
     */
    ArrayList<Integer> al = new ArrayList<Integer>();
    public ArrayList<Integer> preorderTraversal(TreeNode root) {
        // write your code here
        if(root == null)
            return al;
        al.add(root.val);
        if(root.left != null)
            preorderTraversal(root.left);
        if(root.right != null)
            preorderTraversal(root.right);
        return al;
    }
}

/* Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.
 Example:
     1 -> NULL
    /  \
   2 -> 3 -> NULL
  / \  / \
 4->5->6->7 -> NULL
 */

public class Solution {
    public void connect(TreeLinkNode root) {
        if(root == null)
            return;
        TreeLinkNode curr = root;
        if(curr.left != null && curr.right != null)
            curr.left.next = curr.right;
        if(curr.next != null && curr.right != null)
            curr.right.next = curr.next.left;
        connect(curr.left);
        connect(curr.right);
    }
}

/* instruction: Given a binary tree, return the postorder traversal of its nodes' values. */
public class Solution {
    ArrayList<Integer> al = new ArrayList<Integer>();
    public ArrayList<Integer> postorderTraversal(TreeNode root) {
        if(root == null)
            return al;
        if(root.left != null)
            postorderTraversal(root.left);
        if(root.right != null)
            postorderTraversal(root.right);
        al.add(root.val);
        return al;
    }
}

/* binary tree inorder traversal */
public class Solution {
    /**
     * @param root: The root of binary tree.
     * @return: Inorder in ArrayList which contains node values.
     */
    ArrayList<Integer> al = new ArrayList<Integer>();
    public ArrayList<Integer> inorderTraversal(TreeNode root) {
        if(root == null)
            return al;
        if(root.left != null)
            inorderTraversal(root.left);
        al.add(root.val);
        if(root.right != null)
            inorderTraversal(root.right);
        return al;
    }
}

/* instruction: Given inorder and postorder traversal of a tree, construct the binary tree. */
import java.util.Arrays;
public class Solution {
    public TreeNode buildTree(int[] inorder, int[] postorder) {
        int len = inorder.length;
        if(len == 0)
            return null;
        if(len == 1)
            return new TreeNode(inorder[0]);
        TreeNode root = new TreeNode(postorder[len - 1]);
        int index = find_index(postorder[len - 1], inorder);
        int[] leftInorder = Arrays.copyOfRange(inorder, 0, index);
        int[] rightInorder = Arrays.copyOfRange(inorder, index + 1, len);
        int[] leftPostorder = Arrays.copyOfRange(postorder, 0, leftInorder.length);
        int[] rightPostorder = Arrays.copyOfRange(postorder, leftInorder.length, leftInorder.length + rightInorder.length);
        TreeNode left = buildTree(leftInorder, leftPostorder);
        TreeNode right = buildTree(rightInorder, rightPostorder);
        root.left = left;
        root.right = right;
        return root;
    }
    public int find_index(int target, int[] array){
        int i;
        int len = array.length;
        for(i = 0; i < len; i ++)
            if(array[i] == target)
                return i;
        return -1;
    }
}

/* instruction: Construct Binary Tree from Preorder and Inorder Traversal  */
import java.util.Arrays;
public class Solution {
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        int len = inorder.length;
        if(len == 0)
            return null;
        if(len == 1)
            return new TreeNode(inorder[0]);
        TreeNode root = new TreeNode(preorder[0]);
        int index = find_index(preorder[0], inorder);
        int[] leftInorder = Arrays.copyOfRange(inorder, 0, index);
        int[] rightInorder = Arrays.copyOfRange(inorder, index + 1, len);
        int[] leftPreorder = Arrays.copyOfRange(preorder, 1, leftInorder.length + 1);
        int[] rightPreorder = Arrays.copyOfRange(preorder, leftInorder.length + 1, leftInorder.length + rightInorder.length + 1);
        TreeNode left = buildTree(leftPreorder, leftInorder);
        TreeNode right = buildTree(rightPreorder, rightInorder);
        root.left = left;
        root.right = right;
        return root;
    }
    public int find_index(int target, int[] array){
        int i;
        int len = array.length;
        for(i = 0; i < len; i ++)
            if(array[i] == target)
                return i;
        return -1;
    }
}

/* instruction: Given a binary tree containing digits from 0-9 only, each root-to-leaf path could represent a number.
 
 An example is the root-to-leaf path 1->2->3 which represents the number 123.
 
 Find the total sum of all root-to-leaf numbers.
 */
public class Solution {
    public int sumNumbers(TreeNode root) {
        if(root == null)
            return 0;
        if(root.left == null && root.right == null)
            return root.val;
        int sum = 0;
        if(root.left != null){
            root.left.val += (root.val * 10);
            sum += sumNumbers(root.left);
        }
        if(root.right != null){
            root.right.val += (root.val * 10);
            sum += sumNumbers(root.right);
        }
        return sum;
        
    }
}

/*******************Tree Ends*************************/

/*******************List Node*************************/
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */

/**
 * Definition for singly-linked list.
 * class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */

/* instruction: remove elements in Linked List */
public class Solution {
    public ListNode removeElements(ListNode head, int val) {
        ListNode pointer = head;
        ListNode pre = pointer;
        while(pointer != null)
        {
            boolean moved = false;
            ListNode next = pointer.next;
            if(pointer.val == val)
            {
                if(pre == pointer)
                {
                    head = head.next;
                }
                else
                {
                    pre.next = pointer.next;
                    moved = true;
                }
            }
            if(!moved)
                pre = pointer;
            pointer = next;
            if(pointer == head)
                pre = pointer;
        }
        return head;
    }
}

/* instruction: Sort a linked list using insertion sort. */
public class Solution {
    public ListNode insertionSortList(ListNode head) {
        if(head == null || head.next == null)
            return head;
        ListNode pointer = head.next;
        ListNode sortedList = head;
        sortedList.next = null;
        while(pointer != null)
        {
            ListNode current = pointer;
            pointer = pointer.next;
            boolean insertion = false;
            ListNode pre = null;
            for(ListNode i = sortedList; i != null; i = i.next)
            {
                if(i.val > current.val && pre == null)
                    break;
                if(i.val >= current.val && pre != null)
                {
                    pre.next = current;
                    current.next = i;
                    insertion = true;
                    break;
                }
                pre = i;
            }
            if(insertion == false)
            {
                if(current.val < sortedList.val)
                {
                    current.next = sortedList;
                    sortedList = current;
                }
                else
                {
                    pre.next = current;
                    current.next = null;
                }
            }
        }
        return sortedList;
    }
}

/* instruction: determine happy number
 A happy number is a number defined by the following process: Starting with any positive integer, replace the number by the sum of the squares of its digits, and repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1. Those numbers for which this process ends in 1 are happy numbers.
*/
public class Solution {
    public boolean isHappy(int n) {
        if(n <= 0)
            return false;
        HashSet<Integer> set = new HashSet<Integer>();
        set.add(n);
        while(n != 1)
        {
            ArrayList<Integer> list = getDigits(n);
            n = getSquareSum(list);
            if(set.contains(n))
                return false;
            set.add(n);
        }
        return true;
    }
    public int getSquareSum(ArrayList<Integer> list)
    {
        int n = 0;
        for(Integer a: list)
        {
            n += a * a;
        }
        return n;
    }
    public ArrayList<Integer> getDigits(int n)
    {
        ArrayList<Integer> list = new ArrayList<Integer>();
        while(n != 0)
        {
            list.add(n % 10);
            n /= 10;
        }
        return list;
    }
}

/* instruction: sort linked list in O(nlogn) */
public class Solution {
    public ListNode sortList(ListNode head) {
        if(head == null || head.next == null)
            return head;

        int length = 0;
        ListNode dummy = head;
        while(dummy != null)
        {
            dummy = dummy.next;
            length ++;
        }
        ListNode left = head, right = null;
        ListNode dummy2 = head;
        int half_count = 0;
        if(length == 2)
        {
            right = head.next;
            head.next = null;
        }
        else
        {
            while(dummy2 != null)
            {
                ListNode next = dummy2.next;
                half_count ++;
                if(half_count == length / 2)
                {
                    right = next;
                    dummy2.next = null;
                    break;
                }
                dummy2 = next;
            }
        }


        left = sortList(left);
        right = sortList(right);

        ListNode h = merge(left, right);
        return h;

    }
    public ListNode merge(ListNode a, ListNode b)
    {
        ListNode head = new ListNode(0);
        ListNode dummy = head;
        while(a != null && b != null)
        {
            if(a.val < b.val)
            {
                dummy.next = a;
                a = a.next;
            }
            else
            {
                dummy.next = b;
                b = b.next;
            }
            dummy = dummy.next;
        }
        if(a != null)
        {
            dummy.next = a;
        }
        if(b != null)
        {
            dummy.next = b;
        }
        return head.next;
    }
}

/* instruction: You are given two linked lists representing two non-negative numbers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.
 
 Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
 Output: 7 -> 0 -> 8
 */
public class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        if(l1 == null)
            return l2;
        if(l2 == null)
            return l1;
        ListNode head = new ListNode((l1.val + l2.val) % 10);
        ListNode pointer = head;
        ListNode node;
        int flag = (l1.val + l2.val)/10;
        
        while(l1.next != null && l2.next != null){
            l1 = l1.next;
            l2 = l2.next;
            node = new ListNode(l1.val);
            node.val = (l1.val + l2.val + flag) % 10;
            flag = (l1.val + l2.val + flag)/10;
            pointer.next = node;
            pointer = pointer.next;
        }
        if(l2 != null)
            while(l2.next != null){
                l2 = l2.next;
                node = new ListNode(0);
                node.val = (l2.val + flag)%10;
                flag = (l2.val + flag)/10;
                pointer.next = node;
                pointer = pointer.next;
            }
        if(l1 != null)
            while(l1.next != null){
                l1 = l1.next;
                node = new ListNode(0);
                node.val = (l1.val + flag)%10;
                flag = (l1.val + flag)/10;
                pointer.next = node;
                pointer = pointer.next;
            }
        if(flag == 1){
            node = new ListNode(1);
            pointer.next = node;
        }
        return head;
    }
}

/* instruction: Given a list, rotate the list to the right by k places, where k is non-negative. */
public class Solution {
    public ListNode rotateRight(ListNode head, int k)
    {
        if(head == null || k == 0)
            return head;
        int length = 0;
        ListNode pointer = head, previous = null;
        while(pointer != null)
        {
            length ++;
            previous = pointer;
            pointer = pointer.next;
        }
        if(length == 1 || k % length == 0)
            return head;
        ListNode tail = previous;
        pointer = head;
        for(int i = 0; i < length - k % length; i ++)
        {
            previous = pointer;
            pointer = pointer.next;
        }
        tail.next = head;
        previous.next = null;
        return pointer;
    }
}

/* instruction: Write a program to find the node at which the intersection of two singly linked lists begins. */
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        int lenA = 0, lenB = 0;
        ListNode head1 = headA;
        ListNode head2 = headB;
        while(head1 != null)
        {
            head1 = head1.next;
            lenA ++;
        }
        while(head2 != null)
        {
            head2 = head2.next;
            lenB ++;
        }
        while(lenA > lenB)
        {
            headA = headA.next;
            lenA --;
        }
        while(lenB > lenA)
        {
            headB = headB.next;
            lenB --;
        }
        ListNode value = null;
        while(headA != null && headB != null)
        {
            if(headA == headB)
            {
                value = headA;
                break;
            }
            headA = headA.next;
            headB = headB.next;
        }
        return value;
    }
}

/* instruction: Merge two sorted linked lists and return it as a new list. The new list should be made by splicing together the nodes of the first two lists. */
public class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if(l1 == null)
            return l2;
        if(l2 == null)
            return l1;
        ListNode head = new ListNode(0);
        head.val = l1.val < l2.val ? l1.val: l2.val;
        if(head.val == l1.val)
            l1 = l1.next;
        else
            l2 = l2.next;
        ListNode node, pointer = head;
        while(l1 != null && l2 != null){
            node = new ListNode(0);
            if(l1.val < l2.val){
                node.val = l1.val;
                l1 = l1.next;
            }
            else{
                node.val = l2.val;
                l2 = l2.next;
            }
            pointer.next = node;
            pointer = pointer.next;
        }
        while(l1 != null){
            node = new ListNode(l1.val);
            pointer.next = node;
            pointer = pointer.next;
            l1 = l1.next;
        }
        while(l2 != null){
            node = new ListNode(l2.val);
            pointer.next = node;
            pointer = pointer.next;
            l2 = l2.next;
        }
        return head;
    }
}

/* instruction: Given a sorted linked list, delete all nodes that have duplicate numbers, leaving only distinct numbers from the original list.*/
public class Solution {
    private ListNode pointer, tail, previous;
    private boolean moveHead;
    public ListNode deleteDuplicates(ListNode head) {
        HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
        pointer = head;
        while(pointer != null)
        {
            if(!map.containsKey(pointer.val))
                map.put(pointer.val, 1);
            else
                map.put(pointer.val, map.get(pointer.val) + 1);
            pointer = pointer.next;
        }
        tail = pointer;
        pointer = head;
        previous = head;
        while(pointer != null)
        {
            if(map.get(pointer.val) > 1)
            {
                pointer = removeNode();
                if(moveHead)
                    head = head.next;
            }
            else
            {
                previous = pointer;
                pointer = pointer.next;
            }
        }
        return head;
    }
    public ListNode removeNode()
    {
        moveHead = false;
        if(previous == pointer)
        {
            previous = previous.next;
            pointer = pointer.next;
            moveHead =true;
            return pointer;
        }
        if(pointer == tail)
            return previous;
        previous.next = pointer.next;
        return previous;
    }
}


/* instruction: Given a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x. */
public class Solution {
    public ListNode partition(ListNode head, int x) {
        if(head == null)
            return null;
        ListNode left = null, right = null, pointer, left_pointer = null, right_pointer = null;
        pointer = head;
        while(pointer != null)
        {
            if(pointer.val < x)
            {
                if(left == null)
                {
                    left = new ListNode(pointer.val);
                    left_pointer = left;
                }
                else
                {
                    left.next = new ListNode(pointer.val);
                    left = left.next;
                }
            }
            else
            {
                if(right == null)
                {
                    right = new ListNode(pointer.val);
                    right_pointer = right;
                }
                else
                {
                    right.next = new ListNode(pointer.val);
                    right = right.next;
                }
            }
            pointer = pointer.next;
        }
        if(left == null)
            return right_pointer;
        while(right_pointer != null)
        {
            left.next = new ListNode(right_pointer.val);
            right_pointer = right_pointer.next;
            left = left.next;
        }
        return left_pointer;
        
    }
}

/* instruction: Given a sorted linked list, delete all duplicates such that each element appear only once.*/
public class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        ListNode current = head;
        if(head == null){
            return head;
        }
        while(current.next != null){
            if(current.val == current.next.val){
                if(current.next.next != null)
                    current.next = current.next.next;
                else
                    current.next = null;
            }
            else
                current = current.next;
        }
        return head;
    }
}

/* instruction: Reverse bits of a given 32 bits unsigned integer.
 
 For example, given input 43261596 (represented in binary as 00000010100101000001111010011100), return 964176192 (represented in binary as 00111001011110000010100101000000). */
public class Solution {
    // you need treat n as an unsigned value
    public int reverseBits(int n) {
        long value = 0;
        for(int i = 0; i < 32; i ++)
        {
            int lastBit = n & 1;
            n = n >> 1;
            value += Math.pow(2, 31 - i) * lastBit;
        }
        return (int)value;
    }
}

/* Given a linked list, remove the nth node from the end of list and return its head. */
public class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode pointer;
        pointer = head;
        for(int i = 0; i < n - 1; i ++){
            pointer = pointer.next;
        }
        ListNode real_pointer = head;
        ListNode pre = null;
        while(pointer.next != null){
            pointer = pointer.next;
            pre = real_pointer;
            real_pointer = real_pointer.next;
        }
        if(pre == null){
            head = head.next;
        }
        else if(real_pointer.next == null){
            pre.next = null;
        }
        else{
            real_pointer = real_pointer.next;
            pre.next = real_pointer;
        }
        return head;
    }
}

/* Write a function to find the longest common prefix string amongst an array of strings. */
public class Solution {
    public String longestCommonPrefix(String[] strs) {
        String prefix = new String();
        if(strs.length == 0)
            return "";
        prefix = strs[0];
        int n;
        for(int i = 0; i < strs.length; i ++){
            String temp = strs[i];
            for(n = 0; n < temp.length() && n < prefix.length(); n ++){
                if(prefix.charAt(n) != temp.charAt(n)){
                    break;
                }
            }
            prefix = prefix.substring(0,n);
        }
        return prefix;
    }
}

/* calculate the total area shaded by two rectangle */
public class Solution {
    public int computeArea(int A, int B, int C, int D, int E, int F, int G, int H) {
        int areaA = Math.abs(C - A) * Math.abs(B - D);
        int areaB = Math.abs(E - G) * Math.abs(F - H);
        int overlap = (Math.min(D, H) - Math.max(B, F)) * (Math.min(C, G) - Math.max(A, E));
        if (overlap < 0)
            return areaA + areaB;
        else if (Math.min(D, H) - Math.max(B, F) < 0 && Math.min(C, G) - Math.max(A, E) < 0)
            return areaA + areaB;
        else
            return areaA + areaB - overlap;
        
    }
}

/* instruction: Given a singly linked list L: L0→L1→…→Ln-1→Ln,
 reorder it to: L0→Ln→L1→Ln-1→L2→Ln-2→… */
public class Solution {
    public ListNode reverse(ListNode head){
        ListNode temp,previous;
        previous = null;
        while(head != null){
            temp = head.next;
            head.next = previous;
            previous = head;
            head = temp;
        }
        return previous;
    }
    public void reorderList(ListNode head) {
        //find the mid point
        if(head == null)
            return;
        ListNode pointer_mid = head, pointer = head, temp, temp1;
        int count = 0;
        while(pointer.next != null){
            pointer = pointer.next;
            count ++;
        }
        count = count / 2;
        while(count > 0){
            pointer_mid = pointer_mid.next;
            count --;
        }
        ListNode head2 = pointer_mid.next;
        pointer_mid.next = null;
        head2 = reverse(head2);
        //link the two linked lists
        pointer = head;
        while(pointer!= null && head2!= null){
            temp = pointer.next;
            pointer.next = head2;
            temp1 = head2.next;
            head2.next = temp;
            pointer = temp;
            head2 = temp1;
        }
    }
}

/******************List Node ends*******************/

/* instruction: Given a roman numeral, convert it to an integer.
 Input is guaranteed to be within the range from 1 to 3999. */

public class Solution {
    public int romanToInt(String s) {
        int sum = 0;
        int len = s.length();
        char a = 'x', temp;
        for(int i = 0; i < len; i ++){
            temp = s.charAt(i);
            if(i < len - 1)
                a = s.charAt(i + 1);
            if(temp == 'I' && a == 'V')
                sum -= 1;
            else if(temp == 'I' && a == 'X')
                sum -= 1;
            else if(temp == 'I')
                sum += 1;
            else if(temp == 'V')
                sum += 5;
            else if(temp == 'X' && a== 'C')
                sum -= 10;
            else if(temp == 'X' && a == 'L')
                sum -= 10;
            else if(temp == 'X')
                sum += 10;
            else if(temp == 'L')
                sum += 50;
            else if(temp == 'C' && a == 'D')
                sum -= 100;
            else if(temp == 'C' && a == 'M')
                sum -= 100;
            else if(temp == 'C')
                sum += 100;
            else if(temp == 'D')
                sum += 500;
            else if(temp == 'M')
                sum += 1000;
            a = 'x';
        }
        return sum;
    }
}

/* Given an array of non-negative integers, you are initially positioned at the first index of the array.
 
 Each element in the array represents your maximum jump length at that position.
 
 Determine if you are able to reach the last index.
 
*/
public class Solution {
    public boolean canJump(int[] A) {
        int length = A.length;
        if(length == 0 || length == 1)
            return true;
        if(A[0] == 0)
            return false;
        for(int i = 1; i < length; i ++)
        {
            A[i] = Math.max(A[i - 1] - 1, A[i]);
            if(A[i] >= length - i - 1)
                return true;
            if(A[i] == 0)
                return false;
        }
        return true;
    }
}

/* instruction: Given an array of non-negative integers, you are initially positioned at the first index of the array.
 
 Each element in the array represents your maximum jump length at that position.
 
 Your goal is to reach the last index in the minimum number of jumps.
*/
public class Solution {
    public int jump(int[] A) {
        int length = A.length;
        if(length <= 1)
            return 0;
        int high = 0, low = 0;
        int count = 0;
        while(high < length - 1)
        {
            int prehigh = high;
            for(int i = low; i <= prehigh; i ++)
            {
                high = Math.max(A[i] + i, high);
            }
            low = prehigh + 1;
            count ++;
        }
        return count;
    }
}

/* instruction: Compare two version numbers version1 and version1.
 If version1 > version2 return 1, if version1 < version2 return -1, otherwise return 0. */
public class Solution {
    public int compareVersion(String version1, String version2) {
        int index1 = 0, index2 = 0;
        int v1 = 0, v2 = 0;
        int result = 0;
        while(result == 0 && (index1 >= 0 || index2 >= 0))
        {
            index1 = version1.indexOf('.');
            index2 = version2.indexOf('.');
            if(index1 < 0)
            {
                v1 = Integer.parseInt(version1);
                version1 = "0";
            }
            else
            {
                v1 = Integer.parseInt(version1.substring(0, index1));
                version1 = version1.substring(index1 + 1);
            }
            if(index2 < 0)
            {
                v2 = Integer.parseInt(version2);
                version2 = "0";
            }
            else
            {
                v2 = Integer.parseInt(version2.substring(0, index2));
                version2 = version2.substring(index2 + 1);
            }
            result = compare(v1, v2);
        }
        return result;
    }
    public int compare(int number1, int number2)
    {
        if(number1 == number2)
            return 0;
        else if(number1 < number2)
            return -1;
        else
            return 1;
    }
}

//Given an input string, reverse the string word by word. A word is defined as a sequence of non-space characters.
//
//The input string does not contain leading or trailing spaces and the words are always separated by a single space.

public class Solution {
    public void reverseWords(char[] s) {
        // first rotate the word
        int start = 0;
        int length = s.length;
        for(int i = 0; i < length; i ++) {
            if(s[i] == ' ') {
                reverse(start, i - 1, s);
                start = i + 1;
                
            }
            else if(i == length - 1) {
                reverse(start, i, s);
            }
        }
        reverse(0, length - 1, s);
    }
    
    public void reverse(int start, int end, char []s) {
        while (start < end) {
            char temp = s[start];
            s[start] = s[end];
            s[end] = temp;
            start ++;
            end --;
        }
    }
}

/* instruction: The gray code is a binary numeral system where two successive values differ in only one bit.
 
 Given a non-negative integer n representing the total number of bits in the code, print the sequence of gray code. A gray code sequence must begin with 0. */
import java.util.List;
import java.util.ArrayList;
public class Solution {
    public List<Integer> grayCode(int n) {
        List<Integer> list = new ArrayList<Integer>();
        list.add(0);
        if(n == 0)
            return list;
        int count = (int)Math.pow(2, n);
        for(int i = 1; i < count; i ++)
            list.add((i >> 1) ^ i);
        return list;
    }
}

/* instruction: Given an array of integers, every element appears three times except for one. Find that single one. */
import java.util.HashMap;
public class Solution {
    public int singleNumber(int[] A) {
        HashMap<Integer, Integer> hm = new HashMap<Integer, Integer>();
        int key = 0, value = 0;
        for(int n = 0; n < A.length; n ++){
            value = A[n];
            key = A[n];
            if(hm.containsKey(key)){
                value = hm.get(key);
                value ++;
            }
            hm.put(key, value);
        }
        int solution = 0;
        for(int n = 0; n < A.length; n ++){
            if(hm.get(A[n]) - A[n] != 2)
                solution = A[n];
        }
        return solution;
    }
}

/* instruction: Given a matrix of m x n elements (m rows, n columns), return all elements of the matrix in spiral order. */
import java.util.List;
import java.util.ArrayList;
public class Solution {
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> list = new ArrayList<Integer>();
        if(matrix == null)
            return list;
        int rowcount = 0, colcount = 0;
        int colsize = matrix.length; //3
        if(colsize == 0)
            return list;
        int rowsize = matrix[0].length;  //1
        while(rowcount < rowsize - rowcount && colcount < colsize - colcount)
        {
            /* from left to right */
            for(int i = rowcount; i < rowsize - rowcount; i ++)
            {
                list.add(matrix[colcount][i]);
            }
            
            /* from up to down */
            for(int i = colcount + 1; i < colsize - colcount; i ++)
            {
                list.add(matrix[i][rowsize - colcount - 1]);
            }
            
            /* from right to left */
            if(colcount < colsize - colcount - 1)
                for(int i = rowsize - rowcount - 2; i >= rowcount; i --)
                {
                    list.add(matrix[colsize - rowcount - 1][i]);
                }
            
            /* from down to up */
            if(rowcount < rowsize - rowcount - 1)
                for(int i = colsize - colcount - 2; i > colcount; i --)
                {
                    list.add(matrix[i][colcount]);
                }
            colcount ++;
            rowcount ++;
        }
        return list;
    }
}

/* instruction: Given an integer n, generate a square matrix filled with elements from 1 to n2 in spiral order. */
public class Solution {
    public int[][] generateMatrix(int n) {
        int [][]matrix  = new int[n][n];
        int count = 0, number = 1;
        while(number <= n * n)
        {
            for(int i = count; i < n - count; i ++)
            {
                matrix[count][i] = number;
                number ++;
            }
            
            for(int i = count + 1; i < n - count; i ++)
            {
                matrix[i][n - count - 1] = number;
                number ++;
            }
            
            for(int i = n - count - 2; i >= count; i --)
            {
                matrix[n - count - 1][i] = number;
                number ++;
            }
            for(int i = n - count - 2; i > count ; i --)
            {
                matrix[i][count] = number;
                number ++;
            }
            count ++;
        }
        return matrix;
    }
}

/* instruction: compute sqrt */
public class Solution {
    public int sqrt(int x) {
        int low = 0, high = x;
        int mid = (low + high) / 2;
        if(x == 1)
            return 1;
        while(low < high)
        {
            mid = low + (high - low) / 2;
            if(mid == x / mid)
                return mid;
            else if((mid + 1) == x / (mid + 1))
                return mid + 1;
            else if(mid < x / mid && (mid + 1) > x / (mid + 1))
                return mid;
            else if(mid > x / mid)
                high = mid + 1;
            else if(mid < x / mid)
                low = mid - 1;
        }
        return mid;
    }
}

/* instruction: Given an input string, reverse the string word by word. */
public class Solution {
    public String reverseWords(String s) {
        if(s == null)
            return "";
        StringBuilder value = new StringBuilder();
        String[] sp = s.trim().split("\\s+");
        int length = sp.length;
        for(int i = length - 1; i >= 0; i --)
        {
            value.append(sp[i]);
            if(i != 0)
                value.append(" ");
        }
        return value.toString();
    }
}

/* instruction: Given an array and a value, remove all instances of that value in place and return the new length. */
import java.util.Arrays;
public class Solution {
    public int removeElement(int[] A, int elem) {
        if(A.length == 0)
            return 0;
        int count = 0;
        for(int i = 0; i < A.length; i ++){
            if(A[i] == elem){
                count ++;
                continue;
            }
            if(count > 0){
                A[i - count] = A[i];
            }
        }
        return A.length - count;
    }
}

/* instruction: Given n non-negative integers representing the histogram's bar height where the width of each bar is 1, find the area of largest rectangle in the histogram. */
public class Solution {
    public int largestRectangleArea(int[] height) {
        int length = height.length;
        if(length == 0)
            return 0;
        int area = 0, i = 0;
        Stack<Integer> stack = new Stack<Integer>();
        while(i < length)
        {
            if(stack.isEmpty() || height[i] >= height[stack.peek()])
            {
                stack.push(i);
                i ++;
            }
            else
            {
                int high = stack.pop();
                int size = height[high] * (stack.isEmpty()? i: i - stack.peek() - 1);
                area = Math.max(size, area);
            }
        }
        while(!stack.isEmpty())
        {
            int high = stack.pop();
            int size = height[high] * (stack.isEmpty()? i: i - stack.peek() - 1);
            area = Math.max(size, area);
        }
        return area;
    }
}

/* instruction:
 Given an array of strings, return all groups of strings that are anagrams.
 
 Note: All inputs will be in lower-case.
 */
import java.util.*;
public class Solution {
    public List<String> anagrams(String[] strs) {
        HashMap<String, List<String>> hash = new HashMap<String, List<String>>();
        List<String> list = new LinkedList<String>();
        int length = strs.length;
        if(length == 0)
            return list;
        for(String s : strs)
        {
            char []temp = s.toCharArray();
            Arrays.sort(temp);
            String s1 = new String(temp);
            if(!hash.containsKey(s1))
            {
                List<String> l = new LinkedList<String>();
                l.add(s);
                hash.put(s1, l);
            }
            else
            {
                List<String> l = hash.get(s1);
                l.add(s);
                hash.put(s1, l);
            }
        }
        Iterator it = hash.entrySet().iterator();
        while(it.hasNext())
        {
            Map.Entry pairs = (Map.Entry)it.next();
            List<String> l = (List<String>)pairs.getValue();
            int size = l.size();
            if(size > 1)
            {
                list.addAll(l);
            }
            it.remove();
        }
        return list;
        
    }
}

/* instruction: Evaluate the value of an arithmetic expression in Reverse Polish Notation. */
import java.util.Stack;
public class Solution {
    public int evalRPN(String[] tokens) {
        int len = tokens.length;
        Stack<Integer> stack = new Stack<Integer>();
        int nTokens = 0;
        if(len == 0)
            return 0;
        for(int i = 0; i < len; i ++)
        {
            if(tokens[i].equals("+"))
            {
                if(nTokens < 2)
                    return 0;
                int first = stack.pop();
                int second = stack.pop();
                int result = first + second;
                stack.push(result);
                nTokens --;
            }
            else if(tokens[i].equals("-"))
            {
                if(nTokens < 2)
                    return 0;
                int first = stack.pop();
                int second = stack.pop();
                int result = second - first;
                stack.push(result);
                nTokens --;
            }
            else if(tokens[i].equals("*"))
            {
                if(nTokens < 2)
                    return 0;
                int first = stack.pop();
                int second = stack.pop();
                int result = first * second;
                stack.push(result);
                nTokens --;
            }
            else if(tokens[i].equals("/"))
            {
                if(nTokens < 2)
                    return 0;
                int first = stack.pop();
                int second = stack.pop();
                if(first == 0)
                    return Integer.MAX_VALUE;
                int result = second / first;
                stack.push(result);
                nTokens --;
            }
            else
            {
                stack.push(Integer.parseInt(tokens[i]));
                nTokens ++;
            }
        }
        return stack.pop();
    }
}

/* instruction: A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).
 The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).
 How many possible unique paths are there? */

public class Solution {
    public int c(int m, int n){
        double result = 1;
        if(m == n)
            return 1;
        for(int count = 1; count <= n; count ++){
            result = result * (m - n + count);
            result = result / count;
        }
        return (int)result;
    }
    public int uniquePaths(int m, int n) {
        //compute c(m + n - 2,n - 1)
        int a = n - 1, b = m + n - 2;
        if(a > b / 2){
            a = b - a;
        }
        return c(b, a);
    }
}

/* instruction: Given a triangle, find the minimum path sum from top to bottom. Each step you may move to adjacent numbers on the row below. */
public class Solution {
    public int minimumTotal(List<List<Integer>> triangle) {
        int size = triangle.size();
        if(size == 0)
            return 0;
        if(size == 1)
            return triangle.get(0).get(0);
        int []minimumPath = new int[size];
        for(int i = 0; i < size; i ++)
        {
            minimumPath[i] = triangle.get(size - 1).get(i);
        }
        for(int j = size - 2; j >= 0; j --)
        {
            for(int i = 0; i < triangle.get(j).size() ; i ++)
            {
                minimumPath[i] = Math.min(minimumPath[i + 1], minimumPath[i]) + triangle.get(j).get(i);
            }
        }
        return minimumPath[0];
    }
}

/* instruction: Given n non-negative integers a1, a2, ..., an, where each represents a point at coordinate (i, ai). n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). Find two lines, which together with x-axis forms a container, such that the container contains the most water. */
public class Solution {
    public int maxArea(int[] height) {
        int len = height.length, area, max = 0;
        int high = len - 1, low = 0;
        while(low < high){
            area = (high - low) * Math.min(height[high], height[low]);
            if(height[low] < height[high])
                low ++;
            else
                high --;
            if(area > max)
                max = area;
        }
        return max;
    }
}

/* instruction: Given a non-negative number represented as an array of digits, plus one to the number. */
public class Solution {
    public int[] plusOne(int[] digits) {
        int len = digits.length;
        if(len == 0){
            digits[0] = 1;
            return digits;
        }
        digits[len - 1] ++;
        for(int i = len - 1; i > 0; i --){
            if(digits[i] == 10){
                digits[i] = 0;
                digits[i - 1] ++;
            }
        }
        int ret[];
        if(digits[0] == 10){
            ret = new int[len + 1];
            ret[0] = 1;
            ret[1] = 0;
            System.arraycopy(digits, 1, ret ,2 ,len - 1);
        }
        else{
            ret = new int[len];
            System.arraycopy(digits, 0, ret, 0, len);
        }
        return ret;
    }
}

//Given an array of strings, group anagrams together.
//
//For example, given: ["eat", "tea", "tan", "ate", "nat", "bat"],
//Return:
//
//[
//["ate", "eat","tea"],
//["nat","tan"],
//["bat"]
//]

public class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        List<List<String>> lls = new ArrayList<List<String>>();
        HashMap<String, List<String>> hm = new HashMap<String, List<String>>();
        Arrays.sort(strs);
        for (String str : strs) {
            char[] array = str.toCharArray();
            Arrays.sort(array);
            String temp = new String(array);
            if (hm.containsKey(temp)) {
                hm.get(temp).add(str);
            }
            else {
                List<String> a = new ArrayList<String>();
                a.add(str);
                hm.put(temp, a);
            }
        }
        for(Map.Entry<String, List<String>> entry: hm.entrySet()) {
            lls.add(entry.getValue());
        }
        return lls;
    }
}

/* instruction: Given an integer, convert it to a roman numeral. */
import java.util.HashMap;
public class Solution {
    public String intToRoman(int num) {
        HashMap roman = new HashMap<Integer, String>();
        roman.put(1, "I");
        roman.put(4, "IV");
        roman.put(5, "V");
        roman.put(9, "IX");
        roman.put(10, "X");
        roman.put(40, "XL");
        roman.put(50, "L");
        roman.put(90, "XC");
        roman.put(100, "C");
        roman.put(400, "CD");
        roman.put(500, "D");
        roman.put(900, "CM");
        roman.put(1000, "M");
        String stret = "";
        int len = 0;
        int ano_num = num;
        while(ano_num != 0){
            ano_num = ano_num / 10;
            len ++;
        }
        int []number = new int[len];
        ano_num = num;
        for (int i = len - 1; i >= 0; i --){
            number[i] = ano_num % 10;
            ano_num /= 10;
        }
        for (int i = 0; i < len; i ++){
            if(number[i] >= 5 && number[i] < 9){
                stret += roman.get(5 * (int)Math.pow(10, len - i - 1));
                int rest = number[i] - 5;
                for(int j = 0; j < rest; j ++){
                    stret += roman.get((int)Math.pow(10, len - i - 1));
                }
            }
            else if(number[i] == 4 || number[i] == 9){
                stret += roman.get(number[i] * (int)Math.pow(10, len - i - 1));
            }
            else{
                for(int j = 0; j < number[i]; j ++){
                    stret += roman.get((int)Math.pow(10, len - i - 1));
                }
            }
        }
        return stret;
    }
    
}

/* Given a sorted array and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order. */
public class Solution {
    public int searchInsert(int[] A, int target) {
        int position = 0;
        if(A == null)
            return position;
        if(A.length == 1){
            if(target <= A[0])
                return position;
            else
                return position + 1;
        }
        for(int n = 1; n < A.length; n ++){
            if((target <= A[n]) && (target > A[n - 1])){
                position = n;
                break;
            }
            else if(target <= A[n - 1]){
                position = n - 1;
                break;
            }
            else{
                position = n + 1;
            }
        }
        return position;
    }
}

/* instruction: Reverse digits of an integer. */
public class Solution {
    public int reverse(int x) {
        int neg, temp = x, count = 0, ans = 0, temp1;
        if(x < 0){
            neg = 1;
        }
        else{
            neg = 0;
        }
        while(temp != 0){
            temp = temp / 10;
            count ++;
        }
        temp = x;
        if( x > 0x7fffffff){
            return ans;
        }
        while(temp != 0){
            if((temp %10 != 0) || (ans != 0)){
                temp1 = (int)Math.pow(10, count - 1);
                ans = ans + (temp % 10) * temp1;
                temp = temp / 10;
                count --;
            }
            else{
                count --;
                temp = temp / 10;
            }
        }
        ans = ans | (neg << 31);
        return ans;
    }
}

/* instruction:
 Given a set of candidate numbers (C) and a target number (T), find all unique combinations in C where the candidate numbers sums to T.
 
 The same repeated number may be chosen from C unlimited number of times.
 */
public class Solution {
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        Arrays.sort(candidates);
        List<List<Integer>> list = new LinkedList<List<Integer>>();
        int length = candidates.length;
        for(int i = 0; i < length; i ++)
        {
            if(candidates[i] > target)
            {
                continue;
            }
            else if(candidates[i] == target)
            {
                List<Integer> l = new LinkedList<Integer>();
                l.add(target);
                list.add(l);
            }
            else
            {
                int array[] = new int[length - i];
                System.arraycopy(candidates, i, array, 0, length - i);
                List<List<Integer>> temp = combinationSum(array, target - candidates[i]);
                for(List<Integer> l: temp)
                {
                    l.add(0, candidates[i]);
                    list.add(l);
                }
            }
        }
        return list;
        
    }
}

/* instruction: Given a collection of candidate numbers (C) and a target number (T), find all unique combinations in C where the candidate numbers sums to T.
 
 Each number in C may only be used once in the combination.
 Duplicates are allowed */
public class Solution {
    private List<List<Integer>> list;
    private HashSet<List<Integer>> hash;
    public List<List<Integer>> combinationSum2(int[] num, int target) {
        int length = num.length;
        list = new LinkedList<List<Integer>>();
        if(length == 0)
            return list;
        Arrays.sort(num);
        List<Integer> l = new LinkedList<Integer>();
        hash = new HashSet<List<Integer>>();
        FindCombination(l, target, 0, num);
        return list;
    }
    public void FindCombination(List<Integer> l, int target, int last, int []num)
    {
        if(target == 0 && !hash.contains(l))
        {
            list.add(l);
            hash.add(l);
        }
        for(int i = last; i < num.length; i ++)
        {
            if(target < num[i])
                return;
            List<Integer> list = new LinkedList<Integer>();
            list.addAll(l);
            list.add(num[i]);
            FindCombination(list, target - num[i], i + 1, num);
        }
    }
}



/* new version */
/* find max - min in JAVA, O(n) time complexity */

public int maxProfit(int[] prices){
    if(prices.length <= 0)
        return 0;
    int len = prices.length;
    int low = prices[0];
    int max_profit = 0;
    int profit = 0;
    for(int i = 1; i < len; i ++){
        if (prices[i] < low)
            low = prices[i];
        profit = prices[i] - low;
        if(profit > max_profit)
            max_profit = profit;
    }
    return max_profit;
}

/* Say you have an array for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete at most two transactions. */
public class Solution {
    public int maxProfit(int[] prices) {
        int length = prices.length;
        if(length == 0)
            return 0;
        int never = Integer.MIN_VALUE;
        int one_transaction_hold_early = Integer.MIN_VALUE;
        int two_transaction = Integer.MIN_VALUE;
        int one_transaction_hold_late = Integer.MIN_VALUE;

        for(int i = 0; i < length; i ++)
        {
            never = Math.max(never, -prices[i]);
            one_transaction_hold_early = Math.max(one_transaction_hold_early, never + prices[i]);
            one_transaction_hold_late = Math.max(one_transaction_hold_late, one_transaction_hold_early - prices[i]);
            two_transaction = Math.max(two_transaction, one_transaction_hold_late + prices[i]);
        }
        return Math.max(never, Math.max(one_transaction_hold_early, two_transaction));
    }
}

/* instruction: Given a binary tree, find its minimum depth. */
public class Solution {
    public int minDepth(TreeNode root) {
        if(root == null)
            return 0;
        if(root.left == null && root.right == null)
            return 1;
        int left = minDepth(root.left);
        int right = minDepth(root.right);
        if(left > 0 && right > 0)
            return Math.min(left, right) + 1;
        else
            return Math.max(left, right) + 1;
    }
}

/* instruction: Given a string s consists of upper/lower-case alphabets and empty space characters ' ', 
 return the length of last word in the string.
 If the last word does not exist, return 0.
 Note: A word is defined as a character sequence consists of non-space characters only.
 */

import java.util.StringTokenizer;
public class Solution {
    public int lengthOfLastWord(String s) {
        if(s.equals("") || s.equals(" "))
            return 0;
        StringTokenizer st = new StringTokenizer(s, " ");
        String out = "";
        while(st.hasMoreTokens()){
            out = st.nextToken();
        }
        if(s.equals(""))
            return 0;
        return out.length();
    }
}

/* instruction: Determine whether an integer is a palindrome. Do this without extra space. */
public class Solution {
    public boolean isPalindrome(int x) {
        if(x < 0)
            return false;
        int temp = x;
        int length = 0;
        while(temp != 0){
            length += 1;
            temp /= 10;
        }
        temp = x;
        int answer = 0;
        int sum = 0;
        int rest = 0;
        for(int i = 1; i <= length/2; i ++){
            answer = temp % 10;
            sum = answer * (int)Math.pow(10, i - 1) + answer * (int)Math.pow(10, length - i) + sum;
            temp = temp / 10;
        }
        if(length % 2 == 1){
            rest =  (temp % 10) * (int)Math.pow(10, length/2);
        }
        sum += rest;
        return sum == x;
    }
}

/* instruction: Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid. */
import java.util.Stack;
public class Solution {
    public boolean isValid(String s) {
        Stack<Character> mem = new Stack();
        int len = s.length();
        if(len == 0)
            return false;
        for(int i = 0; i <= len - 1; i ++){
            char a = s.charAt(i);
            switch(a){
                case '(':
                    mem.push(a);
                    break;
                case '[':
                    mem.push(a);
                    break;
                case '{':
                    mem.push(a);
                    break;
                case '}':
                    if(mem.isEmpty())
                        return false;
                    if(mem.peek() == '(' || mem.peek() == '[')
                        return false;
                    mem.pop();
                    break;
                case ']':
                    if(mem.isEmpty())
                        return false;
                    if(mem.peek() == '(' || mem.peek() == '{')
                        return false;
                    mem.pop();
                    break;
                case ')':
                    if(mem.isEmpty())
                        return false;
                    if(mem.peek() == '[' || mem.peek() == '{')
                        return false;
                    mem.pop();
                    break;
                default:
                    return false;
            }
        }
        return mem.isEmpty();
    }
}

/* instruction: Given sorted array A = [1,1,1,2,2,3],
 
 Your function should return length = 5, and A is now [1,1,2,2,3].*/
import java.util.HashMap;
public class Solution {
    public int removeDuplicates(int[] A) {
        HashMap<Integer, Integer> hm = new HashMap<Integer, Integer>();
        int length = A.length, i = 0;
        while (i < length) {
            if (hm.containsKey(A[i]))
                hm.put(A[i], hm.get(A[i]) + 1);
            else
                hm.put(A[i], 1);
            if (hm.get(A[i]) > 2) {
                remove(A, i);
                length--;
            } else {
                i++;
            }
        }
        return length;
    }
    public void remove(int []A, int i)
    {
        int length = A.length;
        for(int k = i + 1; k < length; k ++)
        {
            A[k - 1] = A[k];
        }
    }
}

/* instruction: zigzag convertion */
public class Solution {
    public String convert(String s, int nRows) {
        String tempt = new String();
        int len = s.length();
        int i, j;
        if(s == null || nRows >= len || nRows == 1)
            return s;
        //Row = 0
        for(i = 0; i < len; i += 2 * (nRows - 1)){
            tempt += s.charAt(i);
        }
        //Row = 1..nRows - 1
        int Rows;
        for(Rows = 1; Rows <= nRows - 2; Rows ++){
            for(i = Rows, j = 2 *(nRows - 1) - Rows; i < len; i += 2 * (nRows - 1), j += 2 * (nRows - 1)){
                tempt += s.charAt(i);
                if(j < len)
                    tempt += s.charAt(j);
            }
        }
        //Row = nRows
        for(i = nRows - 1; i < len; i += 2 * (nRows - 1)){
            tempt += s.charAt(i);
        }
        return tempt;
    }
}

/* instruction: Given an array S of n integers, find three integers in S such that the sum is closest to a given number, target. Return the sum of the three integers. You may assume that each input would have exactly one solution. */
import java.util.Arrays;
public class Solution {
    public int threeSumClosest(int[] num, int target) {
        int length = num.length;
        if(length < 3)
            throw new ArrayIndexOutOfBoundsException("needs at least 3 elements");
        Arrays.sort(num);
        int result = num[0] + num[1] + num[length - 1];
        int min_diff = Math.abs(result - target);
        int value = result;
        for(int i = 0; i < length - 2; i ++)
        {
            int start = i + 1, end = length - 1;
            while(start < end)
            {
                result = num[i] + num[start] + num[end];
                if(result > target)
                    end --;
                else
                    start ++;
                int diff = Math.abs(result - target);
                if(diff < min_diff)
                {
                    min_diff = diff;
                    value = result;
                }
            }
        }
        return value;
        
    }
}

/* instruction: Count the number of prime numbers less than a non-negative number, n. */
public class Solution {
    public int countPrimes(int n) {
        if(n <= 1)
            return 0;
        boolean prime[] = new boolean[n + 1];
        for(int i = 2; i < n; i ++)
            prime[i] = true;
        int m = (int)Math.sqrt(n);
        for(int i = 2; i <= m; i ++)
        {
            if(prime[i])
            {
                for(int j = i; j * i <= n; j ++)
                {
                    prime[j * i] = false;
                }
            }
        }
        int count = 0;
        for(int i = 2; i <= n; i ++)
            if(prime[i])
                count ++;
        return count;
    }
}

/* instruction: Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero. */
import java.util.List;
import java.util.LinkedList;
import java.util.Arrays;
public class Solution {
    public List<List<Integer>> threeSum(int[] num) {
        List<List<Integer>> list = new LinkedList<List<Integer>>();
        HashSet<List<Integer>> hash = new HashSet<List<Integer>>();
        int length = num.length;
        if(length < 3)
            return list;
        int result;
        Arrays.sort(num);
        for(int i = 0; i < length - 2; i ++)
        {
            int start = i + 1;
            int end = length - 1;
            while(start < end)
            {
                result = num[start] + num[end] + num[i];
                if(result < 0)
                    start ++;
                else if(result > 0)
                    end --;
                else
                {
                    List<Integer> l = new LinkedList<Integer>();
                    l.add(num[i]);
                    l.add(num[start]);
                    l.add(num[end]);
                    if(!hash.contains(l))
                    {
                        list.add(l);
                        hash.add(l);
                    }
                    start ++;
                    end --;
                }
            }
        }
        return list;
    }
}

/* instruction:
 The count-and-say sequence is the sequence of integers beginning as follows:
 1, 11, 21, 1211, 111221, ...
 
 1 is read off as "one 1" or 11.
 11 is read off as "two 1s" or 21.
 21 is read off as "one 2, then one 1" or 1211.
 Given an integer n, generate the nth sequence.
 */
public class Solution {
    public String countAndSay(int n) {
        StringBuffer sb = new StringBuffer();
        StringBuffer temp = new StringBuffer();
        int count = 1;
        if(n == 0)
            return sb.toString();
        sb.append(1);
        temp.append(1);
        if(n == 1)
            return sb.toString();
        for(int i = 1; i < n; i ++){
            sb.delete(0, sb.length());
            for(int j = 0; j < temp.length() - 1; j ++){
                if(temp.charAt(j) == temp.charAt(j + 1))
                    count ++;
                else{
                    sb.append(count);
                    sb.append(temp.charAt(j));
                    count = 1;
                }
            }
            sb.append(count);
            sb.append(temp.charAt(temp.length() - 1));
            if( i != n - 1){
                temp.delete(0, sb.length());
                temp.append(sb);
            }
            count = 1;
        }
        return sb.toString();
    }
}

/* instruction: Given an array of size n, find the majority element. The majority element is the element that appears more than ⌊ n/2 ⌋ times. */
import java.util.HashMap;
public class Solution {
    public int majorityElement(int[] num) {
        HashMap<Integer, Integer> hm = new HashMap<Integer, Integer>();
        int length = num.length;
        for(int i = 0; i < length; i ++)
        {
            if(hm.containsKey(num[i]))
            {
                hm.put(num[i], hm.get(num[i]) + 1);
            }
            else
            {
                hm.put(num[i], 1);
            }
            if(hm.get(num[i]) > length / 2)
                return num[i];
        }
        return -1;
    }
}

/* instruction: Given two binary strings, return their sum (also a binary string). */
public String addBinary(String a, String b) {
    if(a == null || b == null)
        return b == null ? a:b;
    int lena = a.length();
    int lenb = b.length();
    StringBuffer a1;
    StringBuffer b1;
    if(lena < lenb){
        a1 = new StringBuffer(b);
        b1 = new StringBuffer(a);
    }
    else{
        a1 = new StringBuffer(a);
        b1 = new StringBuffer(b);
    }
    for(int i = 0; i < Math.abs(lena - lenb); i ++)
        b1.insert(0, '0');
    int length = Math.max(lena, lenb);

    boolean flag = false;
    StringBuffer sum = new StringBuffer();
    for(int i = length - 1; i >= 0; i --){
        if(a1.charAt(i) == '1' && b1.charAt(i) == '1' && flag == true){
            flag = true;
            sum.insert(0, '1');
        }
        else if(a1.charAt(i) == '1' && b1.charAt(i) == '1'){
            flag = true;
            sum.insert(0, '0');
        }
        else if(a1.charAt(i) == '1' && b1.charAt(i) == '0' && flag == true){
            flag = true;
            sum.insert(0, '0');
        }
        else if(a1.charAt(i) == '0' && b1.charAt(i) == '1' && flag == true){
            flag = true;
            sum.insert(0, '0');
        }
        else if(a1.charAt(i) == '0' && b1.charAt(i) == '0' && flag == false){
            flag = false;
            sum.insert(0, '0');
        }
        else{
            flag = false;
            sum.insert(0, '1');
        }
    }
    if(flag)
        sum.insert(0, '1');
    return sum.toString();

}

/* instruction: Given an integer n, return the number of trailing zeroes in n!. */
public class Solution {
    public int trailingZeroes(int n) {
        int count = 0;
        for(int i = 5; i <= n; i = i * 5)
        {
            count += n / i;
        }
        return count;
    }
}

/* instruction:Now consider if some obstacles are added to the grids. How many unique paths would there be?
 
 An obstacle and empty space is marked as 1 and 0 respectively in the grid.
 
 For example,
 There is one obstacle in the middle of a 3x3 grid as illustrated below.
 
 [
 [0,0,0],
 [0,1,0],
 [0,0,0]
 ]
 */

public class Solution {
    public int uniquePathsWithObstacles(int[][] obstacleGrid)
    {
        int row = obstacleGrid.length;
        int column = obstacleGrid[0].length;
        if(row == 0 && column == 0)
            return 0;
        for(int i = 0; i < row; i ++)
        {
            for(int j = 0; j < column; j ++)
            {
                if(obstacleGrid[i][j] == 1)
                    obstacleGrid[i][j] = 0;
                else if(i == 0 && j == 0)
                    obstacleGrid[i][j] = 1;
                else if(i == 0)
                    obstacleGrid[i][j] = obstacleGrid[i][j - 1];
                else if(j == 0)
                    obstacleGrid[i][j] = obstacleGrid[i - 1][j];
                else
                    obstacleGrid[i][j] = obstacleGrid[i][j - 1] + obstacleGrid[i - 1][j];
                
            }
        }
        return obstacleGrid[row - 1][column - 1];
    }
}

/* instruction: Given a positive integer, return its corresponding column title as appear in an Excel sheet. */
public class Solution {
    public String convertToTitle(int n) {
        StringBuilder sb = new StringBuilder();
        while(n != 0) {
            int convert = n % 26;
            if(convert == 0)
            {
                sb.insert(0, 'Z');
                n = n / 26 - 1;
            }
            else
            {
                sb.insert(0, Character.toChars(convert + 'A' - 1));
                n /= 26;
            }
        }
        return sb.toString();
    }
}

/* instruction: Implement atoi to convert a string to an integer. */
public class Solution {
    public int atoi(String str) {
        if(str.equals(""))
            return 0;
        str = str.trim();
        if(str == null)
            return 0;
        int result = 0;
        long compare  = 0;
        int length = str.length(), start = 0, end = 0;
        boolean flag = false;
        if(str.charAt(0) == '-')
        {
            flag = true;
            start = 1;
        }
        else if(str.charAt(0) == '+')
            start = 1;
        for(int i = start; i < length; i ++)
        {
            if(!Character.isDigit(str.charAt(i)))
            {
                end = i;
                break;
            }
            if(i == length - 1)
                end = length;
        }
        if(start == end && end != length)
            return 0;
        String number = str.substring(start, end);
        length = number.length();
        for(int i = 0; i < length; i ++)
        {
            result += Character.digit(number.charAt(i), 10) * Math.pow(10, length - i - 1);
            compare += Character.digit(number.charAt(i), 10) * Math.pow(10, length - i - 1);
        }
        
        if(flag && result == Integer.MAX_VALUE && compare != result)
        {
            result = Integer.MIN_VALUE;
        }
        else if(flag)
        {
            result = -result;
        }
        return result;
    }
}

/* instruction: Given a column title as appear in an Excel sheet, return its corresponding column number. */
public class Solution {
    public int titleToNumber(String s) {
        if(s == null)
            return 0;
        int length = s.length();
        s = s.toLowerCase();
        int number = 0;
        for(int i = 0; i < length; i ++)
        {
            number += (s.charAt(i) - 'a' + 1) * Math.pow(26, length - i - 1);
        }
        return number;
    }
}

/* Given a set of distinct integers, S, return all possible subsets.
 
 Note:
 Elements in a subset must be in non-descending order.
 The solution set must not contain duplicate subsets.
 */
import java.util.List;
import java.util.Arrays;
import java.util.ArrayList;
public class Solution {
    public List<List<Integer>> subsets(int[] S) {
        List<List<Integer>> aai = new ArrayList<List<Integer>>();
        List<Integer> array = new ArrayList<Integer>();
        int len = S.length;
        if(len == 0)
            return aai;
        Arrays.sort(S);
        aai = getSubset(S, len - 1);
        aai.add(array);
        return aai;
    }
    public List<List<Integer>> getSubset(int[] s, int maxidx){
        List<List<Integer>> value = new ArrayList<List<Integer>>();
        if(maxidx < 0)
            return value;
        List<List<Integer>> subsets = getSubset(s, maxidx - 1);
        value.addAll(subsets);
        for(int i = 0; i < subsets.size(); i ++){
            List<Integer> sub = new ArrayList<Integer>();
            sub.addAll(subsets.get(i));
            sub.add(s[maxidx]);
            value.add(sub);
        }
        value.add (Arrays.asList(s[maxidx]));
        return value;
    }
}

/* instruction: Given a collection of integers that might contain duplicates, S, return all possible subsets */
import java.util.List;
import java.util.LinkedList;
import java.util.HashSet;
import java.util.Arrays;
public class Solution {
    public List<List<Integer>> subsetsWithDup(int[] num) {
        List<Integer> l = new LinkedList<Integer>();
        List<List<Integer>> list = new LinkedList<List<Integer>>();
        int length = num.length;
        if(length == 0)
        {
            return list;
        }
        Arrays.sort(num);
        list = getSubsets(num, length - 1);
        removeDuplicates(list);
        list.add(l);
        return list;
    }
    public void removeDuplicates(List<List<Integer>> list)
    {
        HashSet<List<Integer>> hash = new HashSet<List<Integer>>();
        int size = list.size();
        int count = 0;
        for(int i = 0; i < size - count; i ++)
        {
            if(!hash.contains(list.get(i)))
            {
                hash.add(list.get(i));
            }
            else
            {
                list.remove(i);
                i --;
                count ++;
            }
        }
    }
    public List<List<Integer>> getSubsets(int []num, int maxIndex)
    {
        List<List<Integer>> list = new LinkedList<List<Integer>>();
        if(maxIndex < 0)
            return list;
        List<List<Integer>> subsets = getSubsets(num, maxIndex - 1);
        list.addAll(subsets);
        int size = subsets.size();
        for(int i = 0; i < size; i ++)
        {
            List<Integer> sub = new LinkedList<Integer>();
            sub.addAll(subsets.get(i));
            sub.add(num[maxIndex]);
            list.add(sub);
        }
        list.add(Arrays.asList(num[maxIndex]));
        return list;
    }
}

/* instruction: Implement strStr().
 
 Returns the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack. */
public class Solution {
    public int strStr(String haystack, String needle) {
        if(needle.length() > haystack.length())
            return -1;
        int haylen = haystack.length();
        int neelen = needle.length();
        if(neelen == 0)
            return 0;
        int i = 0, j = 0;
        int return_pos = -1;
        boolean reset = true;
        while(true){
            if(j == neelen)
                break;
            if(i == haylen)
                return -1;
            if(haystack.charAt(i) == needle.charAt(j)){
                if(reset){
                    return_pos = i;
                    reset = false;
                }
                i ++;
                j ++;
            }
            else{
                j = 0;
                if(!reset)
                    i = return_pos + 1;
                else
                    i ++;
                reset = true;
            }
        }
        return return_pos;
    }
}

/* instruction: Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.
 
 push(x) -- Push element x onto stack.
 pop() -- Removes the element on top of the stack.
 top() -- Get the top element.
 getMin() -- Retrieve the minimum element in the stack.
 */

import java.util.Stack;
import java.util.ArrayList;

class MinStack {
    Stack<Integer> stack = new Stack<Integer>();
    ArrayList<Integer> al = new ArrayList<Integer>();
    int index = -1;
    public void push(int x) {
        if(al.size() == 0){
            al.add(x);
            index ++;
        }
        else if(al.get(index) >= x){
            al.add(x);
            index ++;
        }
        stack.push(x);
    }
    
    public void pop() {
        int member = stack.pop();
        if(member == al.get(index)){
            al.remove(index);
            index --;
        }
    }
    
    public int top() {
        return stack.peek();
    }
    
    public int getMin() {
        if(index < 0)
            return -1;
        return al.get(index);
    }
}

/* instruction :Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.
 */
public class Solution {
    public boolean isPalindrome(String s) {
        if(s.length() == 0)
            return true;
        s = s.replaceAll(" ", "");
        s = s.replaceAll("[^A-Za-z0-9 ]", "");
        s = s.toLowerCase();
        int start = 0;
        int end = s.length() - 1;
        while(start != end && start < end){
            if(s.charAt(start) != s.charAt(end))
                return false;
            start ++;
            end --;
        }
        return true;
    }
}

/* instruction: 
 Suppose a sorted array is rotated at some pivot unknown to you beforehand.
 
 (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
 
 You are given a target value to search. If found in the array return its index, otherwise return -1.
 
 You may assume no duplicate exists in the array.
 */
public class Solution {
    public int search(int[] A, int target) {
        int start = 0, end = A.length - 1;
        return binarySearch(A, start, end, target);
    }
    public int binarySearch(int []A, int start, int end, int target) {
        if (start > end)
            return -1;
        int mid = (start + end) / 2;
        if (A[mid] == target)
            return mid;
        if(A[end] == target)
            return end;
        if(A[start] == target)
            return start;
        if (A[mid] > A[start]) {
            if (target > A[start] && target < A[mid])
                return binarySearch(A, start + 1, mid, target);
            else
                return binarySearch(A, mid + 1, end, target);
        }
        if (A[end] > A[mid])
        {
            if(target > A[mid] && target < A[end])
                return binarySearch(A, mid + 1, end, target);
            else
                return binarySearch(A, start, mid - 1, target);
        }
        return -1;
    }
}

/* instruction: same as above but duplicates are allowed */
public class Solution {
    public boolean search(int[] A, int target) {
        int start = 0, end = A.length - 1;
        return binarySearch(A, start, end, target);
    }
    public boolean binarySearch(int []A, int start, int end, int target) {
        if (start > end)
            return false;
        int mid = (start + end) / 2;
        if (A[mid] == target || A[start] == target || A[end] == target)
            return true;
        if (A[mid] > A[start]) {
            if(target < A[mid] && target > A[start])
                return binarySearch(A, start + 1, mid, target);
            else
                return binarySearch(A, mid + 1, end, target);
        }
        else if (A[end] > A[mid]) {
            if(target > A[mid] && target < A[end])
                return binarySearch(A, mid + 1, end, target);
            else
                return binarySearch(A, start + 1, mid, target);
        }
        else
        {
            boolean p = binarySearch(A, start + 1, mid, target);
            boolean q = binarySearch(A, mid + 1, end, target);
            if (p)
                return p;
            if (q)
                return q;
        }
        return false;
    }
}

/* instruction: Rotate an array of n elements to the right by k steps. */
public class Solution {
    public void rotate(int[] nums, int k) {
        int length = nums.length;
        if(length == 0)
            return;
        if(k > length)
            k = k % length;
        if(k > length)
            k = length;
        rotateArray(nums, 0, length - k);
        rotateArray(nums, length - k , length);
        rotateArray(nums, 0, length);
        
    }
    public void rotateArray(int []nums, int start, int end)
    {
        for(int i = start; i < (start + end) / 2; i ++)
        {
            int temp = nums[end - (i - start) - 1];
            nums[end - (i - start) - 1] = nums[i];
            nums[i] = temp;
        }
    }
}

/* instruction:
 Suppose a sorted array is rotated at some pivot unknown to you beforehand.
 
 (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
 
 Find the minimum element.
 
 The array may contain duplicates.
 */
public class Solution {
    public int findMin(int[] num) {
        int length = num.length;
        if(length == 0)
            return 0;
        int target = num[0];
        return findMinNumber(num, 0, length - 1, target);
    }
    public int findMinNumber(int[] num, int start, int end, int target)
    {
        if(start > end)
            return target;
        if(num[start] < target)
            target = num[start];
        if(num[end] < target)
            target = num[end];
        int mid = (start + end) / 2;
        if(num[mid] < target)
            target = num[mid];
        if(num[mid] > num[start])
        {
            if(target < num[mid] && target > num[start])
                return findMinNumber(num, start + 1, mid, target);
            else
                return findMinNumber(num, mid + 1, end, target);
        }
        else if(num[end] > num[mid])
        {
            if(target < num[end] && target > num[mid])
                return findMinNumber(num, mid + 1, end, target);
            else
                return findMinNumber(num, start + 1, mid, target);
        }
        else
        {
            int q = findMinNumber(num, start + 1, mid ,target);
            int p = findMinNumber(num, mid + 1, end, target);
            if(p < q)
                return p;
            else
                return q;
        }
    }
}

/* instruction:
 Given two integers representing the numerator and denominator of a fraction, return the fraction in string format.
 
 If the fractional part is repeating, enclose the repeating part in parentheses.
 
 For example,
 
 Given numerator = 1, denominator = 2, return "0.5".
 Given numerator = 2, denominator = 1, return "2".
 Given numerator = 2, denominator = 3, return "0.(6)".
 */
public class Solution {
    public String fractionToDecimal(int numerator, int denominator) {
        StringBuilder result = new StringBuilder();
        if(denominator == 0)
            return result.toString();
        if((numerator < 0 && denominator > 0) || (denominator < 0 && numerator > 0))
        {
            result.append("-");
        }
        long num = (long)numerator;
        long den = (long)denominator;
        num = Math.abs(num);
        den = Math.abs(den);
        result.append(num / den);
        if(num % den == 0)
            return result.toString();
        result.append(".");
        HashMap<Long, Integer> hs = new HashMap<Long, Integer>();
        num = num % den;
        while(num % den != 0)
        {
            num = num * 10;
            long rest = num / den;
            long remain = num % den;
            long test = rest * 10 + remain;
            if(hs.containsKey(test))
            {
                result.insert(hs.get(test), "(");
                result.append(")");
                break;
            }
            result.append(rest);
            hs.put(test, result.length() - 1);
            num = num % den;
        }
        return result.toString();
    }
}

/* instruction: binary search, return the first occurence of a number, -1 if not found */
class Solution {
    /**
     * @param nums: The integer array.
     * @param target: Target to find.
     * @return: The first position of target. Position starts from 0.
     */
    public int binarySearch(int[] nums, int target) {
        int len = nums.length;
        int first_occur = Integer.MIN_VALUE;
        if(len == 0)
            return -1;
        int start = 0;
        int end = len - 1;
        while(end >= start){
            int middle = (end + start) / 2;
            if(nums[middle] == target){
                first_occur = middle;
                end = middle - 1;
            }
            else if(nums[middle] < target){
                start = middle + 1;
            }
            else{
                end = middle - 1;
            }
        }
        if(first_occur == Integer.MIN_VALUE)
            return -1;
        return first_occur;
    }
}

/* instruction: A contains a substring as B's anagram */
public class Solution {
    /**
     * @param A : A string includes Upper Case letters
     * @param B : A string includes Upper Case letter
     * @return :  if string A contains all of the characters in B return true else return false
     */
    public boolean compareStrings(String A, String B) {
        // write your code here
        int lena = A.length();
        int lenb = B.length();
        if(lena == lenb &&  lena == 0)
            return true;
        if(lena == 0)
            return false;
        for(int i = 0; i < lenb; i ++){
            char b = B.charAt(i);
            String bs = Character.toString(b);
            A = A.replaceFirst(bs, "");
        }
        int count = 0;
        int newlen = A.length();
        if(newlen == lena - lenb)
            return true;
        return false;
    }
}

/* instruction :
You are climbing a stair case. It takes n steps to reach to the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
*/
//Time Out!!
public class Solution {
    /**
     * @param n: an integer
     * @return: an integer
     */
    public int climbStairs(int n) {
        if(n == 0)
            return 1;
        if(n == 1)
            return 1;
        return climbStairs(n - 1) + climbStairs(n - 2);

    }
}

/* instruction: Given a digit string, return all possible letter combinations that the number could represent. */

import java.util.LinkedList;
import java.util.List;
import java.util.HashMap;
public class Solution {
    HashMap<Integer, String> hm;
    public List<String> letterCombinations(String digits) {
        hm = new HashMap<Integer, String>();
        hm.put(0, "");
        hm.put(1, "");
        hm.put(2, "abc");
        hm.put(3, "def");
        hm.put(4, "ghi");
        hm.put(5, "jkl");
        hm.put(6, "mno");
        hm.put(7, "pqrs");
        hm.put(8, "tuv");
        hm.put(9, "wxyz");
        int len = digits.length();
        LinkedList<String> ls = new LinkedList<String>();
        ls.add("");
        if(len == 0)
            return ls;
        int[] numbers = new int[len];
        for(int i = 0; i < len; i ++)
            numbers[i] = Character.getNumericValue(digits.charAt(i));
        for (int i = 0; i < len; i++) {
            int num = numbers[i];
            int size = ls.size();
            for (int k = 0; k < size; k++) {
                String tmp = ls.pop();
                for (int j = 0; j < hm.get(num).length(); j++)
                    ls.add(tmp + hm.get(num).charAt(j));
            }
        }
        List<String> ret = new LinkedList<String>();
        ret.addAll(ls);
        return ret;
    }
}

/* instruction: You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security system connected and it will automatically contact the police if two adjacent houses were broken into on the same night.
 
 Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police. */
public class Solution {
    public int rob(int[] nums) {
        int length = nums.length;
        if(length <= 0)
            return 0;
        if(length == 1)
            return nums[0];
        if(length == 2)
            return Math.max(nums[0], nums[1]);
        int dp[] = new int[length];
        for(int i = 0; i < length; i ++)
            dp[i] = 0;
        dp[0] = nums[0];
        dp[1] = Math.max(nums[1], nums[0]);
        for(int i = 2; i < length; i ++)
        {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
        }
        return dp[length - 1];
        
    }
}

/* instruction: Determine if a Sudoku is valid */
import java.util.HashSet;
public class Solution {
    private int number = 9;
    public boolean isValidSudoku(char[][] board) {
        HashSet<Character> hs = new HashSet<Character>();
        int row, column;
        for(column = 0; column < number; column ++)
        {
            for(row = 0; row < number; row ++)
            {
                if(board[column][row] == '0')
                    return false;
                if(board[column][row] != '.')
                {
                    if(hs.contains(board[column][row]))
                        return false;
                    hs.add(board[column][row]);
                }
            }
            hs.clear();
        }
        for(row = 0; row < number; row ++)
        {
            for(column = 0; column < number;  column ++)
            {
                if(board[column][row] == '0')
                    return false;
                if(board[column][row] != '.')
                {
                    if(hs.contains(board[column][row]))
                        return false;
                    hs.add(board[column][row]);
                }
            }
            hs.clear();
        }
        int trial = 0;
        while(trial < 9)
        {
            for(column = 3 * (trial % 3); column < 3 * (trial % 3) + 3; column ++)
            {
                for(row = 3 * (trial / 3); row < 3 * (trial / 3) + 3; row ++)
                {
                    if(board[column][row] == '0')
                        return false;
                    if(board[column][row] != '.')
                    {
                        if(hs.contains(board[column][row]))
                            return false;
                        hs.add(board[column][row]);
                    }
                }
            }
            hs.clear();
            trial ++;
        }
        return true;
    }
}

/* instruction: Given an array of integers, find two numbers such that they add up to a specific target number.
 
 The function twoSum should return indices of the two numbers such that they add up to the target, where index1 must be less than index2. Please note that your returned answers (both index1 and index2) are not zero-based.
 
 You may assume that each input would have exactly one solution.
 */
import java.util.HashMap;
public class Solution {
    public int[] twoSum(int A[],int target) {
        int count[] = new int[2];
        HashMap<Integer, Integer> hm = new HashMap<Integer, Integer>();
        int len = A.length;
        for(int i = 0; i < len; i ++){
            if(hm.containsKey(A[i])){
                int index = hm.get(A[i]);
                count[0] = index + 1;
                count[1] = i + 1;
            }
            else{
                hm.put(target - A[i], i);
            }
        }
        return count;
    }
}

/* instruction: Given an array S of n integers, are there elements a, b, c, and d in S such that a + b + c + d = target? Find all unique quadruplets in the array which gives the sum of target. */
public class Solution {
    public List<List<Integer>> fourSum(int[] num, int target) {
        int length = num.length;
        List<List<Integer>> list = new LinkedList<List<Integer>>();
        if(length < 4)
            return list;
        HashSet<List<Integer>> map = new HashSet<List<Integer>>();
        Arrays.sort(num);
        for(int i = 0; i < length - 3; i ++)
        {
            if(i > 0 && num[i] == num[i - 1])
                continue;
            for(int e = length - 1; e >= i + 3; e --)
            {
                if(e < length - 1 && num[e] == num[e + 1])
                    continue;
                int local = target - num[i] - num[e];
                int start = i + 1;
                int end = e - 1;
                while(start < end)
                {
                    if(num[start] + num[end] > local)
                        end --;
                    else if(num[start] + num[end] < local)
                        start ++;
                    else
                    {
                        List<Integer> l = new LinkedList<Integer>();
                        l.add(num[i]);
                        l.add(num[start]);
                        l.add(num[end]);
                        l.add(num[e]);
                        if(!map.contains(l))
                        {
                            list.add(l);
                            map.add(l);
                        }
                        start ++;
                        end --;
                    }
                }
            }
        }
        return list;
        
    }
}

/* instruction: Given two integers n and k, return all possible combinations of k numbers out of 1 ... n. */
import java.util.LinkedList;
import java.util.List;
public class Solution {
    public List<List<Integer>> combine(int n, int k) {
        LinkedList<LinkedList<Integer>> lli = new LinkedList<LinkedList<Integer>>();
        LinkedList<Integer> li;
        if(n < 1)
            return null;
        for(int i = 1; i <= n; i ++){
            li = new LinkedList<Integer>();
            li.add(i);
            lli.add(li);
        }
        for(int i = 0; i < k - 1; i ++){
            int size = lli.size();
            for(int l = 0; l < size; l ++){
                li = lli.pop();
                int start = li.getLast();
                for(int j = start + 1; j <= n; j ++){
                    LinkedList<Integer> temp = new LinkedList<Integer>();
                    temp.addAll(li);
                    temp.add(j);
                    lli.add(temp);
                }
            }
        }
        for(int i = 0; i < lli.size(); i ++){
            if(lli.get(i).size() != k){
                lli.remove(i);
                i --;
            }
        }
        List<List<Integer>> ret= new LinkedList<List<Integer>>();
        ret.addAll(lli);
        return ret;
    }
}

/* instruction: Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
 
 For example, given n = 3, a solution set is:
 
 "((()))", "(()())", "(())()", "()(())", "()()()"
 */
import java.util.LinkedList;
import java.util.List;
import java.util.Stack;
public class Solution {
    List<String> ls = new LinkedList<String>();
    public List<String> generateParenthesis(int n) {
        String s = new String();
        generator(0, 0, s, n);
        return ls;
    }
    public void generator(int open, int close, String s, int n){
        if(open == n && close == n){
            if(check(s))
                ls.add(s);
        }
        if(open < n)
            generator(open + 1, close, s + "(", n);
        if(close < n)
            generator(open, close + 1, s + ")", n);
    }
    public boolean check(String s){
        Stack ns = new Stack();
        int len = s.length();
        for(int i = 0; i < len; i ++){
            if(s.charAt(i) == '(')
                ns.push('(');
            else{
                if(ns.empty())
                    return false;
                ns.pop();
            }
        }
        return true;
    }
}

/* instruction: Given a collection of numbers, return all possible permutations. */
public class Solution {
    List<List<Integer>> lli = new LinkedList<List<Integer>>();
    List<Integer> li = new LinkedList<Integer>();
    boolean used[];
    public List<List<Integer>> permute(int[] num) {
        if(num == null)
            return null;
        int len = num.length;
        used = new boolean[num.length];
        do_insert(num, 0, len);
        return lli;
    }
    public void do_insert(int []num, int index, int len){
        List<Integer> temp;
        if(index == len){
            temp = new LinkedList<Integer>();
            temp.addAll(li);
            lli.add(temp);
            return;
        }
        for(int i = 0; i < len; i ++){
            if(!used[i]){
                li.add(num[i]);
                used[i] = true;
                do_insert(num, index + 1, len);
                used[i] = false;
                li.remove(index);
            }
        }
        
    }
}

/* instruction:
 Given a sorted array of integers, find the starting and ending position of a given target value.
 
 Your algorithm's runtime complexity must be in the order of O(log n).
 
 If the target is not found in the array, return [-1, -1].
 */
public class Solution {
    public int[] searchRange(int[] A, int target) {
        int [] result = new int[]{-1, -1};
        int length = A.length;
        if(length == 0)
            return result;
        int start = 0, end = length - 1, maximum = 0, minimum = length - 1;
        boolean found = false;
        while(start <= end)
        {
            if(A[start] == target)
            {
                found = true;
                if(start >= maximum)
                    maximum = start;
                if(start <= minimum)
                    minimum = start;
            }
            if(A[end] == target)
            {
                found = true;
                if(end >= maximum)
                    maximum = end;
                if(end <= minimum)
                    minimum = end;
            }
            
            int mid = (start + end) / 2;
            if(A[mid] < target)
            {
                start ++;
            }
            else if (A[mid] > target)
            {
                end --;
            }
            else
            {
                if(mid >= maximum)
                    maximum = mid;
                if(mid <= minimum)
                    minimum = mid;
                start ++;
                found = true;
            }
        }
        if(found)
        {
            result[0] = minimum;
            result[1] = maximum;
        }
        return result;
    }
}

/* instruction: Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining. */
public class Solution {
    public int trap(int[] A) {
        int i = 0;
        int len = A.length;
        if(len == 0)
            return 0;
        if(len == 3){
            int water = Math.min(A[0] - A[1], A[2] - A[1]);
            return Math.max(water, 0);
        }
        int maxleft = A[i], maxright = A[len - 1];
        int water = 0;
        int left = 1, right = len - 2;
        while(left <= right){
            if(maxleft < maxright){
                water += Math.max(0, maxleft - A[left]);
                maxleft = Math.max(A[left], maxleft);
                left ++;
            }
            else{
                water += Math.max(0, maxright - A[right]);
                maxright = Math.max(A[right], maxright);
                right --;
            }
        }
        return water;
    }
}

/* instruction: You are given an n x n 2D matrix representing an image.
 
 Rotate the image by 90 degrees (clockwise).
 */
//O(n^2) space
public class Solution {
    public void rotate(int[][] matrix) {
        int len1 = matrix.length;
        if(len1 == 0)
            return;
        int temp[][] = new int[len1][];
        for(int i = 0; i < len1; i ++)
            temp[i] = new int[len1];
        for(int i = 0; i < len1; i ++)
            for(int j = 0; j < len1; j ++)
                temp[j][len1 - i - 1] = matrix[i][j];
        for(int i = 0; i < len1; i ++)
            for(int j = 0; j < len1; j ++)
                matrix[i][j] = temp[i][j];
    }
}
/* download from internet
 inplace swap
 
 public class Solution {
    public void rotate(int[][] matrix) {
        if(matrix==null) return ;
            int row=matrix.length;
        for(int i=0;i<row/2;i++){
            for(int j=i;j<row-i-1;j++){
                swap(matrix,i,j,row-1-j,i);  // an interesting thing here is that : j+(row-1-j)=row-1
                swap(matrix,row-1-j,i,row-1-i,row-1-j); // i+(row-1-i)=row-1
                swap(matrix,row-1-i,row-1-j,j,row-1-i); // row-1-j+j=row-1
            }
        }
    }
    public void swap(int[][] matrix,int i,int j,int p,int q){
        int tmp=matrix[i][j];
        matrix[i][j]=matrix[p][q];
        matrix[p][q]=tmp;
    }
 }
 */
/* another solution from cc 150 */
public void rotate(int [][] matrix, int n){
    for (int layer = 0; layer < n/2; layer ++){
        int first = layer;
        int last = n - 1 - layer;
        for(int i = first; i < last; i ++){
            int offset = i - first;
            //save top
            int top = matrix[first][i];
            //left -> top
            matrix[first][i] = matrix[last - offset][first];
            //bottom -> left
            matrix[last - offset][first] = matrix[last][last - offset];
            //right -> bottom
            matrix[last][last - offset] = matrix[i][last];
            //top -> right
            matrix[i][last] = top;
        }
    }
}

/* instruction: Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right which minimizes the sum of all numbers along its path. */
public class Solution {
    public int minPathSum(int[][] grid) {
        int lenrow = grid.length;
        if(lenrow == 0)
            return 0;
        int lencol = grid[0].length;
        for(int i = 1; i < lenrow; i ++)
            grid[i][0] += grid[i - 1][0];
        for(int j = 1; j < lencol; j ++)
            grid[0][j] += grid[0][j - 1];
        for(int i = 1; i < lenrow; i ++)
            for(int j = 1; j < lencol; j ++)
                grid[i][j] += Math.min(grid[i][j - 1], grid[i - 1][j]);
        return grid[lenrow - 1][lencol - 1];
    }
}

/* instruction: Given a m x n matrix, if an element is 0, set its entire row and column to 0. Do it in place. */
public class Solution {
    public void setZeroes(int[][] matrix){
        int lenrow = matrix.length;
        if(lenrow == 0)
            return;
        int lencol = matrix[0].length;
        boolean temprow[] = new boolean[lenrow];
        boolean tempcol[] = new boolean[lencol];
        for(int i = 0; i < lenrow; i ++)
            for(int j = 0; j < lencol; j ++)
                if(matrix[i][j] == 0){
                    temprow[i] = true;
                    tempcol[j] = true;
                }
        for(int i = 0; i < lenrow; i++)
            if(temprow[i] == true)
                zeroesRow(matrix, i, lencol);
        for(int i = 0; i < lencol; i ++)
            if(tempcol[i] == true)
                zeroesCol(matrix, i, lenrow);
    }
    public void zeroesRow(int [][]matrix, int row, int lencol){
        for(int i = 0; i < lencol; i ++)
            matrix[row][i] = 0;
    }
    public void zeroesCol(int [][]matrix, int col, int lenrow){
        for(int i = 0; i < lenrow; i ++)
            matrix[i][col] = 0;
    }
}

/* instruction: A peak element is an element that is greater than its neighbors.
 Given an input array where num[i] ≠ num[i+1], find a peak element and return its index.*/

public class Solution {
    public int findPeakElement(int[] num) {
        int length = num.length;
        if(length == 0 || length == 1)
            return 0;
        if(num[0] > num[1])
            return 0;
        if(num[length - 1] > num[length - 2])
            return length - 1;
        for(int i = 1; i < length - 1; i ++)
        {
            if(num[i] > num[i - 1] && num[i] > num[i + 1])
                return i;
        }
        return -1;
        
    }
}

/************************ cc 150 ********************************/

/* instruction: implement an algorithm determine if a string has all unique characters. Do this without using extra data structure */
public boolean unique(String s){
    if(s.length() > 128)
        return false;
    boolean check[] = new boolean[128];
    int len = s.length();
    for (int i = 0; i < len; i ++){
        if(check[s.charAt(i)] == true)
            return false;
        check[s.charAt(i)] = true;
    }
    return true;
}

/* instruction: check whther string a is a permutation of string b */
public boolean isPermutation(String a, String b){
    int lena = a.length();
    int lenb = b.length();
    if( lena != lenb)
        return false;
    for(int i = 0; i < lena; i ++){
        if(b.equals(""))
            return false;
        char c = a.charAt(i);
        String temp = Character.toString(c);
        b = b.replaceFirst(temp, "");
    }
    if(b.equals(""))
        return true;
    return false;
}

/* instruction: Given an unsorted integer array, find the first missing positive integer. */
public class Solution {
    public int firstMissingPositive(int[] A) {
        int number = 1;
        int length = A.length;
        Arrays.sort(A);
        for(int i = 0; i < length; i ++)
        {
            if(A[i] == number)
                number ++;
        }
        return number;
    }
}

/* instruction: replace the white space in an array with %20, assume the array has enough space and the length is the actual length */
public void replaceSpace(char []array, int length){
    int space = 0;
    for(int i = 0; i < length; i ++)
        if(array[i] == ' ')
            space ++;
    int newlen = length + space * 2;
    array[newlen] = '\0';
    newlen --;
    for(int i = length - 1; i >= 0; i --){
        if(array[i] == ' '){
            array[newlen ] = '0';
            array[newlen - 1] = '2';
            array[newlen - 2] = '%';
            newlen = newlen - 3;
        }
        else{
            array[newlen] = array[i];
            newlen --;
        }
    }
}

/* instruction: merge a series of non-overlapping intervals */
/**
 * Definition for an interval.
 * public class Interval {
 *     int start;
 *     int end;
 *     Interval() { start = 0; end = 0; }
 *     Interval(int s, int e) { start = s; end = e; }
 * }
 */
public class Solution {
    public List<Interval> insert(List<Interval> intervals, Interval newInterval) {
        int length = intervals.size();
        List<Interval> list = new LinkedList<Interval>();
        if(length == 0)
        {
            list.add(newInterval);
            return list;
        }
        /* find an interval for the start value in newInterval */
        int startPos = -1, endPos = -1;
        for(int i = 0; i < length; i ++)
        {
            if(intervals.get(i).start <= newInterval.start && intervals.get(i).end >= newInterval.start)
            {
                startPos = i;
            }
            if(intervals.get(i).start <= newInterval.end && intervals.get(i).end >= newInterval.end)
            {
                endPos = i;
                break;
            }
        }
        if(startPos == -1 && endPos == -1)
        {
            /* if the new interval doesn't overlap with any existing intervals */
            /* find the place it should lay */
            boolean added = false;
            for(int i = 0; i < length - 1; i ++)
            {
                if(intervals.get(i).end < newInterval.start && intervals.get(i + 1).start > newInterval.start)
                {
                    startPos = i;
                    break;
                }
            }
            if(newInterval.start > intervals.get(length - 1).end)
                startPos = length - 1;
            if(startPos == -1 && newInterval.start > intervals.get(0).start)
                startPos = 0;
            for(int i = 1; i < length; i ++)
            {
                if(intervals.get(i - 1).end < newInterval.end && intervals.get(i).start > newInterval.end)
                {
                    endPos = i - 1;
                }
            }
            if(newInterval.end < intervals.get(0).start)
                endPos = startPos;
            else if(endPos == -1 && newInterval.end < intervals.get(length - 1).end)
                endPos = length - 2;
            else if(endPos == -1)
                endPos = length - 1;
            for(int i = 0; i <= startPos; i ++)
            {
                list.add(intervals.get(i));
            }
            list.add(newInterval);
            for(int i = endPos + 1; i < length; i ++)
            {
                list.add(intervals.get(i));
            }
        }
        else if(startPos == -1 && endPos != -1)
        {
            int newEnding = Math.max(newInterval.end, intervals.get(endPos).end);
            for(int i = 0; i < length - 1; i ++)
            {
                if(intervals.get(i).end < newInterval.start && intervals.get(i + 1).start > newInterval.start)
                {
                    startPos = i;
                    break;
                }
            }
            if(startPos == -1 && newInterval.start > intervals.get(0).start)
                startPos = 0;
            for(int i = 0; i <= startPos; i ++)
            {
                list.add(intervals.get(i));
            }
            Interval tmp = new Interval(newInterval.start, newEnding);
            list.add(tmp);
            for(int i = endPos + 1; i < length; i ++)
            {
                list.add(intervals.get(i));
            }
        }
        else if(startPos != -1 && endPos == -1)
        {
            int newStart = Math.min(newInterval.start, intervals.get(startPos).start);
            for(int i = 1; i < length; i ++)
            {
                if(intervals.get(i - 1).end < newInterval.end && intervals.get(i).start > newInterval.end)
                {
                    endPos = i - 1;
                }
            }
            if(endPos == -1)
                endPos = length - 1;
            for(int i = 0; i < startPos; i ++)
            {
                list.add(intervals.get(i));
            }
            Interval tmp = new Interval(newStart, newInterval.end);
            list.add(tmp);
            for(int i = endPos + 1; i < length; i ++)
                list.add(intervals.get(i));
        }
        else
        {
            if(startPos != endPos)
            {
                if(detectMerge(newInterval, intervals.get(startPos), intervals.get(endPos)))
                {
                    intervals.get(startPos).end = intervals.get(endPos).end;
                    for(int i = 0; i <= startPos; i ++)
                        list.add(intervals.get(i));
                    for(int i = endPos + 1; i < length; i ++)
                        list.add(intervals.get(i));
                }
                else
                {
                    int newStart = Math.min(newInterval.start, intervals.get(startPos).start);
                    int newEnding = Math.max(newInterval.end, intervals.get(endPos).end);
                    for(int i = 0; i <= startPos; i ++)
                        list.add(intervals.get(i));
                    list.add(new Interval(newStart, newEnding));
                    for(int i = endPos + 1; i < length; i ++)
                        list.add(intervals.get(i));
                }
            }
            else
                list.addAll(intervals);
        }
        return list;
    }
    public boolean detectMerge(Interval toAdd, Interval a, Interval b)
    {
        if(toAdd.start <= a.end && toAdd.end >= b.start)
            return true;
        return false;
    }
}

/* instruction: Given an unsorted array of integers, find the length of the longest consecutive elements sequence. */
public class Solution {
    public int longestConsecutive(int[] num) {
        int length = num.length;
        if(length == 0)
            return 0;
        Arrays.sort(num);
        int longest = 1, consecutive = 1;
        for(int i = 0; i < length - 1; i ++)
        {
            if(num[i] + 1 == num[i + 1])
                consecutive ++;
            else if(num[i] == num[i + 1])
                continue;
            else
            {
                longest = Math.max(consecutive, longest);
                consecutive = 1;
            }
            
        }
        longest = Math.max(consecutive, longest);
        return longest;
    }
}


//Given an array of integers that is already sorted in ascending order, find two numbers such that they add up to a specific target number.
//
//The function twoSum should return indices of the two numbers such that they add up to the target, where index1 must be less than index2. Please note that your returned answers (both index1 and index2) are not zero-based.
public class Solution {
    public int[] twoSum(int[] numbers, int target) {
        HashMap<Integer, Integer> hm = new HashMap<Integer, Integer>();
        int right = numbers.length - 1;
        int left = 0;
        int []result = new int[2];
        while (left < right) {
            if (numbers[left] + numbers[right] == target) {
                result[0] = left + 1;
                result[1] = right + 1;
                break;
            }
            else if (numbers[left] + numbers[right] < target) {
                left += 1;
            }
            else {
                right -= 1;
            }
        }
        return result;
    }
}

/* instruction: Assume you have a method isSubstring which checks if one word is a substring of another. 
Given two strings, s1 and s2, write code to check if s2 is a rotation of s1 using only one call to is Substring */

public boolean isRotation(String s1, String s2){
    int len = s1.length();
    int len2 = s2.length();
    if(len == le2 && len > 0){
        String s1s2 = s1 + s2;
        return isSubstring(s1s2, s2);
    }
    return false;
}

/* instruction: remove duplicates from a linked list */
import java.util.HashSet;
public void removeDup(ListNode head){
    HashSet<Integer> hs = new HashSet<Integer>();
    ListNode pointer = head;
    ListNode pre = null;
    while(pointer != null){
        if(hs.contains(pointer.val)){
            pre.next = pointer.next;
        }
        else{
            hs.add(pointer.next.val);
            pre = pointer;
        }
        pointer = pointer.next;            
    }
}

/* algorithm: find the kth last element in a singly linked list */
public int kLast(ListNode node, int x){
    if(node == null)
        return -1; //error
    int index = kLast(node.next, x) + 1;
    if(index == x)
        System.out.println(index);
    return index;
        
}
public int kLast(ListNode node, int x){
    if(node == null || x < 0)
        return -1;
    ListNode pointer = node;
    ListNode faster = node;
    for(int i = 0 ; i < x - 1 ; i ++)
        faster = faster.next;
    if(faster == null)
        return -1;
    while(faster != null){
        faster = faster.next;
        pointer = pointer.next;
    }
    return pointer.val;
}

/* instruction: implement a function to check if the list is a palindrome */
public boolean isPalindrome(ListNode head)
{
    if(head == null)
        return true;
    Stack<Integer> stack = new Stack<Integer>();
    ListNode slow = head, fast = head;
    while(fast != null && fast.next != null)
    {
        stack.push(slow.val);
        slow = slow.next;
        fast = fast.next.next;
    }
    if(fast != null)
        slow = slow.next;
    while(slow != null)
    {
        if(slow.val != stack.pop())
            return false;
        slow = slow.next;
    }
    return true;
}

/* design three stacks using one array */
public class Stacks {
    int array[];
    final int stackSize = 100;
    int firstStack, secondStack, thirdStack;
    Stacks()
    {
        array = new int[stackSize * 3];
        firstStack = 0;
        secondStack = stackSize;
        thirdStack = stackSize * 2;
    }
    public int pop(int num)
    {
        switch(num)
        {
            case 1:
                if(firstStack == 0)
                    return -1;
                else
                {
                    int value = array[firstStack --];
                    return value;
                }
            case 2:
                if(secondStack == stackSize)
                    return -1;
                else
                {
                    int value = array[secondStack --];
                    return value;
                }
            case 3:
                if(thirdStack == 2 * stackSize)
                    return -1;
                else
                {
                    int value = array[thirdStack --];
                    return value;
                }
            default:
                return -1;
        }
    }
    public void push(int value, int dest)
    {
        switch(dest) {
            case 1:
                if(firstStack >= stackSize)
                    throw new StackOverflowError("the stack is full");
                array[firstStack++] = value;
                break;
            case 2:
                if(secondStack >= stackSize * 2)
                    throw new StackOverflowError("the stack is full");
                array[secondStack++] = value;
                break;
            case 3:
                if(thirdStack > stackSize * 3)
                    throw new StackOverflowError("the stack is full");
                array[thirdStack++] = value;
                break;
        }
    }
    public boolean isEmpty(int num)
    {
        switch (num)
        {
            case 1:
                return firstStack == 0 ? true:false;
            case 2:
                return secondStack == stackSize ? true : false;
            case 3:
                return thirdStack == stackSize * 2 ? true : false;
            default:
                return false;
        }
    }
    public int peak(int num)
    {
        if(isEmpty(num))
            return -1;
        switch (num)
        {
            case 1:
                return array[firstStack];
            case 2:
                return array[secondStack];
            case 3:
                return array[thirdStack];
            default:
                return -1;
        }
    }
}

/* instruction: implement a queue with two stacks */
public class MyQueue<T> {
    Stack<T> stack1;
    Stack<T> stack2;
    MyQueue()
    {
        stack1 = new Stack<T>();
        stack2 = new Stack<T>();
    }
    public boolean isEmpty()
    {
        return stack1.isEmpty() && stack2.isEmpty();
    }
    public int size()
    {
        return stack1.size() + stack2.size();
    }
    public void push(T val)
    {
        stack2.push(val);
    }
    public T remove()
    {
        if(stack1.isEmpty())
        {
            while(!stack2.isEmpty())
            {
                T val = stack2.pop();
                stack1.push(val);
            }
        }
        return stack1.pop();
    }
    public T peek()
    {
        if(stack1.isEmpty())
        {
            while(!stack2.isEmpty())
            {
                T val = stack2.pop();
                stack1.push(val);
            }
        }
        return stack1.peek();
    }
}

/* instruction: sort a stack in ascending order */
public class Stacks {
    Stack<Integer> stack1;
    Stack<Integer> helper;
    Stacks()
    {
        stack1 = new Stack<Integer>();
        helper = new Stack<Integer>();
    }
    public void push(int val)
    {
        if(stack1.isEmpty())
        {
            stack1.push(val);
            return;
        }
        while(stack1.peek() > val)
        {
            int temp = stack1.pop();
            helper.push(temp);
        }
        stack1.push(val);
        while(!helper.isEmpty())
        {
            stack1.push(helper.pop());
        }
    }
    public int pop()
    {
        return stack1.pop();
    }
    public int peek()
    {
        return stack1.peek();
    }
    public boolean isEmpty()
    {
        return stack1.isEmpty();
    }
}

/* create a linkedlist of all nodes at each depth */
public void createLinkedList(TreeNode root, List<LinkedList<TreeNode>> list, int level)
{
    if(root == null)
        return;
    LinkedList<TreeNode> l;
    if(list.size() == level)
    {
        /* if the list's size is same as level, create a new list */
        l = new LinkedList<TreeNode>();
        list.add(l);
    }
    else
    {
        l = list.get(level);
    }
    l.add(root);
    createLinkedList(root.left, list, level + 1);
    createLinkedList(root.right, list, level + 1);

}
List<LinkedList<TreeNode>> createLevelLinkedList(TreeNode root)
{
    List<LinkedList<TreeNode>> list = new LinkedList<LinkedList<TreeNode>>();
    createLinkedList(root, list, 0);
    return list;
}

/* do it in BFS, both use O(N) time and O(N) space */
List<LinkedList<TreeNode>> createLevelLinkedList(TreeNode root)
{
    List<LinkedList<TreeNode>> list = new LinkedList<LinkedList<TreeNode>>();
    LinkedList<TreeNode> current = new LinkedList<TreeNode>();
    current.add(root);
    while(current.size() != 0)
    {
        list.add(current);
        LinkedList<TreeNode> parent = current;
        current = new LinkedList<TreeNode>();
        for(TreeNode p: parent)
        {
            if(p.left != null)
                current.add(p.left);
            if(p.right != null)
                current.add(p.right);
        }
    }
    return list;
}

/* find the inorder successor of a node n */
public TreeNode inorderSucc(TreeNode n)
{
    if(n == null)
        return null;
    if(n.right != null)
        return findLeftMostChild(n.right);
    TreeNode parent = n.parent;
    while(parent != null && parent.left != n) {
        n = parent;
        parent = parent.parent;
    }
    return n;

}
public TreeNode findLeftMostChild(TreeNode n)
{
    if(n == null)
        return null;
    while(n.left != null)
        n = n.left;
    return n;
}

/* instruction: find the common ancestor of two nodes */
public boolean isOnSameSide(TreeNode root, TreeNode a)
{
    if(root == null)
        return false;
    if(root == a)
        return true;
    return isOnSameSide(root.left, a) || isOnSameSide(root.right, a);
}
public TreeNode findCommonAncestor(TreeNode a, TreeNode b, TreeNode root)
{
    if(a == null || b == null)
        return null;
    if(a == root || b == root)
        return root;
    boolean isALeft = isOnSameSide(root.left, a);
    boolean isBLeft = isOnSameSide(root.left, b);

    if(isALeft != isBLeft)
        return root;
    else if(isALeft)
        return findCommonAncestor(a, b, root.left);
    else
        return findCommonAncestor(a, b, root.right);

}


/* find path sum, note: it doesn't need to start at root and end at leave */
public int findTreeDepth(TreeNode n)
{
    if(n == null)
        return 0;
    return Math.max(findTreeDepth(n.left), findTreeDepth(n.right)) + 1;
}
public void pathSum(TreeNode n, int sum)
{
    int depth = findTreeDepth(n);
    int array[] = new int[depth];
    findSum(n, sum, array, 0);
}
public void findSum(TreeNode n, int sum, int[] array, int index)
{
    if(n == null)
        return;
    array[index] = n.val;
    int curr = 0;
    for(int i = index; i >= 0; i --)
    {
        curr += array[i];
        if(curr == sum)
        {
            for(int j = index; j >=  i; j --)
                System.out.print(array[j] + " ");
            System.out.println("");
        }
    }
    findSum(n.left, sum, array, index + 1);
    findSum(n.right, sum, array, index + 1);

}

// 3 sum closest
public class Solution {
    public int threeSumClosest(int[] nums, int target) {
        int length = nums.length;
        if (length < 3)
            return 0;
        int closest = nums[0] + nums[1] + nums[2];
        Arrays.sort(nums);
        for(int i = 0; i < nums.length; i ++) {
            int first = nums[i];
            int start = i + 1;
            int end = nums.length - 1;
            while (start < end) {
                int s = first + nums[start] + nums[end];
                if (s == target)
                    return target;
                if (Math.abs(s - target) < Math.abs(closest - target))
                {
                    closest = s;
                }
                if (s > target)
                    end --;
                if (s < target)
                    start ++;
            }
        }
        return closest;
    }
}
/*********************** cc 150 ends ***************************/


// Amazon OA
public String longestPalindrome(String s) {
    int n = s.length();
    if(n == 0)
        return "";
    String longest = s.substring(0, 1);
    for(int i = 0; i < n - 1; i ++) {
    String p1 = expandAroundCenter(s, i, i, n);
    if (p1.length() > longest.length())
        longest = p1;
    String p2 = expandAroundCenter(s, i, i + 1, n);
    if (p2.length() > longest.length())
        longest = p2;
    }
    return longest;
}

public String expandAroundCenter(String s, int a, int b, int length) {
    int left = a, right = b;
    while (left >= 0 && right < length && s.charAt(left) == s.charAt(right)) {
        left --;
        right ++;
    }
    return s.substring(left + 1, right);
}



public boolean isValidParenthesis(String s) {
    int length = s.length();
    Stack<Character> stack = new Stack<Character>();
    for(int i = 0; i < length; i ++) {
        if(stack.empty())
            stack.push(s.charAt(i));
        else if(s.charAt(i) - stack.peek() == 1 || s.charAt(i) - stack.peek() == 2)
            stack.pop();
        else
            stack.push(s.charAt(i));
    }
    return stack.empty();
}

public int twoSumPair(int []nums, int pair) {
    int length = nums.length;
    if(nums == null || length < 2)
        return 0;
    HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
    int count = 0;
    for(int i = 0; i < length; i ++) {
        int rest = pair - nums[i];
        if (map.containsKey(nums[i]))
            count += map.get(nums[i]);
        else if(!map.containsKey(rest))
            map.put(rest, 1);
        else
        map.put(rest, map.get(rest) + 1);
    }
    return count;

}


public boolean isSubTree(TreeNode a, TreeNode b) {
    if (a == null)
        return false;
    if (b == null)
        return true;

    return isSameTree(a, b) || isSubTree(a.left, b) || isSubTree(a.right, b);

}

public boolean isSameTree(TreeNode a, TreeNode b) {
    if (a == null && b == null)
        return true;
    if (a != null && b == null)
        return false;
    if (a == null && b != null)
        return false;
    if (a != b)
        return false;
    return isSameTree(a.left, b.left) && isSameTree(a.right, b.right);
}


public ListNode reverse(ListNode a) {
    if (a == null || a.next == null)
        return a;
    ListNode slow = a;
    ListNode fast = a;
    while(fast.next != null && fast.next.next != null) {
        fast = fast.next.next;
        slow = slow.next;
    }
    ListNode pre = slow.next;
    ListNode current = pre.next;
    while(current != null) {
        pre.next = current.next;
        current.next = slow.next;
        slow.next = current;
        pre = current.next;
    }
    return a;
}


public ListNode merge(ListNode a, ListNode b) {
    ListNode head = new ListNode(-1);
    ListNode current = head;
    while (a != null && b != null) {
        if (a.value < b.value) {
            current.next = a;
            a = a.next;
        }
        else {
            current.next = b;
            b = b.next;
        }
        current = current.next;
    }
    if (a != null)
        current.next = a;
    if (b != null)
        current.next = b;
    return head.next;
}
