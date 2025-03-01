# data_structures.py

"""
This file contains implementations of various data structures and algorithms.
"""

# ------------------------- 2.1 Arrays -------------------------

class Arrays:
    @staticmethod
    def traverse(arr):
        """Traverse an array."""
        for i in arr:
            print(i)

    @staticmethod
    def insert(arr, index, value):
        """Insert an element into an array at a specific index."""
        arr.insert(index, value)
        return arr

    @staticmethod
    def delete(arr, index):
        """Delete an element from an array at a specific index."""
        arr.pop(index)
        return arr

    @staticmethod
    def prefix_sum(arr):
        """Calculate the prefix sum of an array."""
        for i in range(1, len(arr)):
            arr[i] += arr[i - 1]
        return arr

    @staticmethod
    def suffix_sum(arr):
        """Calculate the suffix sum of an array."""
        for i in range(len(arr) - 2, -1, -1):
            arr[i] += arr[i + 1]
        return arr

    @staticmethod
    def two_pointers(arr, target):
        """Two pointers technique to find pairs in a sorted array."""
        left, right = 0, len(arr) - 1
        while left < right:
            if arr[left] + arr[right] == target:
                return (left, right)
            elif arr[left] + arr[right] < target:
                left += 1
            else:
                right -= 1
        return None

    @staticmethod
    def sliding_window(arr, k):
        """Sliding window technique to find maximum sum of subarrays of size k."""
        max_sum = sum(arr[:k])
        window_sum = max_sum
        for i in range(k, len(arr)):
            window_sum += arr[i] - arr[i - k]
            max_sum = max(max_sum, window_sum)
        return max_sum

    @staticmethod
    def kadane(arr):
        """Kadane's algorithm to find maximum subarray sum."""
        max_current = max_global = arr[0]
        for i in range(1, len(arr)):
            max_current = max(arr[i], max_current + arr[i])
            if max_current > max_global:
                max_global = max_current
        return max_global

    @staticmethod
    def bubble_sort(arr):
        """Bubble sort implementation."""
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr

    @staticmethod
    def selection_sort(arr):
        """Selection sort implementation."""
        for i in range(len(arr)):
            min_idx = i
            for j in range(i + 1, len(arr)):
                if arr[j] < arr[min_idx]:
                    min_idx = j
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
        return arr

    @staticmethod
    def insertion_sort(arr):
        """Insertion sort implementation."""
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0 and key < arr[j]:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
        return arr

    @staticmethod
    def merge_sort(arr):
        """Merge sort implementation."""
        if len(arr) > 1:
            mid = len(arr) // 2
            left = arr[:mid]
            right = arr[mid:]
            Arrays.merge_sort(left)
            Arrays.merge_sort(right)
            i = j = k = 0
            while i < len(left) and j < len(right):
                if left[i] < right[j]:
                    arr[k] = left[i]
                    i += 1
                else:
                    arr[k] = right[j]
                    j += 1
                k += 1
            while i < len(left):
                arr[k] = left[i]
                i += 1
                k += 1
            while j < len(right):
                arr[k] = right[j]
                j += 1
                k += 1
        return arr

    @staticmethod
    def quick_sort(arr):
        """Quick sort implementation."""
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return Arrays.quick_sort(left) + middle + Arrays.quick_sort(right)

    @staticmethod
    def counting_sort(arr):
        """Counting sort implementation."""
        max_val = max(arr)
        count = [0] * (max_val + 1)
        for num in arr:
            count[num] += 1
        sorted_arr = []
        for i in range(len(count)):
            sorted_arr.extend([i] * count[i])
        return sorted_arr

    @staticmethod
    def radix_sort(arr):
        """Radix sort implementation."""
        max_val = max(arr)
        exp = 1
        while max_val // exp > 0:
            Arrays.counting_sort(arr)
            exp *= 10
        return arr

    @staticmethod
    def linear_search(arr, target):
        """Linear search implementation."""
        for i in range(len(arr)):
            if arr[i] == target:
                return i
        return -1

    @staticmethod
    def binary_search(arr, target):
        """Binary search implementation."""
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1

    @staticmethod
    def dutch_national_flag(arr):
        """Dutch National Flag algorithm for 3-way partitioning."""
        low, mid, high = 0, 0, len(arr) - 1
        while mid <= high:
            if arr[mid] == 0:
                arr[low], arr[mid] = arr[mid], arr[low]
                low += 1
                mid += 1
            elif arr[mid] == 1:
                mid += 1
            else:
                arr[mid], arr[high] = arr[high], arr[mid]
                high -= 1
        return arr

    @staticmethod
    def rotate_array(arr, k):
        """Rotate an array using the reversal algorithm."""
        def reverse(arr, start, end):
            while start < end:
                arr[start], arr[end] = arr[end], arr[start]
                start += 1
                end -= 1
        n = len(arr)
        k %= n
        reverse(arr, 0, n - 1)
        reverse(arr, 0, k - 1)
        reverse(arr, k, n - 1)
        return arr

    @staticmethod
    def majority_element(arr):
        """Moore's Voting Algorithm to find majority element."""
        candidate, count = None, 0
        for num in arr:
            if count == 0:
                candidate = num
            count += 1 if num == candidate else -1
        return candidate

    @staticmethod
    def merge_intervals(intervals):
        """Merge overlapping intervals."""
        intervals.sort(key=lambda x: x[0])
        merged = []
        for interval in intervals:
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                merged[-1][1] = max(merged[-1][1], interval[1])
        return merged


# ------------------------- 2.2 Strings -------------------------

class Strings:
    @staticmethod
    def reverse_string(s):
        """Reverse a string."""
        return s[::-1]

    @staticmethod
    def is_palindrome(s):
        """Check if a string is a palindrome."""
        return s == s[::-1]

    @staticmethod
    def substrings(s):
        """Generate all substrings of a string."""
        n = len(s)
        return [s[i:j] for i in range(n) for j in range(i + 1, n + 1)]

    @staticmethod
    def kmp(s, pattern):
        """KMP algorithm for pattern matching."""
        def compute_lps(pattern):
            lps = [0] * len(pattern)
            length = 0
            i = 1
            while i < len(pattern):
                if pattern[i] == pattern[length]:
                    length += 1
                    lps[i] = length
                    i += 1
                else:
                    if length != 0:
                        length = lps[length - 1]
                    else:
                        lps[i] = 0
                        i += 1
            return lps

        lps = compute_lps(pattern)
        i, j = 0, 0
        while i < len(s):
            if pattern[j] == s[i]:
                i += 1
                j += 1
            if j == len(pattern):
                return i - j
            elif i < len(s) and pattern[j] != s[i]:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1
        return -1

    @staticmethod
    def rabin_karp(s, pattern):
        """Rabin-Karp algorithm for pattern matching."""
        d = 256
        q = 101
        n, m = len(s), len(pattern)
        h = pow(d, m - 1, q)
        p = 0
        t = 0
        for i in range(m):
            p = (d * p + ord(pattern[i])) % q
            t = (d * t + ord(s[i])) % q
        for i in range(n - m + 1):
            if p == t:
                if s[i:i + m] == pattern:
                    return i
            if i < n - m:
                t = (d * (t - ord(s[i]) * h) + ord(s[i + m])) % q
        return -1

    @staticmethod
    def z_algorithm(s):
        """Z algorithm for pattern matching."""
        n = len(s)
        z = [0] * n
        l, r = 0, 0
        for i in range(1, n):
            if i > r:
                l = r = i
                while r < n and s[r - l] == s[r]:
                    r += 1
                z[i] = r - l
                r -= 1
            else:
                k = i - l
                if z[k] < r - i + 1:
                    z[i] = z[k]
                else:
                    l = i
                    while r < n and s[r - l] == s[r]:
                        r += 1
                    z[i] = r - l
                    r -= 1
        return z

    @staticmethod
    def rolling_hash(s, base=256, mod=101):
        """Rolling hash function."""
        hash_value = 0
        for char in s:
            hash_value = (hash_value * base + ord(char)) % mod
        return hash_value

    @staticmethod
    def lcs(s1, s2):
        """Longest Common Subsequence (LCS) using dynamic programming."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    @staticmethod
    def longest_palindromic_substring(s):
        """Manacher's algorithm to find the longest palindromic substring."""
        t = '#'.join(f'^{s}$')
        n = len(t)
        p = [0] * n
        c, r = 0, 0
        for i in range(1, n - 1):
            mirror = 2 * c - i
            if i < r:
                p[i] = min(r - i, p[mirror])
            while t[i + p[i] + 1] == t[i - p[i] - 1]:
                p[i] += 1
            if i + p[i] > r:
                c, r = i, i + p[i]
        max_len, center = max((n, i) for i, n in enumerate(p))
        return s[(center - max_len) // 2:(center + max_len) // 2]

    @staticmethod
    def are_anagrams(s1, s2):
        """Check if two strings are anagrams."""
        return sorted(s1) == sorted(s2)

    @staticmethod
    def run_length_encoding(s):
        """Run-length encoding for string compression."""
        encoded = []
        count = 1
        for i in range(1, len(s)):
            if s[i] == s[i - 1]:
                count += 1
            else:
                encoded.append(f"{s[i - 1]}{count}")
                count = 1
        encoded.append(f"{s[-1]}{count}")
        return ''.join(encoded)


# ------------------------- 2.3 Linked List -------------------------

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        """Append a node to the end of the linked list."""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        last = self.head
        while last.next:
            last = last.next
        last.next = new_node

    def prepend(self, data):
        """Prepend a node to the beginning of the linked list."""
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node

    def delete(self, key):
        """Delete a node with the given key."""
        temp = self.head
        if temp and temp.data == key:
            self.head = temp.next
            temp = None
            return
        prev = None
        while temp and temp.data != key:
            prev = temp
            temp = temp.next
        if not temp:
            return
        prev.next = temp.next
        temp = None

    def reverse(self):
        """Reverse the linked list."""
        prev = None
        current = self.head
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        self.head = prev

    def detect_cycle(self):
        """Detect a cycle in the linked list using Floyd's algorithm."""
        slow = fast = self.head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False

    def remove_nth_from_end(self, n):
        """Remove the nth node from the end of the linked list."""
        fast = slow = self.head
        for _ in range(n):
            fast = fast.next
        if not fast:
            return self.head.next
        while fast.next:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return self.head

    def merge_sorted_lists(self, l2):
        """Merge two sorted linked lists."""
        dummy = Node(0)
        tail = dummy
        l1 = self.head
        while l1 and l2:
            if l1.data <= l2.data:
                tail.next = l1
                l1 = l1.next
            else:
                tail.next = l2
                l2 = l2.next
            tail = tail.next
        if l1:
            tail.next = l1
        if l2:
            tail.next = l2
        return dummy.next

    def find_intersection(self, l2):
        """Find the intersection point of two linked lists."""
        len1, len2 = 0, 0
        temp1, temp2 = self.head, l2.head
        while temp1:
            len1 += 1
            temp1 = temp1.next
        while temp2:
            len2 += 1
            temp2 = temp2.next
        temp1, temp2 = self.head, l2.head
        if len1 > len2:
            for _ in range(len1 - len2):
                temp1 = temp1.next
        else:
            for _ in range(len2 - len1):
                temp2 = temp2.next
        while temp1 and temp2:
            if temp1 == temp2:
                return temp1
            temp1 = temp1.next
            temp2 = temp2.next
        return None

    def clone_with_random_pointers(self):
        """Clone a linked list with random pointers."""
        if not self.head:
            return None
        mapping = {}
        current = self.head
        while current:
            mapping[current] = Node(current.data)
            current = current.next
        current = self.head
        while current:
            mapping[current].next = mapping.get(current.next)
            mapping[current].random = mapping.get(current.random)
            current = current.next
        return mapping[self.head]


# ------------------------- 2.4 Stack and Queue -------------------------

class Stack:
    def __init__(self):
        self.stack = []

    def push(self, item):
        """Push an item onto the stack."""
        self.stack.append(item)

    def pop(self):
        """Pop an item from the stack."""
        if not self.is_empty():
            return self.stack.pop()
        return None

    def top(self):
        """Get the top item of the stack."""
        if not self.is_empty():
            return self.stack[-1]
        return None

    def is_empty(self):
        """Check if the stack is empty."""
        return len(self.stack) == 0

    def min_stack(self):
        """Get the minimum element in the stack."""
        return min(self.stack) if self.stack else None

class Queue:
    def __init__(self):
        self.queue = []

    def enqueue(self, item):
        """Enqueue an item into the queue."""
        self.queue.append(item)

    def dequeue(self):
        """Dequeue an item from the queue."""
        if not self.is_empty():
            return self.queue.pop(0)
        return None

    def is_empty(self):
        """Check if the queue is empty."""
        return len(self.queue) == 0

    def circular_queue(self, size):
        """Implement a circular queue."""
        self.queue = [None] * size
        self.front = self.rear = -1

    def next_greater_element(self, arr):
        """Find the next greater element for each element in the array."""
        stack = []
        result = [-1] * len(arr)
        for i in range(len(arr)):
            while stack and arr[stack[-1]] < arr[i]:
                result[stack.pop()] = arr[i]
            stack.append(i)
        return result

    def balanced_parentheses(self, s):
        """Check if the parentheses are balanced."""
        stack = []
        mapping = {')': '(', '}': '{', ']': '['}
        for char in s:
            if char in mapping:
                top = stack.pop() if stack else '#'
                if mapping[char] != top:
                    return False
            else:
                stack.append(char)
        return not stack

    def lru_cache(self, capacity):
        """Implement an LRU cache."""
        from collections import OrderedDict
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        """Get an item from the LRU cache."""
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        """Put an item into the LRU cache."""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def sliding_window_maximum(self, arr, k):
        """Find the maximum in each sliding window of size k."""
        from collections import deque
        q = deque()
        result = []
        for i in range(len(arr)):
            while q and arr[q[-1]] <= arr[i]:
                q.pop()
            q.append(i)
            if q[0] == i - k:
                q.popleft()
            if i >= k - 1:
                result.append(arr[q[0]])
        return result


# ------------------------- 2.5 Hashing -------------------------

class Hashing:
    @staticmethod
    def hash_table():
        """Create a hash table."""
        return {}

    @staticmethod
    def open_addressing():
        """Open addressing for collision resolution."""
        pass

    @staticmethod
    def separate_chaining():
        """Separate chaining for collision resolution."""
        pass

    @staticmethod
    def count_distinct_elements(arr):
        """Count distinct elements in an array."""
        return len(set(arr))

    @staticmethod
    def longest_consecutive_subsequence(arr):
        """Find the longest consecutive subsequence."""
        num_set = set(arr)
        longest = 0
        for num in num_set:
            if num - 1 not in num_set:
                current_num = num
                current_streak = 1
                while current_num + 1 in num_set:
                    current_num += 1
                    current_streak += 1
                longest = max(longest, current_streak)
        return longest

    @staticmethod
    def two_sum(arr, target):
        """Find two numbers that add up to the target."""
        num_map = {}
        for i, num in enumerate(arr):
            complement = target - num
            if complement in num_map:
                return (num_map[complement], i)
            num_map[num] = i
        return None

    @staticmethod
    def group_anagrams(strs):
        """Group anagrams together."""
        from collections import defaultdict
        anagrams = defaultdict(list)
        for s in strs:
            key = ''.join(sorted(s))
            anagrams[key].append(s)
        return list(anagrams.values())


# ------------------------- 2.6 Binary Trees & Binary Search Trees -------------------------

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BinaryTree:
    def inorder_traversal(self, root):
        """Inorder traversal of a binary tree."""
        result = []
        def traverse(node):
            if node:
                traverse(node.left)
                result.append(node.val)
                traverse(node.right)
        traverse(root)
        return result

    def preorder_traversal(self, root):
        """Preorder traversal of a binary tree."""
        result = []
        def traverse(node):
            if node:
                result.append(node.val)
                traverse(node.left)
                traverse(node.right)
        traverse(root)
        return result

    def postorder_traversal(self, root):
        """Postorder traversal of a binary tree."""
        result = []
        def traverse(node):
            if node:
                traverse(node.left)
                traverse(node.right)
                result.append(node.val)
        traverse(root)
        return result

    def level_order_traversal(self, root):
        """Level order traversal of a binary tree."""
        from collections import deque
        result = []
        if not root:
            return result
        queue = deque([root])
        while queue:
            level_size = len(queue)
            current_level = []
            for _ in range(level_size):
                node = queue.popleft()
                current_level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            result.append(current_level)
        return result

    def height(self, root):
        """Calculate the height of a binary tree."""
        if not root:
            return 0
        return 1 + max(self.height(root.left), self.height(root.right))

    def diameter(self, root):
        """Calculate the diameter of a binary tree."""
        self.diameter = 0
        def height(node):
            if not node:
                return 0
            left = height(node.left)
            right = height(node.right)
            self.diameter = max(self.diameter, left + right)
            return 1 + max(left, right)
        height(root)
        return self.diameter

    def lowest_common_ancestor(self, root, p, q):
        """Find the lowest common ancestor of two nodes in a binary tree."""
        if not root or root == p or root == q:
            return root
        left = self.lowest_common_ancestor(root.left, p, q)
        right = self.lowest_common_ancestor(root.right, p, q)
        if left and right:
            return root
        return left if left else right

    def is_balanced(self, root):
        """Check if a binary tree is balanced."""
        def check(node):
            if not node:
                return 0
            left = check(node.left)
            right = check(node.right)
            if left == -1 or right == -1 or abs(left - right) > 1:
                return -1
            return 1 + max(left, right)
        return check(root) != -1

    def is_identical(self, root1, root2):
        """Check if two binary trees are identical."""
        if not root1 and not root2:
            return True
        if not root1 or not root2:
            return False
        return (root1.val == root2.val and
                self.is_identical(root1.left, root2.left) and
                self.is_identical(root1.right, root2.right))

    def kth_smallest(self, root, k):
        """Find the kth smallest element in a BST."""
        stack = []
        while True:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            k -= 1
            if k == 0:
                return root.val
            root = root.right

    def sorted_array_to_bst(self, arr):
        """Convert a sorted array to a balanced BST."""
        if not arr:
            return None
        mid = len(arr) // 2
        root = TreeNode(arr[mid])
        root.left = self.sorted_array_to_bst(arr[:mid])
        root.right = self.sorted_array_to_bst(arr[mid + 1:])
        return root


# ------------------------- 2.7 Heaps & Priority Queues -------------------------

import heapq

class Heaps:
    @staticmethod
    def min_heap(arr):
        """Create a min heap."""
        heapq.heapify(arr)
        return arr

    @staticmethod
    def max_heap(arr):
        """Create a max heap."""
        heapq._heapify_max(arr)
        return arr

    @staticmethod
    def heap_sort(arr):
        """Heap sort implementation."""
        heapq.heapify(arr)
        return [heapq.heappop(arr) for _ in range(len(arr))]

    @staticmethod
    def kth_largest(arr, k):
        """Find the kth largest element in an array."""
        return heapq.nlargest(k, arr)[-1]

    @staticmethod
    def kth_smallest(arr, k):
        """Find the kth smallest element in an array."""
        return heapq.nsmallest(k, arr)[-1]

    @staticmethod
    def merge_k_sorted_lists(lists):
        """Merge k sorted lists."""
        import heapq
        heap = []
        for i, lst in enumerate(lists):
            if lst:
                heapq.heappush(heap, (lst[0], i, 0))
        result = []
        while heap:
            val, list_idx, element_idx = heapq.heappop(heap)
            result.append(val)
            if element_idx + 1 < len(lists[list_idx]):
                heapq.heappush(heap, (lists[list_idx][element_idx + 1], list_idx, element_idx + 1))
        return result

    @staticmethod
    def median_of_running_stream():
        """Find the median of a running stream of numbers."""
        import heapq
        min_heap = []
        max_heap = []
        def add_num(num):
            heapq.heappush(max_heap, -heapq.heappushpop(min_heap, num))
            if len(max_heap) > len(min_heap):
                heapq.heappush(min_heap, -heapq.heappop(max_heap))
        def find_median():
            if len(min_heap) == len(max_heap):
                return (min_heap[0] - max_heap[0]) / 2
            return min_heap[0]
        return add_num, find_median

    @staticmethod
    def top_k_frequent_elements(arr, k):
        """Find the top k frequent elements in an array."""
        from collections import Counter
        count = Counter(arr)
        return heapq.nlargest(k, count.keys(), key=count.get)


# ------------------------- 2.8 Graphs -------------------------

class Graph:
    def __init__(self, vertices):
        self.vertices = vertices
        self.adj_list = {v: [] for v in range(vertices)}

    def add_edge(self, u, v):
        """Add an edge to the graph."""
        self.adj_list[u].append(v)
        self.adj_list[v].append(u)

    def bfs(self, start):
        """Breadth-First Search (BFS) for graph traversal."""
        visited = set()
        queue = [start]
        result = []
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                result.append(node)
                queue.extend(self.adj_list[node])
        return result

    def dfs(self, start):
        """Depth-First Search (DFS) for graph traversal."""
        visited = set()
        result = []
        def traverse(node):
            if node not in visited:
                visited.add(node)
                result.append(node)
                for neighbor in self.adj_list[node]:
                    traverse(neighbor)
        traverse(start)
        return result

    def detect_cycle_undirected(self):
        """Detect a cycle in an undirected graph."""
        visited = set()
        def dfs(node, parent):
            visited.add(node)
            for neighbor in self.adj_list[node]:
                if neighbor not in visited:
                    if dfs(neighbor, node):
                        return True
                elif neighbor != parent:
                    return True
            return False
        for node in range(self.vertices):
            if node not in visited:
                if dfs(node, -1):
                    return True
        return False

    def detect_cycle_directed(self):
        """Detect a cycle in a directed graph."""
        visited = set()
        rec_stack = set()
        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            for neighbor in self.adj_list[node]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(node)
            return False
        for node in range(self.vertices):
            if node not in visited:
                if dfs(node):
                    return True
        return False

    def topological_sort(self):
        """Topological sorting using Kahn's algorithm."""
        in_degree = [0] * self.vertices
        for u in range(self.vertices):
            for v in self.adj_list[u]:
                in_degree[v] += 1
        queue = [u for u in range(self.vertices) if in_degree[u] == 0]
        result = []
        while queue:
            u = queue.pop(0)
            result.append(u)
            for v in self.adj_list[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        return result if len(result) == self.vertices else []

    def dijkstra(self, start):
        """Dijkstra's algorithm for shortest path."""
        import heapq
        distances = [float('inf')] * self.vertices
        distances[start] = 0
        heap = [(0, start)]
        while heap:
            current_distance, u = heapq.heappop(heap)
            if current_distance > distances[u]:
                continue
            for v in self.adj_list[u]:
                distance = current_distance + 1  # Assuming unweighted graph
                if distance < distances[v]:
                    distances[v] = distance
                    heapq.heappush(heap, (distance, v))
        return distances

    def bellman_ford(self, start):
        """Bellman-Ford algorithm for shortest path."""
        distances = [float('inf')] * self.vertices
        distances[start] = 0
        for _ in range(self.vertices - 1):
            for u in range(self.vertices):
                for v in self.adj_list[u]:
                    if distances[u] + 1 < distances[v]:  # Assuming unweighted graph
                        distances[v] = distances[u] + 1
        return distances

    def floyd_warshall(self):
        """Floyd-Warshall algorithm for shortest path."""
        distances = [[float('inf')] * self.vertices for _ in range(self.vertices)]
        for u in range(self.vertices):
            distances[u][u] = 0
            for v in self.adj_list[u]:
                distances[u][v] = 1  # Assuming unweighted graph
        for k in range(self.vertices):
            for i in range(self.vertices):
                for j in range(self.vertices):
                    distances[i][j] = min(distances[i][j], distances[i][k] + distances[k][j])
        return distances

    def kruskal(self):
        """Kruskal's algorithm for minimum spanning tree."""
        parent = [i for i in range(self.vertices)]
        def find(u):
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]
        def union(u, v):
            u_root = find(u)
            v_root = find(v)
            if u_root != v_root:
                parent[v_root] = u_root
        edges = []
        for u in range(self.vertices):
            for v in self.adj_list[u]:
                edges.append((1, u, v))  # Assuming unweighted graph
        edges.sort()
        mst = []
        for weight, u, v in edges:
            if find(u) != find(v):
                union(u, v)
                mst.append((u, v))
        return mst

    def prim(self):
        """Prim's algorithm for minimum spanning tree."""
        import heapq
        visited = set()
        heap = [(0, 0)]  # (weight, node)
        mst = []
        while heap:
            weight, u = heapq.heappop(heap)
            if u not in visited:
                visited.add(u)
                mst.append((u, weight))
                for v in self.adj_list[u]:
                    if v not in visited:
                        heapq.heappush(heap, (1, v))  # Assuming unweighted graph
        return mst

    def kosaraju(self):
        """Kosaraju's algorithm for strongly connected components."""
        visited = set()
        order = []
        def dfs(u):
            visited.add(u)
            for v in self.adj_list[u]:
                if v not in visited:
                    dfs(v)
            order.append(u)
        for u in range(self.vertices):
            if u not in visited:
                dfs(u)
        reversed_graph = {v: [] for v in range(self.vertices)}
        for u in range(self.vertices):
            for v in self.adj_list[u]:
                reversed_graph[v].append(u)
        visited = set()
        sccs = []
        def dfs_reversed(u, component):
            visited.add(u)
            component.append(u)
            for v in reversed_graph[u]:
                if v not in visited:
                    dfs_reversed(v, component)
        for u in reversed(order):
            if u not in visited:
                component = []
                dfs_reversed(u, component)
                sccs.append(component)
        return sccs

    def find_bridges(self):
        """Find bridges in a graph."""
        visited = set()
        low = [float('inf')] * self.vertices
        disc = [float('inf')] * self.vertices
        parent = [-1] * self.vertices
        bridges = []
        time = 0
        def dfs(u):
            nonlocal time
            visited.add(u)
            disc[u] = low[u] = time
            time += 1
            for v in self.adj_list[u]:
                if v not in visited:
                    parent[v] = u
                    dfs(v)
                    low[u] = min(low[u], low[v])
                    if low[v] > disc[u]:
                        bridges.append((u, v))
                elif v != parent[u]:
                    low[u] = min(low[u], disc[v])
        for u in range(self.vertices):
            if u not in visited:
                dfs(u)
        return bridges

    def articulation_points(self):
        """Find articulation points in a graph."""
        visited = set()
        low = [float('inf')] * self.vertices
        disc = [float('inf')] * self.vertices
        parent = [-1] * self.vertices
        ap = set()
        time = 0
        def dfs(u):
            nonlocal time
            visited.add(u)
            disc[u] = low[u] = time
            time += 1
            children = 0
            for v in self.adj_list[u]:
                if v not in visited:
                    parent[v] = u
                    children += 1
                    dfs(v)
                    low[u] = min(low[u], low[v])
                    if parent[u] == -1 and children > 1:
                        ap.add(u)
                    if parent[u] != -1 and low[v] >= disc[u]:
                        ap.add(u)
                elif v != parent[u]:
                    low[u] = min(low[u], disc[v])
        for u in range(self.vertices):
            if u not in visited:
                dfs(u)
        return ap

    def ford_fulkerson(self, source, sink):
        """Ford-Fulkerson algorithm for maximum flow."""
        def bfs(u, v, parent):
            visited = set()
            queue = [u]
            visited.add(u)
            while queue:
                u = queue.pop(0)
                for v in self.adj_list[u]:
                    if v not in visited and residual[u][v] > 0:
                        queue.append(v)
                        visited.add(v)
                        parent[v] = u
                        if v == sink:
                            return True
            return False
        residual = [[0] * self.vertices for _ in range(self.vertices)]
        for u in range(self.vertices):
            for v in self.adj_list[u]:
                residual[u][v] = 1  # Assuming unweighted graph
        max_flow = 0
        parent = [-1] * self.vertices
        while bfs(source, sink, parent):
            path_flow = float('inf')
            s = sink
            while s != source:
                path_flow = min(path_flow, residual[parent[s]][s])
                s = parent[s]
            max_flow += path_flow
            v = sink
            while v != source:
                u = parent[v]
                residual[u][v] -= path_flow
                residual[v][u] += path_flow
                v = parent[v]
        return max_flow


# ------------------------- Print All Functions -------------------------

class DataStructureExamples:
    """
    A class to provide examples and usage instructions for each data structure and algorithm.
    """

    @staticmethod
    def arrays_examples():
        """
        Returns examples and usage instructions for the Arrays class.
        """
        return """
        ## Arrays Class Examples

        - traverse(arr): Traverse an array.
          Example:
            Arrays.traverse([1, 2, 3])  # Output: 1 2 3

        - insert(arr, index, value): Insert an element into an array at a specific index.
          Example:
            Arrays.insert([1, 2, 3], 1, 5)  # Output: [1, 5, 2, 3]

        - delete(arr, index): Delete an element from an array at a specific index.
          Example:
            Arrays.delete([1, 2, 3], 1)  # Output: [1, 3]

        - prefix_sum(arr): Calculate the prefix sum of an array.
          Example:
            Arrays.prefix_sum([1, 2, 3, 4])  # Output: [1, 3, 6, 10]

        - suffix_sum(arr): Calculate the suffix sum of an array.
          Example:
            Arrays.suffix_sum([1, 2, 3, 4])  # Output: [10, 9, 7, 4]

        - two_pointers(arr, target): Two pointers technique to find pairs in a sorted array.
          Example:
            Arrays.two_pointers([1, 2, 3, 4], 5)  # Output: (1, 2)

        - sliding_window(arr, k): Sliding window technique to find maximum sum of subarrays of size k.
          Example:
            Arrays.sliding_window([1, 2, 3, 4, 5], 3)  # Output: 12

        - kadane(arr): Kadane's algorithm to find maximum subarray sum.
          Example:
            Arrays.kadane([-2, 1, -3, 4, -1, 2, 1, -5, 4])  # Output: 6

        - bubble_sort(arr): Bubble sort implementation.
          Example:
            Arrays.bubble_sort([3, 1, 4, 2])  # Output: [1, 2, 3, 4]

        - selection_sort(arr): Selection sort implementation.
          Example:
            Arrays.selection_sort([3, 1, 4, 2])  # Output: [1, 2, 3, 4]

        - insertion_sort(arr): Insertion sort implementation.
          Example:
            Arrays.insertion_sort([3, 1, 4, 2])  # Output: [1, 2, 3, 4]

        - merge_sort(arr): Merge sort implementation.
          Example:
            Arrays.merge_sort([3, 1, 4, 2])  # Output: [1, 2, 3, 4]

        - quick_sort(arr): Quick sort implementation.
          Example:
            Arrays.quick_sort([3, 1, 4, 2])  # Output: [1, 2, 3, 4]

        - counting_sort(arr): Counting sort implementation.
          Example:
            Arrays.counting_sort([3, 1, 4, 2])  # Output: [1, 2, 3, 4]

        - radix_sort(arr): Radix sort implementation.
          Example:
            Arrays.radix_sort([3, 1, 4, 2])  # Output: [1, 2, 3, 4]

        - linear_search(arr, target): Linear search implementation.
          Example:
            Arrays.linear_search([1, 2, 3, 4], 3)  # Output: 2

        - binary_search(arr, target): Binary search implementation.
          Example:
            Arrays.binary_search([1, 2, 3, 4], 3)  # Output: 2

        - dutch_national_flag(arr): Dutch National Flag algorithm for 3-way partitioning.
          Example:
            Arrays.dutch_national_flag([2, 0, 2, 1, 1, 0])  # Output: [0, 0, 1, 1, 2, 2]

        - rotate_array(arr, k): Rotate an array using the reversal algorithm.
          Example:
            Arrays.rotate_array([1, 2, 3, 4, 5], 2)  # Output: [4, 5, 1, 2, 3]

        - majority_element(arr): Moore's Voting Algorithm to find majority element.
          Example:
            Arrays.majority_element([3, 3, 4, 2, 4, 4, 2, 4, 4])  # Output: 4

        - merge_intervals(intervals): Merge overlapping intervals.
          Example:
            Arrays.merge_intervals([[1, 3], [2, 6], [8, 10]])  # Output: [[1, 6], [8, 10]]
        """

    @staticmethod
    def strings_examples():
        """
        Returns examples and usage instructions for the Strings class.
        """
        return """
        ## Strings Class Examples

        - reverse_string(s): Reverse a string.
          Example:
            Strings.reverse_string("hello")  # Output: "olleh"

        - is_palindrome(s): Check if a string is a palindrome.
          Example:
            Strings.is_palindrome("madam")  # Output: True

        - substrings(s): Generate all substrings of a string.
          Example:
            Strings.substrings("abc")  # Output: ['a', 'ab', 'abc', 'b', 'bc', 'c']

        - kmp(s, pattern): KMP algorithm for pattern matching.
          Example:
            Strings.kmp("hello", "ll")  # Output: 2

        - rabin_karp(s, pattern): Rabin-Karp algorithm for pattern matching.
          Example:
            Strings.rabin_karp("hello", "ll")  # Output: 2

        - z_algorithm(s): Z algorithm for pattern matching.
          Example:
            Strings.z_algorithm("hello")  # Output: [0, 0, 0, 0, 0]

        - rolling_hash(s, base=256, mod=101): Rolling hash function.
          Example:
            Strings.rolling_hash("hello")  # Output: 52

        - lcs(s1, s2): Longest Common Subsequence (LCS) using dynamic programming.
          Example:
            Strings.lcs("abcde", "ace")  # Output: 3

        - longest_palindromic_substring(s): Manacher's algorithm to find the longest palindromic substring.
          Example:
            Strings.longest_palindromic_substring("babad")  # Output: "bab"

        - are_anagrams(s1, s2): Check if two strings are anagrams.
          Example:
            Strings.are_anagrams("listen", "silent")  # Output: True

        - run_length_encoding(s): Run-length encoding for string compression.
          Example:
            Strings.run_length_encoding("aaabbbcc")  # Output: "a3b3c2"
        """

    @staticmethod
    def linked_list_examples():
        """
        Returns examples and usage instructions for the LinkedList class.
        """
        return """
        ## LinkedList Class Examples

        - append(data): Append a node to the end of the linked list.
          Example:
            ll = LinkedList()
            ll.append(1)
            ll.append(2)
            # Linked List: 1 -> 2

        - prepend(data): Prepend a node to the beginning of the linked list.
          Example:
            ll = LinkedList()
            ll.prepend(1)
            ll.prepend(2)
            # Linked List: 2 -> 1

        - delete(key): Delete a node with the given key.
          Example:
            ll = LinkedList()
            ll.append(1)
            ll.append(2)
            ll.delete(1)
            # Linked List: 2

        - reverse(): Reverse the linked list.
          Example:
            ll = LinkedList()
            ll.append(1)
            ll.append(2)
            ll.reverse()
            # Linked List: 2 -> 1

        - detect_cycle(): Detect a cycle in the linked list using Floyd's algorithm.
          Example:
            ll = LinkedList()
            ll.append(1)
            ll.append(2)
            ll.head.next.next = ll.head  # Creates a cycle
            ll.detect_cycle()  # Output: True

        - remove_nth_from_end(n): Remove the nth node from the end of the linked list.
          Example:
            ll = LinkedList()
            ll.append(1)
            ll.append(2)
            ll.append(3)
            ll.remove_nth_from_end(2)
            # Linked List: 1 -> 3

        - merge_sorted_lists(l2): Merge two sorted linked lists.
          Example:
            ll1 = LinkedList()
            ll1.append(1)
            ll1.append(3)
            ll2 = LinkedList()
            ll2.append(2)
            ll2.append(4)
            ll1.merge_sorted_lists(ll2)
            # Linked List: 1 -> 2 -> 3 -> 4

        - find_intersection(l2): Find the intersection point of two linked lists.
          Example:
            ll1 = LinkedList()
            ll1.append(1)
            ll1.append(2)
            ll2 = LinkedList()
            ll2.append(3)
            ll2.head.next = ll1.head.next  # Creates intersection
            ll1.find_intersection(ll2)  # Output: Node with value 2

        - clone_with_random_pointers(): Clone a linked list with random pointers.
          Example:
            ll = LinkedList()
            ll.append(1)
            ll.append(2)
            cloned_ll = ll.clone_with_random_pointers()
            # Cloned Linked List: 1 -> 2
        """

    @staticmethod
    def stack_examples():
        """
        Returns examples and usage instructions for the Stack class.
        """
        return """
        ## Stack Class Examples

        - push(item): Push an item onto the stack.
          Example:
            stack = Stack()
            stack.push(1)
            stack.push(2)
            # Stack: [1, 2]

        - pop(): Pop an item from the stack.
          Example:
            stack = Stack()
            stack.push(1)
            stack.push(2)
            stack.pop()  # Output: 2

        - top(): Get the top item of the stack.
          Example:
            stack = Stack()
            stack.push(1)
            stack.push(2)
            stack.top()  # Output: 2

        - is_empty(): Check if the stack is empty.
          Example:
            stack = Stack()
            stack.is_empty()  # Output: True

        - min_stack(): Get the minimum element in the stack.
          Example:
            stack = Stack()
            stack.push(2)
            stack.push(1)
            stack.min_stack()  # Output: 1
        """

    @staticmethod
    def queue_examples():
        """
        Returns examples and usage instructions for the Queue class.
        """
        return """
        ## Queue Class Examples

        - enqueue(item): Enqueue an item into the queue.
          Example:
            queue = Queue()
            queue.enqueue(1)
            queue.enqueue(2)
            # Queue: [1, 2]

        - dequeue(): Dequeue an item from the queue.
          Example:
            queue = Queue()
            queue.enqueue(1)
            queue.enqueue(2)
            queue.dequeue()  # Output: 1

        - is_empty(): Check if the queue is empty.
          Example:
            queue = Queue()
            queue.is_empty()  # Output: True

        - circular_queue(size): Implement a circular queue.
          Example:
            queue = Queue()
            queue.circular_queue(5)
            queue.enqueue(1)
            queue.enqueue(2)
            # Circular Queue: [1, 2, None, None, None]

        - next_greater_element(arr): Find the next greater element for each element in the array.
          Example:
            queue = Queue()
            queue.next_greater_element([4, 5, 2, 25])  # Output: [5, 25, 25, -1]

        - balanced_parentheses(s): Check if the parentheses are balanced.
          Example:
            queue = Queue()
            queue.balanced_parentheses("()[]{}")  # Output: True

        - lru_cache(capacity): Implement an LRU cache.
          Example:
            cache = Queue()
            cache.lru_cache(2)
            cache.put(1, 1)
            cache.put(2, 2)
            cache.get(1)  # Output: 1

        - sliding_window_maximum(arr, k): Find the maximum in each sliding window of size k.
          Example:
            queue = Queue()
            queue.sliding_window_maximum([1, 3, -1, -3, 5, 3, 6, 7], 3)  # Output: [3, 3, 5, 5, 6, 7]
        """

    @staticmethod
    def hashing_examples():
        """
        Returns examples and usage instructions for the Hashing class.
        """
        return """
        ## Hashing Class Examples

        - hash_table(): Create a hash table.
          Example:
            hash_table = Hashing.hash_table()
            hash_table["key"] = "value"
            # Hash Table: {"key": "value"}

        - open_addressing(): Open addressing for collision resolution.
          Example:
            Hashing.open_addressing()

        - separate_chaining(): Separate chaining for collision resolution.
          Example:
            Hashing.separate_chaining()

        - count_distinct_elements(arr): Count distinct elements in an array.
          Example:
            Hashing.count_distinct_elements([1, 2, 2, 3])  # Output: 3

        - longest_consecutive_subsequence(arr): Find the longest consecutive subsequence.
          Example:
            Hashing.longest_consecutive_subsequence([100, 4, 200, 1, 3, 2])  # Output: 4

        - two_sum(arr, target): Find two numbers that add up to the target.
          Example:
            Hashing.two_sum([2, 7, 11, 15], 9)  # Output: (0, 1)

        - group_anagrams(strs): Group anagrams together.
          Example:
            Hashing.group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
            # Output: [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]
        """

    @staticmethod
    def binary_tree_examples():
        """
        Returns examples and usage instructions for the BinaryTree class.
        """
        return """
        ## BinaryTree Class Examples

        - inorder_traversal(root): Inorder traversal of a binary tree.
          Example:
            root = TreeNode(1)
            root.left = TreeNode(2)
            root.right = TreeNode(3)
            BinaryTree().inorder_traversal(root)  # Output: [2, 1, 3]

        - preorder_traversal(root): Preorder traversal of a binary tree.
          Example:
            root = TreeNode(1)
            root.left = TreeNode(2)
            root.right = TreeNode(3)
            BinaryTree().preorder_traversal(root)  # Output: [1, 2, 3]

        - postorder_traversal(root): Postorder traversal of a binary tree.
          Example:
            root = TreeNode(1)
            root.left = TreeNode(2)
            root.right = TreeNode(3)
            BinaryTree().postorder_traversal(root)  # Output: [2, 3, 1]

        - level_order_traversal(root): Level order traversal of a binary tree.
          Example:
            root = TreeNode(1)
            root.left = TreeNode(2)
            root.right = TreeNode(3)
            BinaryTree().level_order_traversal(root)  # Output: [[1], [2, 3]]

        - height(root): Calculate the height of a binary tree.
          Example:
            root = TreeNode(1)
            root.left = TreeNode(2)
            root.right = TreeNode(3)
            BinaryTree().height(root)  # Output: 2

        - diameter(root): Calculate the diameter of a binary tree.
          Example:
            root = TreeNode(1)
            root.left = TreeNode(2)
            root.right = TreeNode(3)
            BinaryTree().diameter(root)  # Output: 2

        - lowest_common_ancestor(root, p, q): Find the lowest common ancestor of two nodes in a binary tree.
          Example:
            root = TreeNode(1)
            root.left = TreeNode(2)
            root.right = TreeNode(3)
            BinaryTree().lowest_common_ancestor(root, root.left, root.right)  # Output: TreeNode(1)

        - is_balanced(root): Check if a binary tree is balanced.
          Example:
            root = TreeNode(1)
            root.left = TreeNode(2)
            root.right = TreeNode(3)
            BinaryTree().is_balanced(root)  # Output: True

        - is_identical(root1, root2): Check if two binary trees are identical.
          Example:
            root1 = TreeNode(1)
            root1.left = TreeNode(2)
            root1.right = TreeNode(3)
            root2 = TreeNode(1)
            root2.left = TreeNode(2)
            root2.right = TreeNode(3)
            BinaryTree().is_identical(root1, root2)  # Output: True

        - kth_smallest(root, k): Find the kth smallest element in a BST.
          Example:
            root = TreeNode(3)
            root.left = TreeNode(1)
            root.right = TreeNode(4)
            BinaryTree().kth_smallest(root, 1)  # Output: 1

        - sorted_array_to_bst(arr): Convert a sorted array to a balanced BST.
          Example:
            BinaryTree().sorted_array_to_bst([1, 2, 3])  # Output: TreeNode(2) with left=1 and right=3
        """

    @staticmethod
    def heaps_examples():
        """
        Returns examples and usage instructions for the Heaps class.
        """
        return """
        ## Heaps Class Examples

        - min_heap(arr): Create a min heap.
          Example:
            Heaps.min_heap([3, 1, 4, 2])  # Output: [1, 2, 4, 3]

        - max_heap(arr): Create a max heap.
          Example:
            Heaps.max_heap([3, 1, 4, 2])  # Output: [4, 3, 2, 1]

        - heap_sort(arr): Heap sort implementation.
          Example:
            Heaps.heap_sort([3, 1, 4, 2])  # Output: [1, 2, 3, 4]

        - kth_largest(arr, k): Find the kth largest element in an array.
          Example:
            Heaps.kth_largest([3, 1, 4, 2], 2)  # Output: 3

        - kth_smallest(arr, k): Find the kth smallest element in an array.
          Example:
            Heaps.kth_smallest([3, 1, 4, 2], 2)  # Output: 2

        - merge_k_sorted_lists(lists): Merge k sorted lists.
          Example:
            Heaps.merge_k_sorted_lists([[1, 4, 5], [1, 3, 4], [2, 6]])  # Output: [1, 1, 2, 3, 4, 4, 5, 6]

        - median_of_running_stream(): Find the median of a running stream of numbers.
          Example:
            add_num, find_median = Heaps.median_of_running_stream()
            add_num(1)
            add_num(2)
            find_median()  # Output: 1.5

        - top_k_frequent_elements(arr, k): Find the top k frequent elements in an array.
          Example:
            Heaps.top_k_frequent_elements([1, 1, 1, 2, 2, 3], 2)  # Output: [1, 2]
        """

    @staticmethod
    def graph_examples():
        """
        Returns examples and usage instructions for the Graph class.
        """
        return """
        ## Graph Class Examples

        - add_edge(u, v): Add an edge to the graph.
          Example:
            graph = Graph(4)
            graph.add_edge(0, 1)
            graph.add_edge(1, 2)
            # Graph: {0: [1], 1: [0, 2], 2: [1]}

        - bfs(start): Breadth-First Search (BFS) for graph traversal.
          Example:
            graph = Graph(4)
            graph.add_edge(0, 1)
            graph.add_edge(1, 2)
            graph.bfs(0)  # Output: [0, 1, 2]

        - dfs(start): Depth-First Search (DFS) for graph traversal.
          Example:
            graph = Graph(4)
            graph.add_edge(0, 1)
            graph.add_edge(1, 2)
            graph.dfs(0)  # Output: [0, 1, 2]

        - detect_cycle_undirected(): Detect a cycle in an undirected graph.
          Example:
            graph = Graph(4)
            graph.add_edge(0, 1)
            graph.add_edge(1, 2)
            graph.add_edge(2, 0)
            graph.detect_cycle_undirected()  # Output: True

        - detect_cycle_directed(): Detect a cycle in a directed graph.
          Example:
            graph = Graph(4)
            graph.add_edge(0, 1)
            graph.add_edge(1, 2)
            graph.add_edge(2, 0)
            graph.detect_cycle_directed()  # Output: True

        - topological_sort(): Topological sorting using Kahn's algorithm.
          Example:
            graph = Graph(4)
            graph.add_edge(0, 1)
            graph.add_edge(1, 2)
            graph.topological_sort()  # Output: [0, 1, 2]

        - dijkstra(start): Dijkstra's algorithm for shortest path.
          Example:
            graph = Graph(4)
            graph.add_edge(0, 1)
            graph.add_edge(1, 2)
            graph.dijkstra(0)  # Output: [0, 1, 2, inf]

        - bellman_ford(start): Bellman-Ford algorithm for shortest path.
          Example:
            graph = Graph(4)
            graph.add_edge(0, 1)
            graph.add_edge(1, 2)
            graph.bellman_ford(0)  # Output: [0, 1, 2, inf]

        - floyd_warshall(): Floyd-Warshall algorithm for shortest path.
          Example:
            graph = Graph(4)
            graph.add_edge(0, 1)
            graph.add_edge(1, 2)
            graph.floyd_warshall()  # Output: [[0, 1, 2, inf], [inf, 0, 1, inf], [inf, inf, 0, inf], [inf, inf, inf, 0]]

        - kruskal(): Kruskal's algorithm for minimum spanning tree.
          Example:
            graph = Graph(4)
            graph.add_edge(0, 1)
            graph.add_edge(1, 2)
            graph.kruskal()  # Output: [(0, 1), (1, 2)]

        - prim(): Prim's algorithm for minimum spanning tree.
          Example:
            graph = Graph(4)
            graph.add_edge(0, 1)
            graph.add_edge(1, 2)
            graph.prim()  # Output: [(0, 1), (1, 2)]

        - kosaraju(): Kosaraju's algorithm for strongly connected components.
          Example:
            graph = Graph(4)
            graph.add_edge(0, 1)
            graph.add_edge(1, 2)
            graph.add_edge(2, 0)
            graph.kosaraju()  # Output: [[0, 1, 2]]

        - find_bridges(): Find bridges in a graph.
          Example:
            graph = Graph(4)
            graph.add_edge(0, 1)
            graph.add_edge(1, 2)
            graph.find_bridges()  # Output: [(1, 2)]

        - articulation_points(): Find articulation points in a graph.
          Example:
            graph = Graph(4)
            graph.add_edge(0, 1)
            graph.add_edge(1, 2)
            graph.articulation_points()  # Output: [1]

        - ford_fulkerson(source, sink): Ford-Fulkerson algorithm for maximum flow.
          Example:
            graph = Graph(4)
            graph.add_edge(0, 1)
            graph.add_edge(1, 2)
            graph.ford_fulkerson(0, 2)  # Output: 1
        """
    
    @staticmethod
    def printStructureOnly():
        """
        Returns a string containing the structure of classes and functions in the `data_structures.py` file.
        """
        return """
        # Structure of Classes and Functions in data_structures.py

        ## 1. Arrays Class
        - traverse(arr)
        - insert(arr, index, value)
        - delete(arr, index)
        - prefix_sum(arr)
        - suffix_sum(arr)
        - two_pointers(arr, target)
        - sliding_window(arr, k)
        - kadane(arr)
        - bubble_sort(arr)
        - selection_sort(arr)
        - insertion_sort(arr)
        - merge_sort(arr)
        - quick_sort(arr)
        - counting_sort(arr)
        - radix_sort(arr)
        - linear_search(arr, target)
        - binary_search(arr, target)
        - dutch_national_flag(arr)
        - rotate_array(arr, k)
        - majority_element(arr)
        - merge_intervals(intervals)

        ## 2. Strings Class
        - reverse_string(s)
        - is_palindrome(s)
        - substrings(s)
        - kmp(s, pattern)
        - rabin_karp(s, pattern)
        - z_algorithm(s)
        - rolling_hash(s, base=256, mod=101)
        - lcs(s1, s2)
        - longest_palindromic_substring(s)
        - are_anagrams(s1, s2)
        - run_length_encoding(s)

        ## 3. LinkedList Class
        - append(data)
        - prepend(data)
        - delete(key)
        - reverse()
        - detect_cycle()
        - remove_nth_from_end(n)
        - merge_sorted_lists(l2)
        - find_intersection(l2)
        - clone_with_random_pointers()

        ## 4. Stack Class
        - push(item)
        - pop()
        - top()
        - is_empty()
        - min_stack()

        ## 5. Queue Class
        - enqueue(item)
        - dequeue()
        - is_empty()
        - circular_queue(size)
        - next_greater_element(arr)
        - balanced_parentheses(s)
        - lru_cache(capacity)
        - sliding_window_maximum(arr, k)

        ## 6. Hashing Class
        - hash_table()
        - open_addressing()
        - separate_chaining()
        - count_distinct_elements(arr)
        - longest_consecutive_subsequence(arr)
        - two_sum(arr, target)
        - group_anagrams(strs)

        ## 7. BinaryTree Class
        - inorder_traversal(root)
        - preorder_traversal(root)
        - postorder_traversal(root)
        - level_order_traversal(root)
        - height(root)
        - diameter(root)
        - lowest_common_ancestor(root, p, q)
        - is_balanced(root)
        - is_identical(root1, root2)
        - kth_smallest(root, k)
        - sorted_array_to_bst(arr)

        ## 8. Heaps Class
        - min_heap(arr)
        - max_heap(arr)
        - heap_sort(arr)
        - kth_largest(arr, k)
        - kth_smallest(arr, k)
        - merge_k_sorted_lists(lists)
        - median_of_running_stream()
        - top_k_frequent_elements(arr, k)

        ## 9. Graph Class
        - add_edge(u, v)
        - bfs(start)
        - dfs(start)
        - detect_cycle_undirected()
        - detect_cycle_directed()
        - topological_sort()
        - dijkstra(start)
        - bellman_ford(start)
        - floyd_warshall()
        - kruskal()
        - prim()
        - kosaraju()
        - find_bridges()
        - articulation_points()
        - ford_fulkerson(source, sink)
        """

def printStructure():
        """
        Returns a string in triple quotes containing the `DataStructureExamples` class
        and all its function names.
        """
        return """
        # DataStructureExamples Class and Functions

        ## Class: DataStructureExamples

        ### Functions:
        - arrays_examples()
        - strings_examples()
        - linked_list_examples()
        - stack_examples()
        - queue_examples()
        - hashing_examples()
        - binary_tree_examples()
        - heaps_examples()
        - graph_examples()
        - printStructureOnly()
        """
       
def printCode():
    return '''
    # data_structures.py

"""
This file contains implementations of various data structures and algorithms.
"""

# ------------------------- 2.1 Arrays -------------------------

class Arrays:
    @staticmethod
    def traverse(arr):
        """Traverse an array."""
        for i in arr:
            print(i)

    @staticmethod
    def insert(arr, index, value):
        """Insert an element into an array at a specific index."""
        arr.insert(index, value)
        return arr

    @staticmethod
    def delete(arr, index):
        """Delete an element from an array at a specific index."""
        arr.pop(index)
        return arr

    @staticmethod
    def prefix_sum(arr):
        """Calculate the prefix sum of an array."""
        for i in range(1, len(arr)):
            arr[i] += arr[i - 1]
        return arr

    @staticmethod
    def suffix_sum(arr):
        """Calculate the suffix sum of an array."""
        for i in range(len(arr) - 2, -1, -1):
            arr[i] += arr[i + 1]
        return arr

    @staticmethod
    def two_pointers(arr, target):
        """Two pointers technique to find pairs in a sorted array."""
        left, right = 0, len(arr) - 1
        while left < right:
            if arr[left] + arr[right] == target:
                return (left, right)
            elif arr[left] + arr[right] < target:
                left += 1
            else:
                right -= 1
        return None

    @staticmethod
    def sliding_window(arr, k):
        """Sliding window technique to find maximum sum of subarrays of size k."""
        max_sum = sum(arr[:k])
        window_sum = max_sum
        for i in range(k, len(arr)):
            window_sum += arr[i] - arr[i - k]
            max_sum = max(max_sum, window_sum)
        return max_sum

    @staticmethod
    def kadane(arr):
        """Kadane's algorithm to find maximum subarray sum."""
        max_current = max_global = arr[0]
        for i in range(1, len(arr)):
            max_current = max(arr[i], max_current + arr[i])
            if max_current > max_global:
                max_global = max_current
        return max_global

    @staticmethod
    def bubble_sort(arr):
        """Bubble sort implementation."""
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr

    @staticmethod
    def selection_sort(arr):
        """Selection sort implementation."""
        for i in range(len(arr)):
            min_idx = i
            for j in range(i + 1, len(arr)):
                if arr[j] < arr[min_idx]:
                    min_idx = j
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
        return arr

    @staticmethod
    def insertion_sort(arr):
        """Insertion sort implementation."""
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0 and key < arr[j]:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
        return arr

    @staticmethod
    def merge_sort(arr):
        """Merge sort implementation."""
        if len(arr) > 1:
            mid = len(arr) // 2
            left = arr[:mid]
            right = arr[mid:]
            Arrays.merge_sort(left)
            Arrays.merge_sort(right)
            i = j = k = 0
            while i < len(left) and j < len(right):
                if left[i] < right[j]:
                    arr[k] = left[i]
                    i += 1
                else:
                    arr[k] = right[j]
                    j += 1
                k += 1
            while i < len(left):
                arr[k] = left[i]
                i += 1
                k += 1
            while j < len(right):
                arr[k] = right[j]
                j += 1
                k += 1
        return arr

    @staticmethod
    def quick_sort(arr):
        """Quick sort implementation."""
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return Arrays.quick_sort(left) + middle + Arrays.quick_sort(right)

    @staticmethod
    def counting_sort(arr):
        """Counting sort implementation."""
        max_val = max(arr)
        count = [0] * (max_val + 1)
        for num in arr:
            count[num] += 1
        sorted_arr = []
        for i in range(len(count)):
            sorted_arr.extend([i] * count[i])
        return sorted_arr

    @staticmethod
    def radix_sort(arr):
        """Radix sort implementation."""
        max_val = max(arr)
        exp = 1
        while max_val // exp > 0:
            Arrays.counting_sort(arr)
            exp *= 10
        return arr

    @staticmethod
    def linear_search(arr, target):
        """Linear search implementation."""
        for i in range(len(arr)):
            if arr[i] == target:
                return i
        return -1

    @staticmethod
    def binary_search(arr, target):
        """Binary search implementation."""
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1

    @staticmethod
    def dutch_national_flag(arr):
        """Dutch National Flag algorithm for 3-way partitioning."""
        low, mid, high = 0, 0, len(arr) - 1
        while mid <= high:
            if arr[mid] == 0:
                arr[low], arr[mid] = arr[mid], arr[low]
                low += 1
                mid += 1
            elif arr[mid] == 1:
                mid += 1
            else:
                arr[mid], arr[high] = arr[high], arr[mid]
                high -= 1
        return arr

    @staticmethod
    def rotate_array(arr, k):
        """Rotate an array using the reversal algorithm."""
        def reverse(arr, start, end):
            while start < end:
                arr[start], arr[end] = arr[end], arr[start]
                start += 1
                end -= 1
        n = len(arr)
        k %= n
        reverse(arr, 0, n - 1)
        reverse(arr, 0, k - 1)
        reverse(arr, k, n - 1)
        return arr

    @staticmethod
    def majority_element(arr):
        """Moore's Voting Algorithm to find majority element."""
        candidate, count = None, 0
        for num in arr:
            if count == 0:
                candidate = num
            count += 1 if num == candidate else -1
        return candidate

    @staticmethod
    def merge_intervals(intervals):
        """Merge overlapping intervals."""
        intervals.sort(key=lambda x: x[0])
        merged = []
        for interval in intervals:
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                merged[-1][1] = max(merged[-1][1], interval[1])
        return merged


# ------------------------- 2.2 Strings -------------------------

class Strings:
    @staticmethod
    def reverse_string(s):
        """Reverse a string."""
        return s[::-1]

    @staticmethod
    def is_palindrome(s):
        """Check if a string is a palindrome."""
        return s == s[::-1]

    @staticmethod
    def substrings(s):
        """Generate all substrings of a string."""
        n = len(s)
        return [s[i:j] for i in range(n) for j in range(i + 1, n + 1)]

    @staticmethod
    def kmp(s, pattern):
        """KMP algorithm for pattern matching."""
        def compute_lps(pattern):
            lps = [0] * len(pattern)
            length = 0
            i = 1
            while i < len(pattern):
                if pattern[i] == pattern[length]:
                    length += 1
                    lps[i] = length
                    i += 1
                else:
                    if length != 0:
                        length = lps[length - 1]
                    else:
                        lps[i] = 0
                        i += 1
            return lps

        lps = compute_lps(pattern)
        i, j = 0, 0
        while i < len(s):
            if pattern[j] == s[i]:
                i += 1
                j += 1
            if j == len(pattern):
                return i - j
            elif i < len(s) and pattern[j] != s[i]:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1
        return -1

    @staticmethod
    def rabin_karp(s, pattern):
        """Rabin-Karp algorithm for pattern matching."""
        d = 256
        q = 101
        n, m = len(s), len(pattern)
        h = pow(d, m - 1, q)
        p = 0
        t = 0
        for i in range(m):
            p = (d * p + ord(pattern[i])) % q
            t = (d * t + ord(s[i])) % q
        for i in range(n - m + 1):
            if p == t:
                if s[i:i + m] == pattern:
                    return i
            if i < n - m:
                t = (d * (t - ord(s[i]) * h) + ord(s[i + m])) % q
        return -1

    @staticmethod
    def z_algorithm(s):
        """Z algorithm for pattern matching."""
        n = len(s)
        z = [0] * n
        l, r = 0, 0
        for i in range(1, n):
            if i > r:
                l = r = i
                while r < n and s[r - l] == s[r]:
                    r += 1
                z[i] = r - l
                r -= 1
            else:
                k = i - l
                if z[k] < r - i + 1:
                    z[i] = z[k]
                else:
                    l = i
                    while r < n and s[r - l] == s[r]:
                        r += 1
                    z[i] = r - l
                    r -= 1
        return z

    @staticmethod
    def rolling_hash(s, base=256, mod=101):
        """Rolling hash function."""
        hash_value = 0
        for char in s:
            hash_value = (hash_value * base + ord(char)) % mod
        return hash_value

    @staticmethod
    def lcs(s1, s2):
        """Longest Common Subsequence (LCS) using dynamic programming."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    @staticmethod
    def longest_palindromic_substring(s):
        """Manacher's algorithm to find the longest palindromic substring."""
        t = '#'.join(f'^{s}$')
        n = len(t)
        p = [0] * n
        c, r = 0, 0
        for i in range(1, n - 1):
            mirror = 2 * c - i
            if i < r:
                p[i] = min(r - i, p[mirror])
            while t[i + p[i] + 1] == t[i - p[i] - 1]:
                p[i] += 1
            if i + p[i] > r:
                c, r = i, i + p[i]
        max_len, center = max((n, i) for i, n in enumerate(p))
        return s[(center - max_len) // 2:(center + max_len) // 2]

    @staticmethod
    def are_anagrams(s1, s2):
        """Check if two strings are anagrams."""
        return sorted(s1) == sorted(s2)

    @staticmethod
    def run_length_encoding(s):
        """Run-length encoding for string compression."""
        encoded = []
        count = 1
        for i in range(1, len(s)):
            if s[i] == s[i - 1]:
                count += 1
            else:
                encoded.append(f"{s[i - 1]}{count}")
                count = 1
        encoded.append(f"{s[-1]}{count}")
        return ''.join(encoded)


# ------------------------- 2.3 Linked List -------------------------

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        """Append a node to the end of the linked list."""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        last = self.head
        while last.next:
            last = last.next
        last.next = new_node

    def prepend(self, data):
        """Prepend a node to the beginning of the linked list."""
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node

    def delete(self, key):
        """Delete a node with the given key."""
        temp = self.head
        if temp and temp.data == key:
            self.head = temp.next
            temp = None
            return
        prev = None
        while temp and temp.data != key:
            prev = temp
            temp = temp.next
        if not temp:
            return
        prev.next = temp.next
        temp = None

    def reverse(self):
        """Reverse the linked list."""
        prev = None
        current = self.head
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        self.head = prev

    def detect_cycle(self):
        """Detect a cycle in the linked list using Floyd's algorithm."""
        slow = fast = self.head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False

    def remove_nth_from_end(self, n):
        """Remove the nth node from the end of the linked list."""
        fast = slow = self.head
        for _ in range(n):
            fast = fast.next
        if not fast:
            return self.head.next
        while fast.next:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return self.head

    def merge_sorted_lists(self, l2):
        """Merge two sorted linked lists."""
        dummy = Node(0)
        tail = dummy
        l1 = self.head
        while l1 and l2:
            if l1.data <= l2.data:
                tail.next = l1
                l1 = l1.next
            else:
                tail.next = l2
                l2 = l2.next
            tail = tail.next
        if l1:
            tail.next = l1
        if l2:
            tail.next = l2
        return dummy.next

    def find_intersection(self, l2):
        """Find the intersection point of two linked lists."""
        len1, len2 = 0, 0
        temp1, temp2 = self.head, l2.head
        while temp1:
            len1 += 1
            temp1 = temp1.next
        while temp2:
            len2 += 1
            temp2 = temp2.next
        temp1, temp2 = self.head, l2.head
        if len1 > len2:
            for _ in range(len1 - len2):
                temp1 = temp1.next
        else:
            for _ in range(len2 - len1):
                temp2 = temp2.next
        while temp1 and temp2:
            if temp1 == temp2:
                return temp1
            temp1 = temp1.next
            temp2 = temp2.next
        return None

    def clone_with_random_pointers(self):
        """Clone a linked list with random pointers."""
        if not self.head:
            return None
        mapping = {}
        current = self.head
        while current:
            mapping[current] = Node(current.data)
            current = current.next
        current = self.head
        while current:
            mapping[current].next = mapping.get(current.next)
            mapping[current].random = mapping.get(current.random)
            current = current.next
        return mapping[self.head]


# ------------------------- 2.4 Stack and Queue -------------------------

class Stack:
    def __init__(self):
        self.stack = []

    def push(self, item):
        """Push an item onto the stack."""
        self.stack.append(item)

    def pop(self):
        """Pop an item from the stack."""
        if not self.is_empty():
            return self.stack.pop()
        return None

    def top(self):
        """Get the top item of the stack."""
        if not self.is_empty():
            return self.stack[-1]
        return None

    def is_empty(self):
        """Check if the stack is empty."""
        return len(self.stack) == 0

    def min_stack(self):
        """Get the minimum element in the stack."""
        return min(self.stack) if self.stack else None

class Queue:
    def __init__(self):
        self.queue = []

    def enqueue(self, item):
        """Enqueue an item into the queue."""
        self.queue.append(item)

    def dequeue(self):
        """Dequeue an item from the queue."""
        if not self.is_empty():
            return self.queue.pop(0)
        return None

    def is_empty(self):
        """Check if the queue is empty."""
        return len(self.queue) == 0

    def circular_queue(self, size):
        """Implement a circular queue."""
        self.queue = [None] * size
        self.front = self.rear = -1

    def next_greater_element(self, arr):
        """Find the next greater element for each element in the array."""
        stack = []
        result = [-1] * len(arr)
        for i in range(len(arr)):
            while stack and arr[stack[-1]] < arr[i]:
                result[stack.pop()] = arr[i]
            stack.append(i)
        return result

    def balanced_parentheses(self, s):
        """Check if the parentheses are balanced."""
        stack = []
        mapping = {')': '(', '}': '{', ']': '['}
        for char in s:
            if char in mapping:
                top = stack.pop() if stack else '#'
                if mapping[char] != top:
                    return False
            else:
                stack.append(char)
        return not stack

    def lru_cache(self, capacity):
        """Implement an LRU cache."""
        from collections import OrderedDict
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        """Get an item from the LRU cache."""
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        """Put an item into the LRU cache."""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def sliding_window_maximum(self, arr, k):
        """Find the maximum in each sliding window of size k."""
        from collections import deque
        q = deque()
        result = []
        for i in range(len(arr)):
            while q and arr[q[-1]] <= arr[i]:
                q.pop()
            q.append(i)
            if q[0] == i - k:
                q.popleft()
            if i >= k - 1:
                result.append(arr[q[0]])
        return result


# ------------------------- 2.5 Hashing -------------------------

class Hashing:
    @staticmethod
    def hash_table():
        """Create a hash table."""
        return {}

    @staticmethod
    def open_addressing():
        """Open addressing for collision resolution."""
        pass

    @staticmethod
    def separate_chaining():
        """Separate chaining for collision resolution."""
        pass

    @staticmethod
    def count_distinct_elements(arr):
        """Count distinct elements in an array."""
        return len(set(arr))

    @staticmethod
    def longest_consecutive_subsequence(arr):
        """Find the longest consecutive subsequence."""
        num_set = set(arr)
        longest = 0
        for num in num_set:
            if num - 1 not in num_set:
                current_num = num
                current_streak = 1
                while current_num + 1 in num_set:
                    current_num += 1
                    current_streak += 1
                longest = max(longest, current_streak)
        return longest

    @staticmethod
    def two_sum(arr, target):
        """Find two numbers that add up to the target."""
        num_map = {}
        for i, num in enumerate(arr):
            complement = target - num
            if complement in num_map:
                return (num_map[complement], i)
            num_map[num] = i
        return None

    @staticmethod
    def group_anagrams(strs):
        """Group anagrams together."""
        from collections import defaultdict
        anagrams = defaultdict(list)
        for s in strs:
            key = ''.join(sorted(s))
            anagrams[key].append(s)
        return list(anagrams.values())


# ------------------------- 2.6 Binary Trees & Binary Search Trees -------------------------

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BinaryTree:
    def inorder_traversal(self, root):
        """Inorder traversal of a binary tree."""
        result = []
        def traverse(node):
            if node:
                traverse(node.left)
                result.append(node.val)
                traverse(node.right)
        traverse(root)
        return result

    def preorder_traversal(self, root):
        """Preorder traversal of a binary tree."""
        result = []
        def traverse(node):
            if node:
                result.append(node.val)
                traverse(node.left)
                traverse(node.right)
        traverse(root)
        return result

    def postorder_traversal(self, root):
        """Postorder traversal of a binary tree."""
        result = []
        def traverse(node):
            if node:
                traverse(node.left)
                traverse(node.right)
                result.append(node.val)
        traverse(root)
        return result

    def level_order_traversal(self, root):
        """Level order traversal of a binary tree."""
        from collections import deque
        result = []
        if not root:
            return result
        queue = deque([root])
        while queue:
            level_size = len(queue)
            current_level = []
            for _ in range(level_size):
                node = queue.popleft()
                current_level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            result.append(current_level)
        return result

    def height(self, root):
        """Calculate the height of a binary tree."""
        if not root:
            return 0
        return 1 + max(self.height(root.left), self.height(root.right))

    def diameter(self, root):
        """Calculate the diameter of a binary tree."""
        self.diameter = 0
        def height(node):
            if not node:
                return 0
            left = height(node.left)
            right = height(node.right)
            self.diameter = max(self.diameter, left + right)
            return 1 + max(left, right)
        height(root)
        return self.diameter

    def lowest_common_ancestor(self, root, p, q):
        """Find the lowest common ancestor of two nodes in a binary tree."""
        if not root or root == p or root == q:
            return root
        left = self.lowest_common_ancestor(root.left, p, q)
        right = self.lowest_common_ancestor(root.right, p, q)
        if left and right:
            return root
        return left if left else right

    def is_balanced(self, root):
        """Check if a binary tree is balanced."""
        def check(node):
            if not node:
                return 0
            left = check(node.left)
            right = check(node.right)
            if left == -1 or right == -1 or abs(left - right) > 1:
                return -1
            return 1 + max(left, right)
        return check(root) != -1

    def is_identical(self, root1, root2):
        """Check if two binary trees are identical."""
        if not root1 and not root2:
            return True
        if not root1 or not root2:
            return False
        return (root1.val == root2.val and
                self.is_identical(root1.left, root2.left) and
                self.is_identical(root1.right, root2.right))

    def kth_smallest(self, root, k):
        """Find the kth smallest element in a BST."""
        stack = []
        while True:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            k -= 1
            if k == 0:
                return root.val
            root = root.right

    def sorted_array_to_bst(self, arr):
        """Convert a sorted array to a balanced BST."""
        if not arr:
            return None
        mid = len(arr) // 2
        root = TreeNode(arr[mid])
        root.left = self.sorted_array_to_bst(arr[:mid])
        root.right = self.sorted_array_to_bst(arr[mid + 1:])
        return root


# ------------------------- 2.7 Heaps & Priority Queues -------------------------

import heapq

class Heaps:
    @staticmethod
    def min_heap(arr):
        """Create a min heap."""
        heapq.heapify(arr)
        return arr

    @staticmethod
    def max_heap(arr):
        """Create a max heap."""
        heapq._heapify_max(arr)
        return arr

    @staticmethod
    def heap_sort(arr):
        """Heap sort implementation."""
        heapq.heapify(arr)
        return [heapq.heappop(arr) for _ in range(len(arr))]

    @staticmethod
    def kth_largest(arr, k):
        """Find the kth largest element in an array."""
        return heapq.nlargest(k, arr)[-1]

    @staticmethod
    def kth_smallest(arr, k):
        """Find the kth smallest element in an array."""
        return heapq.nsmallest(k, arr)[-1]

    @staticmethod
    def merge_k_sorted_lists(lists):
        """Merge k sorted lists."""
        import heapq
        heap = []
        for i, lst in enumerate(lists):
            if lst:
                heapq.heappush(heap, (lst[0], i, 0))
        result = []
        while heap:
            val, list_idx, element_idx = heapq.heappop(heap)
            result.append(val)
            if element_idx + 1 < len(lists[list_idx]):
                heapq.heappush(heap, (lists[list_idx][element_idx + 1], list_idx, element_idx + 1))
        return result

    @staticmethod
    def median_of_running_stream():
        """Find the median of a running stream of numbers."""
        import heapq
        min_heap = []
        max_heap = []
        def add_num(num):
            heapq.heappush(max_heap, -heapq.heappushpop(min_heap, num))
            if len(max_heap) > len(min_heap):
                heapq.heappush(min_heap, -heapq.heappop(max_heap))
        def find_median():
            if len(min_heap) == len(max_heap):
                return (min_heap[0] - max_heap[0]) / 2
            return min_heap[0]
        return add_num, find_median

    @staticmethod
    def top_k_frequent_elements(arr, k):
        """Find the top k frequent elements in an array."""
        from collections import Counter
        count = Counter(arr)
        return heapq.nlargest(k, count.keys(), key=count.get)


# ------------------------- 2.8 Graphs -------------------------

class Graph:
    def __init__(self, vertices):
        self.vertices = vertices
        self.adj_list = {v: [] for v in range(vertices)}

    def add_edge(self, u, v):
        """Add an edge to the graph."""
        self.adj_list[u].append(v)
        self.adj_list[v].append(u)

    def bfs(self, start):
        """Breadth-First Search (BFS) for graph traversal."""
        visited = set()
        queue = [start]
        result = []
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                result.append(node)
                queue.extend(self.adj_list[node])
        return result

    def dfs(self, start):
        """Depth-First Search (DFS) for graph traversal."""
        visited = set()
        result = []
        def traverse(node):
            if node not in visited:
                visited.add(node)
                result.append(node)
                for neighbor in self.adj_list[node]:
                    traverse(neighbor)
        traverse(start)
        return result

    def detect_cycle_undirected(self):
        """Detect a cycle in an undirected graph."""
        visited = set()
        def dfs(node, parent):
            visited.add(node)
            for neighbor in self.adj_list[node]:
                if neighbor not in visited:
                    if dfs(neighbor, node):
                        return True
                elif neighbor != parent:
                    return True
            return False
        for node in range(self.vertices):
            if node not in visited:
                if dfs(node, -1):
                    return True
        return False

    def detect_cycle_directed(self):
        """Detect a cycle in a directed graph."""
        visited = set()
        rec_stack = set()
        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            for neighbor in self.adj_list[node]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(node)
            return False
        for node in range(self.vertices):
            if node not in visited:
                if dfs(node):
                    return True
        return False

    def topological_sort(self):
        """Topological sorting using Kahn's algorithm."""
        in_degree = [0] * self.vertices
        for u in range(self.vertices):
            for v in self.adj_list[u]:
                in_degree[v] += 1
        queue = [u for u in range(self.vertices) if in_degree[u] == 0]
        result = []
        while queue:
            u = queue.pop(0)
            result.append(u)
            for v in self.adj_list[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        return result if len(result) == self.vertices else []

    def dijkstra(self, start):
        """Dijkstra's algorithm for shortest path."""
        import heapq
        distances = [float('inf')] * self.vertices
        distances[start] = 0
        heap = [(0, start)]
        while heap:
            current_distance, u = heapq.heappop(heap)
            if current_distance > distances[u]:
                continue
            for v in self.adj_list[u]:
                distance = current_distance + 1  # Assuming unweighted graph
                if distance < distances[v]:
                    distances[v] = distance
                    heapq.heappush(heap, (distance, v))
        return distances

    def bellman_ford(self, start):
        """Bellman-Ford algorithm for shortest path."""
        distances = [float('inf')] * self.vertices
        distances[start] = 0
        for _ in range(self.vertices - 1):
            for u in range(self.vertices):
                for v in self.adj_list[u]:
                    if distances[u] + 1 < distances[v]:  # Assuming unweighted graph
                        distances[v] = distances[u] + 1
        return distances

    def floyd_warshall(self):
        """Floyd-Warshall algorithm for shortest path."""
        distances = [[float('inf')] * self.vertices for _ in range(self.vertices)]
        for u in range(self.vertices):
            distances[u][u] = 0
            for v in self.adj_list[u]:
                distances[u][v] = 1  # Assuming unweighted graph
        for k in range(self.vertices):
            for i in range(self.vertices):
                for j in range(self.vertices):
                    distances[i][j] = min(distances[i][j], distances[i][k] + distances[k][j])
        return distances

    def kruskal(self):
        """Kruskal's algorithm for minimum spanning tree."""
        parent = [i for i in range(self.vertices)]
        def find(u):
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]
        def union(u, v):
            u_root = find(u)
            v_root = find(v)
            if u_root != v_root:
                parent[v_root] = u_root
        edges = []
        for u in range(self.vertices):
            for v in self.adj_list[u]:
                edges.append((1, u, v))  # Assuming unweighted graph
        edges.sort()
        mst = []
        for weight, u, v in edges:
            if find(u) != find(v):
                union(u, v)
                mst.append((u, v))
        return mst

    def prim(self):
        """Prim's algorithm for minimum spanning tree."""
        import heapq
        visited = set()
        heap = [(0, 0)]  # (weight, node)
        mst = []
        while heap:
            weight, u = heapq.heappop(heap)
            if u not in visited:
                visited.add(u)
                mst.append((u, weight))
                for v in self.adj_list[u]:
                    if v not in visited:
                        heapq.heappush(heap, (1, v))  # Assuming unweighted graph
        return mst

    def kosaraju(self):
        """Kosaraju's algorithm for strongly connected components."""
        visited = set()
        order = []
        def dfs(u):
            visited.add(u)
            for v in self.adj_list[u]:
                if v not in visited:
                    dfs(v)
            order.append(u)
        for u in range(self.vertices):
            if u not in visited:
                dfs(u)
        reversed_graph = {v: [] for v in range(self.vertices)}
        for u in range(self.vertices):
            for v in self.adj_list[u]:
                reversed_graph[v].append(u)
        visited = set()
        sccs = []
        def dfs_reversed(u, component):
            visited.add(u)
            component.append(u)
            for v in reversed_graph[u]:
                if v not in visited:
                    dfs_reversed(v, component)
        for u in reversed(order):
            if u not in visited:
                component = []
                dfs_reversed(u, component)
                sccs.append(component)
        return sccs

    def find_bridges(self):
        """Find bridges in a graph."""
        visited = set()
        low = [float('inf')] * self.vertices
        disc = [float('inf')] * self.vertices
        parent = [-1] * self.vertices
        bridges = []
        time = 0
        def dfs(u):
            nonlocal time
            visited.add(u)
            disc[u] = low[u] = time
            time += 1
            for v in self.adj_list[u]:
                if v not in visited:
                    parent[v] = u
                    dfs(v)
                    low[u] = min(low[u], low[v])
                    if low[v] > disc[u]:
                        bridges.append((u, v))
                elif v != parent[u]:
                    low[u] = min(low[u], disc[v])
        for u in range(self.vertices):
            if u not in visited:
                dfs(u)
        return bridges

    def articulation_points(self):
        """Find articulation points in a graph."""
        visited = set()
        low = [float('inf')] * self.vertices
        disc = [float('inf')] * self.vertices
        parent = [-1] * self.vertices
        ap = set()
        time = 0
        def dfs(u):
            nonlocal time
            visited.add(u)
            disc[u] = low[u] = time
            time += 1
            children = 0
            for v in self.adj_list[u]:
                if v not in visited:
                    parent[v] = u
                    children += 1
                    dfs(v)
                    low[u] = min(low[u], low[v])
                    if parent[u] == -1 and children > 1:
                        ap.add(u)
                    if parent[u] != -1 and low[v] >= disc[u]:
                        ap.add(u)
                elif v != parent[u]:
                    low[u] = min(low[u], disc[v])
        for u in range(self.vertices):
            if u not in visited:
                dfs(u)
        return ap

    def ford_fulkerson(self, source, sink):
        """Ford-Fulkerson algorithm for maximum flow."""
        def bfs(u, v, parent):
            visited = set()
            queue = [u]
            visited.add(u)
            while queue:
                u = queue.pop(0)
                for v in self.adj_list[u]:
                    if v not in visited and residual[u][v] > 0:
                        queue.append(v)
                        visited.add(v)
                        parent[v] = u
                        if v == sink:
                            return True
            return False
        residual = [[0] * self.vertices for _ in range(self.vertices)]
        for u in range(self.vertices):
            for v in self.adj_list[u]:
                residual[u][v] = 1  # Assuming unweighted graph
        max_flow = 0
        parent = [-1] * self.vertices
        while bfs(source, sink, parent):
            path_flow = float('inf')
            s = sink
            while s != source:
                path_flow = min(path_flow, residual[parent[s]][s])
                s = parent[s]
            max_flow += path_flow
            v = sink
            while v != source:
                u = parent[v]
                residual[u][v] -= path_flow
                residual[v][u] += path_flow
                v = parent[v]
        return max_flow


# ------------------------- Print All Functions -------------------------

class DataStructureExamples:
    """
    A class to provide examples and usage instructions for each data structure and algorithm.
    """

    @staticmethod
    def arrays_examples():
        """
        Returns examples and usage instructions for the Arrays class.
        """
        return """
        ## Arrays Class Examples

        - traverse(arr): Traverse an array.
          Example:
            Arrays.traverse([1, 2, 3])  # Output: 1 2 3

        - insert(arr, index, value): Insert an element into an array at a specific index.
          Example:
            Arrays.insert([1, 2, 3], 1, 5)  # Output: [1, 5, 2, 3]

        - delete(arr, index): Delete an element from an array at a specific index.
          Example:
            Arrays.delete([1, 2, 3], 1)  # Output: [1, 3]

        - prefix_sum(arr): Calculate the prefix sum of an array.
          Example:
            Arrays.prefix_sum([1, 2, 3, 4])  # Output: [1, 3, 6, 10]

        - suffix_sum(arr): Calculate the suffix sum of an array.
          Example:
            Arrays.suffix_sum([1, 2, 3, 4])  # Output: [10, 9, 7, 4]

        - two_pointers(arr, target): Two pointers technique to find pairs in a sorted array.
          Example:
            Arrays.two_pointers([1, 2, 3, 4], 5)  # Output: (1, 2)

        - sliding_window(arr, k): Sliding window technique to find maximum sum of subarrays of size k.
          Example:
            Arrays.sliding_window([1, 2, 3, 4, 5], 3)  # Output: 12

        - kadane(arr): Kadane's algorithm to find maximum subarray sum.
          Example:
            Arrays.kadane([-2, 1, -3, 4, -1, 2, 1, -5, 4])  # Output: 6

        - bubble_sort(arr): Bubble sort implementation.
          Example:
            Arrays.bubble_sort([3, 1, 4, 2])  # Output: [1, 2, 3, 4]

        - selection_sort(arr): Selection sort implementation.
          Example:
            Arrays.selection_sort([3, 1, 4, 2])  # Output: [1, 2, 3, 4]

        - insertion_sort(arr): Insertion sort implementation.
          Example:
            Arrays.insertion_sort([3, 1, 4, 2])  # Output: [1, 2, 3, 4]

        - merge_sort(arr): Merge sort implementation.
          Example:
            Arrays.merge_sort([3, 1, 4, 2])  # Output: [1, 2, 3, 4]

        - quick_sort(arr): Quick sort implementation.
          Example:
            Arrays.quick_sort([3, 1, 4, 2])  # Output: [1, 2, 3, 4]

        - counting_sort(arr): Counting sort implementation.
          Example:
            Arrays.counting_sort([3, 1, 4, 2])  # Output: [1, 2, 3, 4]

        - radix_sort(arr): Radix sort implementation.
          Example:
            Arrays.radix_sort([3, 1, 4, 2])  # Output: [1, 2, 3, 4]

        - linear_search(arr, target): Linear search implementation.
          Example:
            Arrays.linear_search([1, 2, 3, 4], 3)  # Output: 2

        - binary_search(arr, target): Binary search implementation.
          Example:
            Arrays.binary_search([1, 2, 3, 4], 3)  # Output: 2

        - dutch_national_flag(arr): Dutch National Flag algorithm for 3-way partitioning.
          Example:
            Arrays.dutch_national_flag([2, 0, 2, 1, 1, 0])  # Output: [0, 0, 1, 1, 2, 2]

        - rotate_array(arr, k): Rotate an array using the reversal algorithm.
          Example:
            Arrays.rotate_array([1, 2, 3, 4, 5], 2)  # Output: [4, 5, 1, 2, 3]

        - majority_element(arr): Moore's Voting Algorithm to find majority element.
          Example:
            Arrays.majority_element([3, 3, 4, 2, 4, 4, 2, 4, 4])  # Output: 4

        - merge_intervals(intervals): Merge overlapping intervals.
          Example:
            Arrays.merge_intervals([[1, 3], [2, 6], [8, 10]])  # Output: [[1, 6], [8, 10]]
        """

    @staticmethod
    def strings_examples():
        """
        Returns examples and usage instructions for the Strings class.
        """
        return """
        ## Strings Class Examples

        - reverse_string(s): Reverse a string.
          Example:
            Strings.reverse_string("hello")  # Output: "olleh"

        - is_palindrome(s): Check if a string is a palindrome.
          Example:
            Strings.is_palindrome("madam")  # Output: True

        - substrings(s): Generate all substrings of a string.
          Example:
            Strings.substrings("abc")  # Output: ['a', 'ab', 'abc', 'b', 'bc', 'c']

        - kmp(s, pattern): KMP algorithm for pattern matching.
          Example:
            Strings.kmp("hello", "ll")  # Output: 2

        - rabin_karp(s, pattern): Rabin-Karp algorithm for pattern matching.
          Example:
            Strings.rabin_karp("hello", "ll")  # Output: 2

        - z_algorithm(s): Z algorithm for pattern matching.
          Example:
            Strings.z_algorithm("hello")  # Output: [0, 0, 0, 0, 0]

        - rolling_hash(s, base=256, mod=101): Rolling hash function.
          Example:
            Strings.rolling_hash("hello")  # Output: 52

        - lcs(s1, s2): Longest Common Subsequence (LCS) using dynamic programming.
          Example:
            Strings.lcs("abcde", "ace")  # Output: 3

        - longest_palindromic_substring(s): Manacher's algorithm to find the longest palindromic substring.
          Example:
            Strings.longest_palindromic_substring("babad")  # Output: "bab"

        - are_anagrams(s1, s2): Check if two strings are anagrams.
          Example:
            Strings.are_anagrams("listen", "silent")  # Output: True

        - run_length_encoding(s): Run-length encoding for string compression.
          Example:
            Strings.run_length_encoding("aaabbbcc")  # Output: "a3b3c2"
        """

    @staticmethod
    def linked_list_examples():
        """
        Returns examples and usage instructions for the LinkedList class.
        """
        return """
        ## LinkedList Class Examples

        - append(data): Append a node to the end of the linked list.
          Example:
            ll = LinkedList()
            ll.append(1)
            ll.append(2)
            # Linked List: 1 -> 2

        - prepend(data): Prepend a node to the beginning of the linked list.
          Example:
            ll = LinkedList()
            ll.prepend(1)
            ll.prepend(2)
            # Linked List: 2 -> 1

        - delete(key): Delete a node with the given key.
          Example:
            ll = LinkedList()
            ll.append(1)
            ll.append(2)
            ll.delete(1)
            # Linked List: 2

        - reverse(): Reverse the linked list.
          Example:
            ll = LinkedList()
            ll.append(1)
            ll.append(2)
            ll.reverse()
            # Linked List: 2 -> 1

        - detect_cycle(): Detect a cycle in the linked list using Floyd's algorithm.
          Example:
            ll = LinkedList()
            ll.append(1)
            ll.append(2)
            ll.head.next.next = ll.head  # Creates a cycle
            ll.detect_cycle()  # Output: True

        - remove_nth_from_end(n): Remove the nth node from the end of the linked list.
          Example:
            ll = LinkedList()
            ll.append(1)
            ll.append(2)
            ll.append(3)
            ll.remove_nth_from_end(2)
            # Linked List: 1 -> 3

        - merge_sorted_lists(l2): Merge two sorted linked lists.
          Example:
            ll1 = LinkedList()
            ll1.append(1)
            ll1.append(3)
            ll2 = LinkedList()
            ll2.append(2)
            ll2.append(4)
            ll1.merge_sorted_lists(ll2)
            # Linked List: 1 -> 2 -> 3 -> 4

        - find_intersection(l2): Find the intersection point of two linked lists.
          Example:
            ll1 = LinkedList()
            ll1.append(1)
            ll1.append(2)
            ll2 = LinkedList()
            ll2.append(3)
            ll2.head.next = ll1.head.next  # Creates intersection
            ll1.find_intersection(ll2)  # Output: Node with value 2

        - clone_with_random_pointers(): Clone a linked list with random pointers.
          Example:
            ll = LinkedList()
            ll.append(1)
            ll.append(2)
            cloned_ll = ll.clone_with_random_pointers()
            # Cloned Linked List: 1 -> 2
        """

    @staticmethod
    def stack_examples():
        """
        Returns examples and usage instructions for the Stack class.
        """
        return """
        ## Stack Class Examples

        - push(item): Push an item onto the stack.
          Example:
            stack = Stack()
            stack.push(1)
            stack.push(2)
            # Stack: [1, 2]

        - pop(): Pop an item from the stack.
          Example:
            stack = Stack()
            stack.push(1)
            stack.push(2)
            stack.pop()  # Output: 2

        - top(): Get the top item of the stack.
          Example:
            stack = Stack()
            stack.push(1)
            stack.push(2)
            stack.top()  # Output: 2

        - is_empty(): Check if the stack is empty.
          Example:
            stack = Stack()
            stack.is_empty()  # Output: True

        - min_stack(): Get the minimum element in the stack.
          Example:
            stack = Stack()
            stack.push(2)
            stack.push(1)
            stack.min_stack()  # Output: 1
        """

    @staticmethod
    def queue_examples():
        """
        Returns examples and usage instructions for the Queue class.
        """
        return """
        ## Queue Class Examples

        - enqueue(item): Enqueue an item into the queue.
          Example:
            queue = Queue()
            queue.enqueue(1)
            queue.enqueue(2)
            # Queue: [1, 2]

        - dequeue(): Dequeue an item from the queue.
          Example:
            queue = Queue()
            queue.enqueue(1)
            queue.enqueue(2)
            queue.dequeue()  # Output: 1

        - is_empty(): Check if the queue is empty.
          Example:
            queue = Queue()
            queue.is_empty()  # Output: True

        - circular_queue(size): Implement a circular queue.
          Example:
            queue = Queue()
            queue.circular_queue(5)
            queue.enqueue(1)
            queue.enqueue(2)
            # Circular Queue: [1, 2, None, None, None]

        - next_greater_element(arr): Find the next greater element for each element in the array.
          Example:
            queue = Queue()
            queue.next_greater_element([4, 5, 2, 25])  # Output: [5, 25, 25, -1]

        - balanced_parentheses(s): Check if the parentheses are balanced.
          Example:
            queue = Queue()
            queue.balanced_parentheses("()[]{}")  # Output: True

        - lru_cache(capacity): Implement an LRU cache.
          Example:
            cache = Queue()
            cache.lru_cache(2)
            cache.put(1, 1)
            cache.put(2, 2)
            cache.get(1)  # Output: 1

        - sliding_window_maximum(arr, k): Find the maximum in each sliding window of size k.
          Example:
            queue = Queue()
            queue.sliding_window_maximum([1, 3, -1, -3, 5, 3, 6, 7], 3)  # Output: [3, 3, 5, 5, 6, 7]
        """

    @staticmethod
    def hashing_examples():
        """
        Returns examples and usage instructions for the Hashing class.
        """
        return """
        ## Hashing Class Examples

        - hash_table(): Create a hash table.
          Example:
            hash_table = Hashing.hash_table()
            hash_table["key"] = "value"
            # Hash Table: {"key": "value"}

        - open_addressing(): Open addressing for collision resolution.
          Example:
            Hashing.open_addressing()

        - separate_chaining(): Separate chaining for collision resolution.
          Example:
            Hashing.separate_chaining()

        - count_distinct_elements(arr): Count distinct elements in an array.
          Example:
            Hashing.count_distinct_elements([1, 2, 2, 3])  # Output: 3

        - longest_consecutive_subsequence(arr): Find the longest consecutive subsequence.
          Example:
            Hashing.longest_consecutive_subsequence([100, 4, 200, 1, 3, 2])  # Output: 4

        - two_sum(arr, target): Find two numbers that add up to the target.
          Example:
            Hashing.two_sum([2, 7, 11, 15], 9)  # Output: (0, 1)

        - group_anagrams(strs): Group anagrams together.
          Example:
            Hashing.group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
            # Output: [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]
        """

    @staticmethod
    def binary_tree_examples():
        """
        Returns examples and usage instructions for the BinaryTree class.
        """
        return """
        ## BinaryTree Class Examples

        - inorder_traversal(root): Inorder traversal of a binary tree.
          Example:
            root = TreeNode(1)
            root.left = TreeNode(2)
            root.right = TreeNode(3)
            BinaryTree().inorder_traversal(root)  # Output: [2, 1, 3]

        - preorder_traversal(root): Preorder traversal of a binary tree.
          Example:
            root = TreeNode(1)
            root.left = TreeNode(2)
            root.right = TreeNode(3)
            BinaryTree().preorder_traversal(root)  # Output: [1, 2, 3]

        - postorder_traversal(root): Postorder traversal of a binary tree.
          Example:
            root = TreeNode(1)
            root.left = TreeNode(2)
            root.right = TreeNode(3)
            BinaryTree().postorder_traversal(root)  # Output: [2, 3, 1]

        - level_order_traversal(root): Level order traversal of a binary tree.
          Example:
            root = TreeNode(1)
            root.left = TreeNode(2)
            root.right = TreeNode(3)
            BinaryTree().level_order_traversal(root)  # Output: [[1], [2, 3]]

        - height(root): Calculate the height of a binary tree.
          Example:
            root = TreeNode(1)
            root.left = TreeNode(2)
            root.right = TreeNode(3)
            BinaryTree().height(root)  # Output: 2

        - diameter(root): Calculate the diameter of a binary tree.
          Example:
            root = TreeNode(1)
            root.left = TreeNode(2)
            root.right = TreeNode(3)
            BinaryTree().diameter(root)  # Output: 2

        - lowest_common_ancestor(root, p, q): Find the lowest common ancestor of two nodes in a binary tree.
          Example:
            root = TreeNode(1)
            root.left = TreeNode(2)
            root.right = TreeNode(3)
            BinaryTree().lowest_common_ancestor(root, root.left, root.right)  # Output: TreeNode(1)

        - is_balanced(root): Check if a binary tree is balanced.
          Example:
            root = TreeNode(1)
            root.left = TreeNode(2)
            root.right = TreeNode(3)
            BinaryTree().is_balanced(root)  # Output: True

        - is_identical(root1, root2): Check if two binary trees are identical.
          Example:
            root1 = TreeNode(1)
            root1.left = TreeNode(2)
            root1.right = TreeNode(3)
            root2 = TreeNode(1)
            root2.left = TreeNode(2)
            root2.right = TreeNode(3)
            BinaryTree().is_identical(root1, root2)  # Output: True

        - kth_smallest(root, k): Find the kth smallest element in a BST.
          Example:
            root = TreeNode(3)
            root.left = TreeNode(1)
            root.right = TreeNode(4)
            BinaryTree().kth_smallest(root, 1)  # Output: 1

        - sorted_array_to_bst(arr): Convert a sorted array to a balanced BST.
          Example:
            BinaryTree().sorted_array_to_bst([1, 2, 3])  # Output: TreeNode(2) with left=1 and right=3
        """

    @staticmethod
    def heaps_examples():
        """
        Returns examples and usage instructions for the Heaps class.
        """
        return """
        ## Heaps Class Examples

        - min_heap(arr): Create a min heap.
          Example:
            Heaps.min_heap([3, 1, 4, 2])  # Output: [1, 2, 4, 3]

        - max_heap(arr): Create a max heap.
          Example:
            Heaps.max_heap([3, 1, 4, 2])  # Output: [4, 3, 2, 1]

        - heap_sort(arr): Heap sort implementation.
          Example:
            Heaps.heap_sort([3, 1, 4, 2])  # Output: [1, 2, 3, 4]

        - kth_largest(arr, k): Find the kth largest element in an array.
          Example:
            Heaps.kth_largest([3, 1, 4, 2], 2)  # Output: 3

        - kth_smallest(arr, k): Find the kth smallest element in an array.
          Example:
            Heaps.kth_smallest([3, 1, 4, 2], 2)  # Output: 2

        - merge_k_sorted_lists(lists): Merge k sorted lists.
          Example:
            Heaps.merge_k_sorted_lists([[1, 4, 5], [1, 3, 4], [2, 6]])  # Output: [1, 1, 2, 3, 4, 4, 5, 6]

        - median_of_running_stream(): Find the median of a running stream of numbers.
          Example:
            add_num, find_median = Heaps.median_of_running_stream()
            add_num(1)
            add_num(2)
            find_median()  # Output: 1.5

        - top_k_frequent_elements(arr, k): Find the top k frequent elements in an array.
          Example:
            Heaps.top_k_frequent_elements([1, 1, 1, 2, 2, 3], 2)  # Output: [1, 2]
        """

    @staticmethod
    def graph_examples():
        """
        Returns examples and usage instructions for the Graph class.
        """
        return """
        ## Graph Class Examples

        - add_edge(u, v): Add an edge to the graph.
          Example:
            graph = Graph(4)
            graph.add_edge(0, 1)
            graph.add_edge(1, 2)
            # Graph: {0: [1], 1: [0, 2], 2: [1]}

        - bfs(start): Breadth-First Search (BFS) for graph traversal.
          Example:
            graph = Graph(4)
            graph.add_edge(0, 1)
            graph.add_edge(1, 2)
            graph.bfs(0)  # Output: [0, 1, 2]

        - dfs(start): Depth-First Search (DFS) for graph traversal.
          Example:
            graph = Graph(4)
            graph.add_edge(0, 1)
            graph.add_edge(1, 2)
            graph.dfs(0)  # Output: [0, 1, 2]

        - detect_cycle_undirected(): Detect a cycle in an undirected graph.
          Example:
            graph = Graph(4)
            graph.add_edge(0, 1)
            graph.add_edge(1, 2)
            graph.add_edge(2, 0)
            graph.detect_cycle_undirected()  # Output: True

        - detect_cycle_directed(): Detect a cycle in a directed graph.
          Example:
            graph = Graph(4)
            graph.add_edge(0, 1)
            graph.add_edge(1, 2)
            graph.add_edge(2, 0)
            graph.detect_cycle_directed()  # Output: True

        - topological_sort(): Topological sorting using Kahn's algorithm.
          Example:
            graph = Graph(4)
            graph.add_edge(0, 1)
            graph.add_edge(1, 2)
            graph.topological_sort()  # Output: [0, 1, 2]

        - dijkstra(start): Dijkstra's algorithm for shortest path.
          Example:
            graph = Graph(4)
            graph.add_edge(0, 1)
            graph.add_edge(1, 2)
            graph.dijkstra(0)  # Output: [0, 1, 2, inf]

        - bellman_ford(start): Bellman-Ford algorithm for shortest path.
          Example:
            graph = Graph(4)
            graph.add_edge(0, 1)
            graph.add_edge(1, 2)
            graph.bellman_ford(0)  # Output: [0, 1, 2, inf]

        - floyd_warshall(): Floyd-Warshall algorithm for shortest path.
          Example:
            graph = Graph(4)
            graph.add_edge(0, 1)
            graph.add_edge(1, 2)
            graph.floyd_warshall()  # Output: [[0, 1, 2, inf], [inf, 0, 1, inf], [inf, inf, 0, inf], [inf, inf, inf, 0]]

        - kruskal(): Kruskal's algorithm for minimum spanning tree.
          Example:
            graph = Graph(4)
            graph.add_edge(0, 1)
            graph.add_edge(1, 2)
            graph.kruskal()  # Output: [(0, 1), (1, 2)]

        - prim(): Prim's algorithm for minimum spanning tree.
          Example:
            graph = Graph(4)
            graph.add_edge(0, 1)
            graph.add_edge(1, 2)
            graph.prim()  # Output: [(0, 1), (1, 2)]

        - kosaraju(): Kosaraju's algorithm for strongly connected components.
          Example:
            graph = Graph(4)
            graph.add_edge(0, 1)
            graph.add_edge(1, 2)
            graph.add_edge(2, 0)
            graph.kosaraju()  # Output: [[0, 1, 2]]

        - find_bridges(): Find bridges in a graph.
          Example:
            graph = Graph(4)
            graph.add_edge(0, 1)
            graph.add_edge(1, 2)
            graph.find_bridges()  # Output: [(1, 2)]

        - articulation_points(): Find articulation points in a graph.
          Example:
            graph = Graph(4)
            graph.add_edge(0, 1)
            graph.add_edge(1, 2)
            graph.articulation_points()  # Output: [1]

        - ford_fulkerson(source, sink): Ford-Fulkerson algorithm for maximum flow.
          Example:
            graph = Graph(4)
            graph.add_edge(0, 1)
            graph.add_edge(1, 2)
            graph.ford_fulkerson(0, 2)  # Output: 1
        """
    
    @staticmethod
    def printStructureOnly():
        """
        Returns a string containing the structure of classes and functions in the `data_structures.py` file.
        """
        return """
        # Structure of Classes and Functions in data_structures.py

        ## 1. Arrays Class
        - traverse(arr)
        - insert(arr, index, value)
        - delete(arr, index)
        - prefix_sum(arr)
        - suffix_sum(arr)
        - two_pointers(arr, target)
        - sliding_window(arr, k)
        - kadane(arr)
        - bubble_sort(arr)
        - selection_sort(arr)
        - insertion_sort(arr)
        - merge_sort(arr)
        - quick_sort(arr)
        - counting_sort(arr)
        - radix_sort(arr)
        - linear_search(arr, target)
        - binary_search(arr, target)
        - dutch_national_flag(arr)
        - rotate_array(arr, k)
        - majority_element(arr)
        - merge_intervals(intervals)

        ## 2. Strings Class
        - reverse_string(s)
        - is_palindrome(s)
        - substrings(s)
        - kmp(s, pattern)
        - rabin_karp(s, pattern)
        - z_algorithm(s)
        - rolling_hash(s, base=256, mod=101)
        - lcs(s1, s2)
        - longest_palindromic_substring(s)
        - are_anagrams(s1, s2)
        - run_length_encoding(s)

        ## 3. LinkedList Class
        - append(data)
        - prepend(data)
        - delete(key)
        - reverse()
        - detect_cycle()
        - remove_nth_from_end(n)
        - merge_sorted_lists(l2)
        - find_intersection(l2)
        - clone_with_random_pointers()

        ## 4. Stack Class
        - push(item)
        - pop()
        - top()
        - is_empty()
        - min_stack()

        ## 5. Queue Class
        - enqueue(item)
        - dequeue()
        - is_empty()
        - circular_queue(size)
        - next_greater_element(arr)
        - balanced_parentheses(s)
        - lru_cache(capacity)
        - sliding_window_maximum(arr, k)

        ## 6. Hashing Class
        - hash_table()
        - open_addressing()
        - separate_chaining()
        - count_distinct_elements(arr)
        - longest_consecutive_subsequence(arr)
        - two_sum(arr, target)
        - group_anagrams(strs)

        ## 7. BinaryTree Class
        - inorder_traversal(root)
        - preorder_traversal(root)
        - postorder_traversal(root)
        - level_order_traversal(root)
        - height(root)
        - diameter(root)
        - lowest_common_ancestor(root, p, q)
        - is_balanced(root)
        - is_identical(root1, root2)
        - kth_smallest(root, k)
        - sorted_array_to_bst(arr)

        ## 8. Heaps Class
        - min_heap(arr)
        - max_heap(arr)
        - heap_sort(arr)
        - kth_largest(arr, k)
        - kth_smallest(arr, k)
        - merge_k_sorted_lists(lists)
        - median_of_running_stream()
        - top_k_frequent_elements(arr, k)

        ## 9. Graph Class
        - add_edge(u, v)
        - bfs(start)
        - dfs(start)
        - detect_cycle_undirected()
        - detect_cycle_directed()
        - topological_sort()
        - dijkstra(start)
        - bellman_ford(start)
        - floyd_warshall()
        - kruskal()
        - prim()
        - kosaraju()
        - find_bridges()
        - articulation_points()
        - ford_fulkerson(source, sink)
        """

def printStructure():
        """
        Returns a string in triple quotes containing the `DataStructureExamples` class
        and all its function names.
        """
        return """
        # DataStructureExamples Class and Functions

        ## Class: DataStructureExamples

        ### Functions:
        - arrays_examples()
        - strings_examples()
        - linked_list_examples()
        - stack_examples()
        - queue_examples()
        - hashing_examples()
        - binary_tree_examples()
        - heaps_examples()
        - graph_examples()
        - printStructureOnly()
        """
       

# ------------------------- Main -------------------------

if __name__ == "__main__":
    printAlls()

    '''
# ------------------------- Main -------------------------

if __name__ == "__main__":
    printAlls()
