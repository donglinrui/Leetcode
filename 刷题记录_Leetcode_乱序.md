#### 001 两数之和

> 给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 的那 两个 整数，并返回它们的数组下标。
>
> 你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。
>
> 你可以按任意顺序返回答案。
>

关键词：hash表

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        d = dict()#定义字典的方法有两种，另一种是d = {}
		# 键-值：数字-下标
        for i, num in enumerate(nums):#i是下标，num是值
            if target-num in d:
                return [d[target-num],i]
            d[num]=i
        return []
```

利用hash表来提升遍历的速度，实现只遍历一遍

#### 007 整数反转

> 给你一个 32 位的有符号整数 x ，返回将 x 中的数字部分反转后的结果。如果反转后整数超过 32 位的有符号整数的范围 [−231,  231 − 1] ，就返回 0。假设环境不允许存储 64 位整数（有符号或无符号）。

关键词：Python语法分片、Python类型转换、Python的乘方表示

```python
class Solution:
    def reverse(self, x: int) -> int:
        if x>=0:
            result = (int(str(x)[::-1]))
        else:
            result = -int(str(abs(x))[::-1])
        if result >= -2**31 and result < 2**31-1:
            return result
        else:
            return 0
```

#### 009 回文数

> 给你一个整数 x ，如果 x 是一个回文整数，返回 true ；否则，返回 false 。
>
> 回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。例如，121 是回文，而 123 不是。
>

如果该数字是回文，其后半部分反转后应该与原始数字的前半部分相同。

```python
class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0:#复数不是回文数
            return False
        if int(x/10) == 0:#个位数是回文数
            return True
        if x%10 == 0:#末尾是0则一定不是回文数
            return False
        back = x%10
        front = int(x/10)
        while back <= front:
            if back == front or back == int(front/10):
                return True
            else:
                back = back*10 + front%10
                front = int(front/10)
        return False
```

#### 013 罗马数字转整数

> 罗马数字包含以下七种字符: I， V， X， L，C，D 和 M。
>
> 字符          数值
> I             1
> V             5
> X             10
> L             50
> C             100
> D             500
> M             1000
> 例如， 罗马数字 2 写做 II ，即为两个并列的 1。12 写做 XII ，即为 X + II 。 27 写做  XXVII, 即为 XX + V + II 。
>
> 通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 IIII，而是 IV。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 IX。这个特殊的规则只适用于以下六种情况：
>
> I 可以放在 V (5) 和 X (10) 的左边，来表示 4 和 9。
> X 可以放在 L (50) 和 C (100) 的左边，来表示 40 和 90。 
> C 可以放在 D (500) 和 M (1000) 的左边，来表示 400 和 900。
> 给定一个罗马数字，将其转换成整数。输入确保在 1 到 3999 的范围内。

关键词：hash、罗马数字的读法

罗马数字由 I,V,X,L,C,D,M 构成；
当小值在大值的左边，则减小值，如 IV=5-1=4；
当小值在大值的右边，则加小值，如 VI=5+1=6；
由上可知，右值永远为正，因此最后一位必然为正。

```python
class Solution:
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        a = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}        
        ans=0        
        for i in range(len(s)):            
            if i<len(s)-1 and a[s[i]]<a[s[i+1]]:                
                ans-=a[s[i]]
            else:
                ans+=a[s[i]]
        return ans
```

#### 017 电话号码的字母组合

> 给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。
>
> 给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。。
>

关键词：hash

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        dic = {
        '2':'abc',
        '3':'def',
        '4':'ghi',
        '5':'jkl',
        '6':'mno',
        '7':'pqrs',
        '8':'tuv',
        '9':'wxyz'
        }
        ret = []
        if digits == '':
            return ret
        for d in digits:
            if ret == []:
                for i in dic[d]:
                    ret.append(i)
            else:
                newret = [] 
                for i in ret:
                    for j in dic[d]:
                        newret.append(i+j)
                ret = newret
        return ret
```

#### 021 合并两个有序链表

> 将两个升序链表合并为一个新的 **升序** 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

关键词：链表

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        prev = ListNode(0)#一个头节点，没有实际意义
        head = prev
        while l1 and l2 :#讲l1和l2中小的那个链接到链表上
            if l1.val < l2.val:
                prev.next = l1
                l1 = l1.next
            else:
                prev.next = l2
                l2 = l2.next
            prev = prev.next
        if l1 != None:#剩下的以此排列
            prev.next = l1
        if l2 != None:
            prev.next = l2
        return head.next
```

#### 110 平衡二叉树

> 给定一个二叉树，判断它是否是高度平衡的二叉树。
>
> 本题中，一棵高度平衡二叉树定义为：
>
> ​	一个二叉树*每个节点* 的左右两个子树的高度差的绝对值不超过 1 。

关键词：二叉树、平衡二叉树

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    #利用递归算法计算树的高度
    def height(self,root):
        if root != None:
            rh = self.height(root.right)
            lh = self.height(root.left)
            if rh == -1 or lh == -1:#如果已经找到一个子树不是平衡二叉树，则直接不用计算返回-1
                return -1
            if abs(rh-lh) > 1:#如果子树的左右节点高度差距>1则该子树不是平衡二叉树
                return -1
            thisheight = max(rh,lh) + 1
            return thisheight
        else:
            return 0#如果是一个叶子节点则返回0（高度是0）
    def isBalanced(self, root: TreeNode) -> bool:
        if self.height(root) == -1:
            return False
        else:
            return True
```

#### 118 杨辉三角

> 给定一个非负整数 *numRows，*生成杨辉三角的前 *numRows* 行。
>
> 在杨辉三角中，每个数是它左上方和右上方的数的和。
>
> **示例:**
>
> ```
> 输入: 5
> 输出:
> [
>      [1],
>     [1,1],
>    [1,2,1],
>   [1,3,3,1],
>  [1,4,6,4,1]
> ]
> ```

关键词：数学

```python
class Solution:
    def generate_oneline(self, before , lenth):#生成一行的方法，before是上一行的数组，lenth是本行长度
        ret = [1]*lenth#数组的初始化方式 生成长度为lenth，元素都是1的数组
        if lenth > 2:
            for i in range(1,lenth-1):
                ret[i] = before[i-1] + before[i]
        return ret
        
    def generate(self, numRows: int) -> List[List[int]]:
        ret = []
        for i in range(numRows):
            if i > 0:
                ret.append(self.generate_oneline(ret[i-1],i+1))
            else:
                ret.append([1])
        return ret
```

#### 172 阶乘后的零

> 给定一个整数 *n*，返回 *n*! 结果尾数中零的数量。

关键词：数学

现在，为了确定最后有多少个零，我们应该看有多少对 2 和 5 的因子。在上面的例子中，我们有一对 2 和 55的因子。

首先，我们可以注意到因子 2数总是比因子 5 大。为什么？因为每四个数字算作额外的因子 2，但是只有每 25 个数字算作额外的因子 5。

```python
class Solution:
    #先分析，当1~n中有一个5则尾数会对应出现一个0，利用因式分解法，求5的个数，即为位数中0的个数
        def trailingZeroes(self, n: int) -> int:
            count = 0
            for i in range(1,n+1):
                num = i
                while(num>0):
                    if(num%5 == 0):
                        count += 1 
                        num = int(num/5)
                    else:
                        break
            return count
```

#### 266 翻转二叉树

> 示例：
>
> 输入：
>
>      	  4
>      	/   \
>       2     7
>      / \   / \
>     1   3 6   9
>
> 输出：
>
>     	 4
>         /   \
>       7     2
>      / \   / \
>     9   6 3   1

关键词：二叉树、层次遍历

利用递归算法，遍历二叉树

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if root == None:
            return root
        right = self.invertTree(root.right)
        left = self.invertTree(root.left)
        root.right,root.left = left,right
        return root
```

#### 976 三角形的最大周长

> 给定由一些正数（代表长度）组成的数组 `A`，返回由其中三个长度组成的、**面积不为零**的三角形的最大周长。
>
> 如果不能形成任何面积不为零的三角形，返回 `0`。

关键词：数学

```python
class Solution:
    def largestPerimeter(self, A: List[int]) -> int:
        A.sort(reverse = True)#排序
        for i in range(len(A)-2):
            if A[i] < A[i+1]+A[i+2]:#判断是否能组成三角形
                return A[i]+A[i+1]+A[i+2]
        return 0
```

### 231 2的幂

> 给你一个整数 n，请你判断该整数是否是 2 的幂次方。如果是，返回 true ；否则，返回 false 。
>
> 如果存在一个整数 x 使得 n == $$2^x$$，则认为 n 是 2 的幂次方。
>

```python 
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        if n == 0: return False#特殊情况1
        if n == 1:#递归终止调节，如果结果是1，则可以整除
            return True
        if n % 2 == 1: #如果不能被2整除则返回FALSE
            return False
        return self.isPowerOfTwo(n/2)
```

