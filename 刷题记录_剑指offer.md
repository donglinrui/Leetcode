### 03 找出数组中重复的数字

> 在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。

关键词：Hash

利用Hash方法，python中是使用字典这个数据结构来实现hashmap

```python
class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
        hashmap = {}
        for num in nums:
            if num in hashmap:#查找键是否在字典中
                return num
            else:
                hashmap[num] = 1#键为num，值为1
```

Hashmap 是根据键的hashCode值存储数据，大多数情况下可以直接定位到它的值，因而具有很快的访问速度，但遍历顺序却是不确定的。

### 04 二维数组的查找

> 在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个高效的函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

关键词：二分查找

由于给定二维数组句有规律，每行的从左到右递增每列从上到下递增，查询的时候就需要想到使用二分查找法。

```python
class Solution:
    def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
        if matrix == []:
            return False
        if len(matrix[0]) == 0:
            return False
        i = len(matrix) - 1 
        j = 0
        while(i >= 0):
            if matrix[i][0] <= target:#二分查找 
                high = len(matrix[i]) - 1
                low = 0
                while(low<=high):
                    mid = int((high+low)/2)
                    if matrix[i][mid] == target:
                        return True
                    elif matrix[i][mid] < target:
                        low = mid + 1
                    else:
                        high = mid -1
            i -= 1
        return False
```

### 05 替换空格

> 请实现一个函数，把字符串 `s` 中的每个空格替换成"%20"。

关键词：Python库函数

```python
class Solution:
    def replaceSpace(self, s: str) -> str:
        return s.replace(' ','%20')
```

### 06 从头到尾打印链表

> 输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。

关键词：回溯算法、递归、Python数组分片

方法一：回溯法

```python
class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        if head == None:
            return []
        else:
            return self.reversePrint(head.next) + [head.val]#注意这里只能用＋来连接，否则就会出现NoneType 的bug
```

方法二：遍历后再倒序

```python
class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        ret = []#返回的数组
        while(head != None):
            ret.append(head.val)
            head = head.next
        return ret[::-1]
```

### 07 重建二叉树

> 输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。

关键词：递归、二叉树、分支法

前序遍历 ：[ root, [左子树的前序遍历结果], [右子树的前序遍历结果] ]

中序遍历：[ [左子树的中序遍历结果], root [右子树的中序遍历结果] ]

树和子树的中序遍历和前序遍历长度一样

基本思想：先通过前序遍历找到root节点，然后通过中序遍历确定左右子树长度，并且得出左右子树的前序遍历和中序遍历结果。再利用递归分别计算左右子树的root节点

递归终止条件：前序遍历序列和中序遍历序列为空

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        indexlist_inorder = {element:i for i,element in enumerate(inorder)}
        def buildSubTree(index_preorder_start,#preoder数组起始和开始的index
                         index_preorder_end,
                         index_inorder_start,#inorder数组起始和开始的index
                         index_inorder_end):
            if index_preorder_start > index_preorder_end:
                return None
            #找出中序遍历中root点的index
            index_preorder_root = index_preorder_start
            index_inorder_root = indexlist_inorder[preorder[index_preorder_root]]
            #计算左子树和右子树的长度
            len_left = index_inorder_root - index_inorder_start
            len_right = index_inorder_end - index_inorder_root 
            #建立一棵树
            root = TreeNode(preorder[index_preorder_root])
            root.left = buildSubTree(index_preorder_start+1,#左节点
                                     index_preorder_start+len_left,
                                     index_inorder_start,
                                     index_inorder_root-1)
            root.right = buildSubTree(index_preorder_start+len_left+1,#右节点
                                      index_preorder_end,
                                      index_inorder_root+1,
                                      index_inorder_end)
            return root
        n = len(preorder)
        return buildSubTree(0,n-1,0,n-1)
```

### 09 用两个栈实现队列

> 用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead 操作返回 -1 )

关键词：栈、队列

两个栈实现队列，一个栈当作顺序记录、另一个作为辅助。

删除的时候需要先将一个栈pop到辅助栈中，实现倒序，然后辅助栈pop出元素，即为队尾元素。（需要恢复到顺序记录栈中）

增加时候直接加入顺序记录栈中即可

由于python没有stack类型，所以用list模拟

```python
class CQueue:

    def __init__(self):
        self.s1 = []
        self.s2 = []

    def appendTail(self, value: int) -> None:
        self.s1.append(value)
#deleteHead()函数有两种写法，此为第一种，比较直观
	def deleteHead(self) -> int:
        if self.s1 == []:
            return -1
        while self.s1 != []:
            self.s2.append(self.s1.pop())
        ret = self.s2.pop()
        while self.s2 != []:
            self.s1.append(self.s2.pop())
        return ret
    def deleteHead(self) -> int:#优化方法，此处没有将s2的元素，重新恢复到s1中是为了节省时间。def deleteHead(self) -> int:
        if self.s1 == []:
            return -1
        while self.s1 != []:
            self.s2.append(self.s1.pop())
        ret = self.s2.pop()
        while self.s2 != []:
            self.s1.append(self.s2.pop())
        return ret
        if self.s2 != []:
            return self.s2.pop()
        if self.s1 == []:
            return -1
        while self.s1 != []:
            self.s2.append(self.s1.pop())
        return self.s2.pop()
```

### 10-I 斐波那契数列

> 写一个函数，输入 n ，求斐波那契（Fibonacci）数列的第 n 项（即 F(N)）。斐波那契数列的定义如下：
>
> F(0) = 0,   F(1) = 1
> F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
> 斐波那契数列由 0 和 1 开始，之后的斐波那契数就是由之前的两数相加而得出。
>
> 答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。
>

关键词：动态规划

```python
class Solution:
    def fib(self, n: int) -> int:
        if n == 0 or n == 1 : return n
        num1 = 0
        num2 = 1
        for i in range(n-1):
            num1,num2 = num2,num2+num1
        return num2 % (10**9 +7)
```

### 10-II 青蛙跳台阶问题

> 一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。
>
> 答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。
>

关键词：动态规划

#### 方法一 斐波那契数列计算法

跳上n阶台阶有两种可能性：

1.从n-1跳上来

2.从n-2跳上来

于是本问题就变成了斐波那契数列问题

```python
class Solution:
    def numWays(self, n: int) -> int:
        a,b = 1,1
        for i in range(n-1):
            a,b = b,a+b
        return b%(10**9+7)
```

#### 方法二 记忆递归法

```python
class Solution:
    def __init__(self):
        self.record = {}#用于记录
    def numWays(self, n: int) -> int:
        if n == 0:
            self.record[0] = 1
            return 1
        if n == 1:
            self.record[1] = 1
            return 1
        return (self.check(n-1) + self.check(n-2))%(10**9+7)
    def check(self,num):#用于checkrecord中是否已经计算过num个台阶的问题
        if num in self.record:
            return self.record[num]
        else:
            self.record[num] = self.numWays(num)
            return self.numWays(num)
```

### 11 旋转数组的最小值

> 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一个旋转，该数组的最小值为1。  
>

关键词：数组、迭代

```python
class Solution:
    def minArray(self, numbers: List[int]) -> int:
        for i in range(1,len(numbers)):
            if numbers[i-1] > numbers[i]:
                return numbers[i]
        return numbers[0]
```

#### 其他方法：二分检索

关键词：二分检索

**1.初始化：** 声明 i, j 双指针分别指向 nums 数组左右两端；
**2.循环二分：** 设 m = (i + j) / 2 为每次二分的中点（ "/" 代表向下取整除法，因此恒有 $$i \leq m < j$$ ），可分为以下三种情况：
**(1)当 nums[m] > nums[j] 时：** m 一定在 左排序数组 中，即旋转点 x 一定在 $$[m + 1, j]$$ 闭区间内，因此执行 i = m + 1i=m+1；
**(2)当 nums[m] < nums[j]时：** m 一定在 右排序数组 中，即旋转点 x 一定在$$[i, m][i,m] $$闭区间内，因此执行 j = m；
**(3)当 nums[m] = nums[j]nums[m]=nums[j] 时：** 无法判断 mm 在哪个排序数组中，即无法判断旋转点 xx 在 $$[i, m] $$还是 $$[m + 1, j]$$间中。解决方案： 执行 j = j - 1缩小判断范围，分析见下文。 

**3.返回值**： 当 i = ji=j 时跳出二分循环，并返回 旋转点的值 nums[i]nums[i] 即可。

```python
class Solution:
    def minArray(self, numbers: [int]) -> int:
        i, j = 0, len(numbers) - 1
        while i < j:
            m = (i + j) // 2
            if numbers[m] > numbers[j]: i = m + 1
            elif numbers[m] < numbers[j]: j = m
            else: j -= 1
        return numbers[i]
```

### 12 矩阵中的路径

> 请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一格开始，每一步可以在矩阵中向左、右、上、下移动一格。如果一条路径经过了矩阵的某一格，那么该路径不能再次进入该格子。例如，在下面的3×4的矩阵中包含一条字符串“bfce”的路径（路径中的字母用加粗标出）。
>
> [["a","b","c","e"],
> ["s","f","c","s"],
> ["a","d","e","e"]]
>
> 但矩阵中不包含字符串“abfb”的路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入这个格子。
>

关键词：回溯算法，DPS算法、递归

```python
class Solution:
    def _init_(self):
        self.record = [[]]
    def exist_check(self,board,word,i,j):
        self.record[i][j] = 1#将该点设为已经走过
        if len(word) == 0:#如果已经找完了word，则返回True
            return True
        #四个方向找符合条件的点
        #条件有三个，不越界、没有走过的点、符合word的元素
        if i+1 < len(board):
            if board[i+1][j] == word[0] and self.record[i+1][j] == 0:
                r= self.exist_check(board,word[1:],i+1,j)
                if r:
                    return r
        if i-1 >= 0 :
            if board[i-1][j] == word[0] and self.record[i-1][j] == 0:
                r = self.exist_check(board,word[1:],i-1,j)
                if r:
                    return r
        if j+1 < len(board[i]):
            if board[i][j+1] == word[0] and self.record[i][j+1] == 0:
                r = self.exist_check(board,word[1:],i,j+1)
                if r:
                    return r
        if j-1 >= 0:
            if board[i][j-1] == word[0] and self.record[i][j-1] == 0:
                r = self.exist_check(board,word[1:],i,j-1)
                if r:
                    return r
        self.record[i][j] = 0#如果没有找到路径则将该条路线重新设置为没有走过的路线
        return False

    def exist(self, board, word: str) -> bool:
        self.record = [[0 for i in range(len(board[0]))] for i in range(len(board))]
        for i,line in enumerate(board):#遍历找到起点
            for j,element in enumerate(line):
                if element == word[0]:
                    r = self.exist_check(board,word[1:],i,j)
                    if r:return r
        return False
```

###  13 机器人运动范围

> 地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？
>

关键词：深度优先遍历（DFS）

易错点：不是所有满足条件的格子都连通（即：可达）

- **深度优先搜索**： 可以理解为暴力法模拟机器人在矩阵中的所有路径。DFS 通过递归，先朝一个方向搜到底，再回溯至上个节点，沿另一个方向搜索，以此类推。
- **剪枝：** 在搜索中，遇到数位和超出目标值、此元素已访问，则应立即返回，称之为 可行性剪枝 。

**算法解析:**

- **递归参数：** 当前元素在矩阵中的行列索引 i 和 j ，两者的数位和 si, sj 。
- **终止条件：** 当 ① 行列索引越界 或 ② 数位和超出目标值 k 或 ③ 当前元素已访问过 时，返回 00 ，代表不计入可达解。
- **递推工作：**
  - 标记当前单元格 ：将索引 (i, j) 存入 Set visited 中，代表此单元格已被访问过。
  - 搜索下一单元格： 计算当前元素的 下、右 两个方向元素的数位和，并开启下层递归 。
- **回溯返回值：** 返回 1 + 右方搜索的可达解总数 + 下方搜索的可达解总数，代表从本单元格递归搜索的可达解总数。

```python
class Solution:
    def c_sum(self,num):
        sum = 0
        while (num > 0):
            sum += num%10
            num = int(num/10)
        return sum
    def judge(self,i,j,k):
        if self.c_sum(i) + self.c_sum(j) > k:
            return True
        return False
    def movingCount(self, m: int, n: int, k: int) -> int:
        def DFS(i,j):
            if i >= m or j >= n or self.judge(i,j,k) or (i,j) in visited:
                return 0
            else:
                visited.add((i,j))
                return DFS(i+1,j) + DFS(i,j+1) +1
        visited = set()
        return DFS(0,0)
```

### 14 剪绳子

> 给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m-1] 。请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18
>

#### 方法一：动态规划

关键词：动态规划

```python
class Solution:
    def cuttingRope(self, n: int) -> int:
        dp = [0 for i in range(n+1)]#一个数组dp用于记录各个长度的结果
        dp[2] = 1
        for i in range(3,n+1):#dp的值的更新区间
            for j in range(2,n):#不剪j的情况（dp[i]）、剪掉j长度后的情况，分为两种：不剪了(j*(i-j))，剩下i-j还需要剪(j*dp[i-j])
                dp[i] = max(dp[i],max(j*(i-j),j*dp[i-j]))
        return dp[n]
```

#### 方法二：贪心算法

关键词：贪心算法

**两个推论：**

1.将绳子以相等的长度等分为多段 ，得到的乘积最大。（利用均值不等式）

2.尽可能将绳子以长度 3 ，等分为多段时，乘积最大。

![image-20210317205146546](E:\Leetcode刷题\刷题记录_剑指offer.assets\image-20210317205146546.png)

```python
import math
class Solution:
    def cuttingRope(self, n: int) -> int:
        if n == 2:return 1
        if n == 3:return 2
        a = int (n // 3)
        b = int(n%3)
        if b == 0: return int(math.pow(3,a))#当余数为0时，直接求3^n
        if b == 1: return int(math.pow(3,a-1)*4)#当余数为1时，需要把1*3拆分成2*2
        return int(math.pow(3,a)*b)
```

### 15 二进制中1的个数

> 请实现一个函数，输入一个整数（以二进制串形式），输出该数二进制表示中 1 的个数。例如，把 9 表示成二进制是 1001，有 2 位是 1。因此，如果输入 9，则该函数输出 2。
>

#### 方法一： 位运算

关键词：位运算：数学

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        ret = 0
        while n > 0:
            ret += n&1
            n = n >> 1
        return ret
```

#### 方法二：一行代码

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        return str(bin(n)).count('1')
```

### 16 数值的整数次方

> 实现 [pow(*x*, *n*)](https://www.cplusplus.com/reference/valarray/pow/) ，即计算 x 的 n 次幂函数（即，xn）。不得使用库函数，同时不需要考虑大数问题。

关键词：二分法、数学、位运算

![image-20210321211148639](E:\Leetcode刷题\刷题记录_剑指offer.assets\image-20210321211148639.png)

算法流程：
1.当 $$x = 0$$时：直接返回 0 （避免后续 $$x = 1 / x$$ 操作报错）
2.初始化 res = 1
3.当 n < 0 时：把问题转化至$$n \geq0$$ 的范围内，即执行 $$x = 1/x$$ ，$$n = - n$$
4.循环计算：当 n = 0 时跳出；
	1.当 n \& 1 = 1 时：将当前 x 乘入 res（即 $$res\times= x$$）；
	2.执行 $$x = x^2$$
	3.执行 n 右移一位（即 n >>= 1）。
8.返回 res 。

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n == 0:
            return 1
        res = 1
        if n < 0:#如果指数小于零，则做一些处理
            n,x = -n,1/x
        while n > 0:
            if n&1 == 1: res *= x#利用位置运算，如果该位是1，则乘入结果中，如果不是1，则增加幂次
            x *= x
            n = n >> 1
        return res 
```

### 17 打印从1到最大的n位数

> 输入数字 `n`，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3 一直到最大的 3 位数 999。

#### 方法一： 深度优先遍历

关键词：深度优先、递归

基于生产大数的全排列优化

![image-20210321220923076](E:\Leetcode刷题\刷题记录_剑指offer.assets\image-20210321220923076.png)

优化方面：

1. **删除左边多余的0**

   ​	1.左边界定义num[start:]	

   ​	2.左边界的变化规律：当所有位数均为9时，start需要变化，因此引入新的变量nine，当nine=n-start时，需要进行左边界变化。

   	3. 统计nine方法，当添加9到字符串中时候+1 回溯结束-1

2. **列表从1开始**

在以上方法的基础上，添加数字字符串前判断其是否为 `"0"` ，若为 `"0"` 则直接跳过。

```python
class Solution:
    def printNumbers(self, n: int) -> List[int]:
        def DFS(x):
            if n == x:
                s = ''.join(self.record[self.start:])
                if s != '0': self.res.append(int(s))
                #判断是否所有数字均为9，如果均为9则最左边元素位置需要改变
                if n-self.start == self.nine: self.start -= 1
                return
            for i in range(10):
                if i == 9 : self.nine += 1
                self.record[x] = str(i)
                DFS(x + 1)
            self.nine -=1
        self.start = n-1#记录最左边的元素位置
        self.nine = 0#记录9的个数
        self.record = [0]*n
        self.res = []
        DFS(0)
        return self.res
```

##### Note：理解深度优先遍历

本题是从 数位的最高位开始向下遍历，从个位开始计算从0-9，可以好好理解下其中蕴含的深度优先思想

#### 方法二：一行代码

```python
class Solution:
    def printNumbers(self, n: int) -> List[int]:
        return list(range(1,10**n))
```

### 18 删除链表的节点

> 给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。
>
> 返回删除后的链表的头节点。

关键词：链表

利用双指针解决

```python
class Solution:
    def deleteNode(self, head: ListNode, val: int) -> ListNode:
        if head.val == val:
            return head.next
        thisone = head.next
        lastone = head
        while (thisone != None):
            if thisone.val == val:
                lastone.next = thisone.next
                return head
            thisone = thisone.next
            lastone = lastone.next
        return head
```

### 19 正则表达式匹配

> 请实现一个函数用来匹配包含'. '和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（含0次）。在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但与"aa.a"和"ab*a"均不匹配。
>

关键词： 动态规划

##### Note： 理解状态转移方差

我们解递归的时候是需要基本情况的。就算是采用自底向上的办法，也需要定义边界条件是什么。状态转移方程只是让我们知道第 i 阶段的状态和决策后就可以得到第 i+1 阶段的状态（反之同理），它把各个阶段的状态给“串”起来了，但是只有我们知道了边界条件后才可以递推得到最后阶段的最优解。

动态规划问题的关键是在于寻找状态转移方程，本题目的状态转移方程如下：

![image-20210328105953913](E:\Leetcode刷题\刷题记录_剑指offer.assets\image-20210328105953913.png)

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m,n = len(s)+1,len(p)+1
        #状态转移矩阵的意思是第i-1个元素和第j-1个元素,可以通过正则匹配
        dp = [[False]*n for _ in range(m)]
        dp[0][0] = True
        for j in range(2,n,2):
            dp[0][j] = dp[0][j-2] and p[j-1] =='*'
        for i in range(1,m):
            for j in range(1,n):
                if p[j-1] == '*':
                    if dp[i][j-2]:#p[j-2]字符串出现0次
                        dp[i][j] = True
                    elif dp[i-1][j] and s[i-1] == p[j-2]:#p[j-2]字符串多出现一次
                        dp[i][j] = True
                    elif dp[i-1][j] and p[j-2] == '.':#让'.'多出现一次
                        dp[i][j] = True
                else:
                    if dp[i-1][j-1] and s[i-1]== p[j-1]:#上一个匹配且这一个字符相同
                        dp[i][j] = True
                    elif dp[i-1][j-1] and p[j-1] == '.':#是'.'的情况
                        dp[i][j] = True
        return dp[-1][-1]
```

### 20 表示数值的字符串

> 请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串"+100"、"5e2"、"-123"、"3.1416"、"-1E-16"、"0123"都表示数值，但"12e"、"1a3.14"、"1.2.3"、"+-5"及"12e+5.4"都不是。
>

关键词：有限状态自动机

##### Note：有限状态自动机

[(33条消息) 字符串匹配算法之：有限状态自动机_tyler_download的专栏-CSDN博客_有限状态自动机](https://blog.csdn.net/tyler_download/article/details/52549315)

转移图：

![image-20210329114257373](E:\Leetcode刷题\刷题记录_剑指offer.assets\image-20210329114257373.png)

```python
class Solution:
    def isNumber(self, s: str) -> bool:
        states = [
            { ' ': 0, 's': 1, 'd': 2, '.': 4 }, # 0. start with 'blank'
            { 'd': 2, '.': 4 } ,                # 1. 'sign' before 'e'
            { 'd': 2, '.': 3, 'e': 5, ' ': 8 }, # 2. 'digit' before 'dot'
            { 'd': 3, 'e': 5, ' ': 8 },         # 3. 'digit' after 'dot'
            { 'd': 3 },                         # 4. 'digit' after 'dot' (‘blank’ before 'dot')
            { 's': 6, 'd': 7 },                 # 5. 'e'
            { 'd': 7 },                         # 6. 'sign' after 'e'
            { 'd': 7, ' ': 8 },                 # 7. 'digit' after 'e'
            { ' ': 8 }                          # 8. end with 'blank'
        ]
        p = 0                           # start with state 0
        for c in s:
            if '0' <= c <= '9': t = 'd' # digit
            elif c in "+-": t = 's'     # sign
            elif c in "eE": t = 'e'     # e or E
            elif c in ". ": t = c       # dot, blank
            else: t = '?'               # unknown
            if t not in states[p]: return False
            p = states[p][t]
        return p in (2, 3, 7, 8)
```

### 21 调整数组顺序使奇数位于偶数前面

> 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分。

关键词：双指针

```python
class Solution:
    def exchange(self, nums):
        n = 0
        m = len(nums)-1
        while (n<m and m>=0 and n <len(nums)):
            while nums[n] %2 == 1:#找出左数第一个奇数
                n += 1
                if n >= len(nums):
                    break
            while nums[m] %2 == 0:#找出右数第一个偶数
                m -= 1
                if m <0:
                    break
            if(n<m):#如果符合条件，则交换
                nums[n],nums[m] = nums[m],nums[n]
        return nums
```

### 22 链表中倒数第k个节点

> 输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。
>
> 例如，一个链表有 6 个节点，从头节点开始，它们的值依次是 1、2、3、4、5、6。这个链表的倒数第 3 个节点是值为 4 的节点。
>

关键词：双指针

```python
class Solution:
    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
        ret = head
     	#不考虑k会越界的情况
        while(k):
            k -= 1
            head = head.next
        while(head!= None):
            ret = ret.next
            head = head.next
        return ret
```

### 23 反转链表

> 定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。

关键词：链表

#### 方法一：迭代

```python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if head == None:#判断特殊情况
            return None
        nodethis = head#先处理头节点
        nodenext = head.next
        head.next = None
        while nodenext!=None:#依次向后处理，反转节点的方向
            nn = nodenext.next
            nodenext.next = nodethis
            nodethis = nodenext
            nodenext = nn
        return nodethis
```

#### 方法二：递归

```python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        def res(pre,cur):
            if cur == None: return pre#递归终止条件，已经找到pre为最后一个节点
            head = res(cur,cur.next)#求出最后一个节点
            cur.next = pre#反转
            return head
        return res(None,head
```

### 25 合并两个排序的链表

> 输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。

关键词：链表

```python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if l1 == None:
            return l2
        if l2 == None:
            return l1
        if l1 == None and l2 == None:
            return None
        head = l1 if l1.val <= l2.val else l2#找出头节点
        temp = head
        if l1.val <= l2.val:
            l1 = l1.next
        else:
            l2 = l2.next
        while l1!= None and l2 != None:
            if l1.val <= l2.val:
                temp.next = l1
                l1 = l1.next
            else:
                temp.next = l2
                l2 = l2.next
            temp = temp.next
        temp.next = l1 if l1 != None else l2
        return head
```

### 26 树的子结构

> 输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)
>
> B是A的子结构， 即 A中有出现和B相同的结构和节点值。

关键词：二叉树、非递归二叉树遍历

先找出可能的匹配的节点，然后再进行匹配

```python
class Solution:
    def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:
        if A == None or B == None:#特殊情况a、b均为空
            return False
        def match(a,b):#进行匹配
            if b == None: return True#若b先结束，则匹配成功
            if a == None: return False#若b没结束，a却先结束，则匹配失败
            if match(a.left,b.left) and match(a.right,b.right):
                if a.val == b.val:
                    return True
                else:
                    return False
            return False
        stack = []#迭代遍历二叉树，找出合乎规范的节点，并进行match函数
        stack.append(A)
        while(stack):
            t = stack.pop()
            print(t.val)
            if t.val == B.val:
                if match(t,B) == True:return True
            if t.left != None:
                stack.append(t.left)
            if t.right != None:
                stack.append(t.right)
        return False
```

### 27 二叉树的镜像

> 请完成一个函数，输入一个二叉树，该函数输出它的镜像。

关键词：二叉树遍历

#### 方法一：迭代

```python
class Solution:
    def mirrorTree(self, root: TreeNode) -> TreeNode:
        if root == None:#注意考虑特殊情况
            return None
        stack = [root]#利用栈遍历节点
        while(stack!= []):
            node = stack.pop()
            if node.left:stack.append(node.left)
            if node.right:stack.append(node.right)
            node.right,node.left = node.left,node.right
        return root
```

#### 方法二：递归

```python
class Solution:
    def mirrorTree(self, root: TreeNode) -> TreeNode:
        if root == None:
            return None
        if root.left == None and root.right == None:
            return root
        right = self.mirrorTree(root.right)
        left = self.mirrorTree(root.left)
        root.right,root.left = left,right
        return root
```

### 28 对称的二叉树

> 请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。

关键词：二叉树

```python
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return True
        def recur(L,R):
            if not R and not L:return True
            if not R or not L or R.val != L.val: return False
            return recur(L.left,R.right) and recur(L.right,R.left)
        return recur(root.left,root.right)
```

### 29 顺时针打印数组

> 输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。

关键词：数组

```python
class Solution:
    def spiralOrder(self, matrix):
        if not matrix:return []
        left,right,up,down = 0,len(matrix[0])-1,0,len(matrix)-1
        res = []
        while up <= down or left <= right:#必须加上等于的状态，因为可能会有仅有一行或仅有一列没有遍历的情况
            if left <= right:#判断是否越界
                for i in range(left , right+1):#从左向右遍历一行
                    res.append(matrix[up][i])
                up += 1
            else:break
            if up <= down :#判断是否越界
                for i in range(up,down+1):#从上到下遍历一列
                    res.append(matrix[i][right])
                right -= 1
            else:break
            if left <= right:#判断是否越界
                for i in range(right,left-1,-1):#从右到左遍历一列
                    res.append(matrix[down][i])
                down -= 1
            else:break
            if up <= down:#判断是否越界
                for i in range(down,up-1,-1):#从下到上遍历一列
                    res.append(matrix[i][left])
                left += 1
            else:break
        return res
```

### 30 包含min函数的栈

> 定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)。

关键词：栈

辅助栈保存最小的元素，这样就算pop()之后也可以快速定位最小元素

```python
class MinStack:
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.A = []#模拟栈
        self.B = []#辅助栈

    def push(self, x: int) -> None:
        self.A.append(x)
        if self.B == []  or x <= self.B[-1]:
            self.B.append(x)

    def pop(self) -> None:
        x = self.A.pop()
        if x == self.B[-1]:
            self.B.pop()

    def top(self) -> int:
        return self.A[-1]

    def min(self) -> int:
        return self.B[-1]

```



### 31 栈的压入、弹出序列

> 输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如，序列 {1,2,3,4,5} 是某栈的压栈序列，序列 {4,5,3,2,1} 是该压栈序列对应的一个弹出序列，但 {4,3,5,1,2} 就不可能是该压栈序列的弹出序列。
>

关键词：栈、模拟

```python
class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        stack  = []#辅助栈，模拟栈的弹出
        x = 0#用于记录poped的位置
        for i in pushed:
            stack.append(i)
            while stack and stack[-1] == popped[x]:#当栈不为空，且栈顶元素等于本次要弹出的元素时
                stack.pop()
                x += 1
        return not stack
```

### 32-I 从上到下打印二叉树I

> 从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。

关键词：二叉树、层次遍历

```python
class Solution:
    def levelOrder(self, root: TreeNode):
        if root == None:
            return []
        stack = []#控制层次遍历的list
        stack.append(root)
        ret = []#记录返回值的list
        ret.append(root.val)
        while stack:
            t = stack.pop(0)
            if t.left:
                stack.append(t.left)
                ret.append(t.left.val)
            if t.right:
                stack.append(t.right)
                ret.append(t.right.val)
        return ret
```

### 32-II 从上到下打印二叉树II

> 从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。

关键词：二叉树、层次遍历

```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        queue = [root]
        ret = [[root.val]]
        while queue:
            tq = []#辅助队列记录下一层元素
            tret = []#辅助list记录
            while queue:#遍历该层
                t = queue.pop(0)
                if t.left:
                    tret.append(t.left.val)
                    tq.append(t.left)
                if t.right:
                    tret.append(t.right.val)
                    tq.append(t.right)
            queue =tq.copy()
            if tret:ret.append(tret)
        return ret
```

### 32-III 从上到下打印二叉树III

> 请实现一个函数按照之字形顺序打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右到左的顺序打印，第三行再按照从左到右的顺序打印，其他行以此类推。

关键词：二叉树、层次遍历

```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:return []
        queue = [root]
        ret = []
        while queue:
            tq,tret= [],[]
            while queue:
                t = queue.pop(0)
                if len(ret) %2 == 0:tret.append(t.val)#判断奇偶
                else:tret.insert(0,t.val)
                if t.left:tq.append(t.left)
                if t.right:tq.append(t.right)
            queue = tq.copy()
            ret.append(tret)
        return(ret)
```

### 33 二叉搜索树的后续遍历序列

> 输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回 `true`，否则返回 `false`。假设输入的数组的任意两个数字都互不相同。

关键词：二叉搜索树、后序遍历

本题解题点在于不断找根节点，然后进行递归

后序遍历的特点：最后一个元素一定是根节点，方便确定根节点

```python
class Solution:
    def verifyPostorder(self, postorder: List[int]) -> bool:
        def recur(i,j):
            if i >= j:
                return True
            #找出从前到后第一个大于根节点的元素
            m = i
            while postorder[m] < postorder[j]: m += 1
            #如果是二叉搜索树的话，该元素之后的元素即为树的右子树，且右子树中不能有小于跟节点的节点
            if m < j:
                if min(postorder[m:j]) < postorder[j]:
                    return False
            return recur(i,m-1) and recur(m,j-1)#循环判断
        return recur(0,len(postorder)-1)
```

### 34 二叉树中和为某一值的路径

> 输入一棵二叉树和一个整数，打印出二叉树中节点值的和为输入整数的所有路径。从树的根节点开始往下一直到叶节点所经过的节点形成一条路径。

关键词：深度优先遍历、二叉树

```python
class Solution:
    def pathSum(self, root: TreeNode, target: int) -> List[List[int]]:
        def DFS(root,route):
            route.append(root.val)
            if root.left == None and root.right == None:
                print(route)
                if sum(route) == target:
                    res.append(route.copy()) 
                return
            if root.left:
                DFS(root.left,route)
                if route:route.pop()
            if root.right:
                DFS(root.right,route)
                if route:route.pop()
            return
        if root == None and target == 0:return []
        if root == None:return []
        route = []
        res = []
        DFS(root,[])
        return res
```



### 35 复杂链表的复制

> 请实现 copyRandomList 函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个 next 指针指向下一个节点，还有一个 random 指针指向链表中的任意节点或者 null。
>

#### 方法一：有向图深度优先遍历

关键词：深度优先遍历、图

```python
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        def DFS(head):
            if not head:return None
            if head in visited:return visited[head]#如果节点已经被创建，则直接返回节点地址
            copy = Node(head.val,None,None)
            visited[head] = copy
            copy.next = DFS(head.next)#深度遍历next	
            copy.random = DFS(head.random)#深度遍历random
            return copy
        visited = {}#visited 记录访问过的元素
        return DFS(head)
```

#### 方法二：Hash表法

关键词：Hash表

```python
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head: return None
        dic = {}
        cur = head
        while cur:#先依照next顺序遍历一遍，并生成所有Node,然后再把其加入hash表中（key=原链表，value=现链表）
            dic[cur] = Node(cur.val,None,None)
            cur = cur.next
        cur = head
        while cur:
            if cur.next:dic[cur].next = dic[cur.next]
            if cur.random:dic[cur].random = dic[cur.random]
            cur = cur.next
        return dic[head]
```

### 36 二叉搜索树与双向链表

> 输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的循环双向链表。要求不能创建任何新的节点，只能调整树中节点指针的指向。

关键词：二叉搜索树、链表

本题我的想法是先中序遍历一遍二叉树，然后再进行链表的构建，其实这两步是可以同时进行而不互相影响的

```python
class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        def DFS(r):#中序遍历
            if not r: return
            DFS(r.left)
            dic.append(r)#保存到list中
            DFS(r.right)
            return
        dic = []
        DFS(root)
        if len(dic) == 0:return
        #构建链表
        if len(dic) == 1:
            dic[0].left =dic[0]
            dic[0].right = dic[0]
        for i in range(len(dic)-1):
            dic[i].right = dic[i+1]
            dic[i+1].left = dic[i]
        dic[0].left = dic[-1]
        dic[-1].right = dic[0]
        head = dic[0]
        return head
```

改进版的代码如下：

```python
class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        def dfs(cur):
            if not cur: return
            dfs(cur.left) # 递归左子树
            if self.pre: # 修改节点引用
                self.pre.right, cur.left = cur, self.pre
            else: # 记录头节点
                self.head = cur
            self.pre = cur # 保存 cur
            dfs(cur.right) # 递归右子树
        
        if not root: return
        self.pre = None
        dfs(root)
        self.head.left, self.pre.right = self.pre, self.head
        return self.head
```

### 37 序列化二叉树

> 请实现两个函数，分别用来序列化和反序列化二叉树。

关键词：二叉树

```python
class Codec:

    def serialize(self, root):#层次遍历二叉树，唯一变的地方是None也需要插入二叉树
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        if not root:return []
        stack = [root]
        res = [root.val]
        while stack:
            t = stack.pop(0)
            if t.left:
                stack.append(t.left)
                res.append(t.left.val)
            else:
                res.append(None)
            if t.right:
                stack.append(t.right)
                res.append(t.right.val)
            else:
                res.append(None)
        return res
            

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        :type data: str
        :rtype: TreeNode
        """
        #比较 用i来标记目前data的位置，queue来存储本行的Node
        if not data: return
        root = TreeNode(data[0])
        queue = [root]
        i = 1
        while queue:
            t = queue.pop(0)
            if data[i] != None:
                t.left = TreeNode(data[i])
                queue.append(t.left)
            i += 1
            if data[i] != None:
                t.right = TreeNode(data[i])
                queue.append(t.right)
            i += 1        
        return root
```

### 34 字符串排列

> 输入一个字符串，打印出该字符串中字符的所有排列。
>
>  
>
> 你可以以任意顺序返回这个字符串数组，但里面不能有重复元素。

关键词：回溯法

根据字符串排列的特点，考虑深度优先搜索所有排列方案。即通过字符交换，先固定第 1位字符（ n 种情况）、再固定第 2 位字符（ n-1 种情况）、... 、最后固定第 n 位字符（ 1 种情况）处。

```python
class Solution:
    def permutation(self, s: str) -> List[str]:
        list_s,res = list(s),[]
        def DFS(x):
            if x+1 == len(list_s):
                res.append(''.join(list_s))
                return
            s = set()#记录是否换过
            for i in range(x,len(list_s)):
                if list_s[i] in s:continue
                s.add(list_s[i])
                #交换的意思是把第x位置上放上第i个元素
                list_s[i],list_s[x] = list_s[x],list_s[i]#当i==x时候不交换
                DFS(x+1)
                list_s[x], list_s[i] = list_s[i], list_s[x]
        DFS(0)
        return res
```

### 39 数组中出现次数超过一半的数字

> 数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
>
>  
>
> 你可以假设数组是非空的，并且给定的数组总是存在多数元素。

#### 方法一：hash表

关键词：hash表

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        dic = {}#记录出现次数的hash表
        for i in nums:
            if i not in dic:
                dic[i] = 1
            else:
                dic[i] +=1
            if dic[i] > len(nums)/2:
                return i
        return
```

### 40 最小的k个数

> 输入整数数组 `arr` ，找出其中最小的 `k` 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。

关键词：快速排序

```python
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        def quickSort(l,r):
            if l < r:
                i,j = l,r
                while i<j:
                    #从右向左找出比哨兵节点小的元素的下标
                    while i<j and arr[l] <= arr[j]:j -= 1
                    #从左向右找出比哨兵节点大的元素的下标
                    while i<j and arr[l] >= arr[i]:i += 1
                    #交换两个元素
                    arr[i],arr[j] = arr[j],arr[i]
                #交换基准数和中间节点
                arr[i],arr[l] = arr[l],arr[i]
                quickSort(l,i-1)
                quickSort(i+1,r)
        quickSort(0,len(arr)-1)
        return arr[:k]
```

### 41 数据流中的中位数

> 如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。
>
> 例如，
>
> [2,3,4] 的中位数是 3
>
> [2,3] 的中位数是 (2 + 3) / 2 = 2.5
>
> 设计一个支持以下两种操作的数据结构：
>
> - void addNum(int num) - 从数据流中添加一个整数到数据结构中。
> - double findMedian() - 返回目前所有元素的中位数。

关键词：堆，中位数

堆就是用数组实现的二叉树，所以它没有使用父指针或者子指针。堆根据“堆属性”来排序，“堆属性”决定了树中节点的位置。

- 堆的常用方法：

  构建优先队列
  支持堆排序
  快速找出一个集合中的最小值（或者最大值）
  在朋友面前装逼

- 实现方法：
  1.每插入一个数之前，先判断两个堆的 size() 是否相等。
  2.若相等，先将这个数插入大顶堆，然后将大顶堆的 top() 插入小顶堆。这么做可以保证小顶堆的所有数永3.远大于等于大顶堆的 top()。
  4.若不相等，先将这个数插入小顶堆，然后将小顶堆的 top() 插入大顶堆。这么做可以保证大顶堆的所有数永远小于等于小顶堆的 top()。
  5.整个过程我们都动态地做到了平衡两个堆的 size()，即保证它们的 size() 最大只相差了 1。


```python
from heapq import *
class MedianFinder:

    def __init__(self):
        """
        initialize your data structure here.
        """
        #Python中heapq模块是小顶堆。实现大顶堆方法：小顶堆的插入和弹出操作均将元素取反即可。
        self.A = []#小顶堆
        self.B = []#大顶堆

    def addNum(self, num: int) -> None:
        #要保持A的元素比B多
        if len(self.A)!=len(self.B):#不相等的情况需要向B添加元素，保证下一次长度相等
            #先向A加入元素，然后将A顶端元素加入B
            heappush(self.A,num)
            heappush(self.B,-heappop(self.A))
        else:#相等的情况需要向A添加元素
            heappush(self.B,-num)
            heappush(self.A,-heappop(self.B))

    def findMedian(self) -> float:
        #若两个堆长度相等则返回平均数，若不等则返回A[0]
        if len(self.A) == len(self.B):
            return (self.A[0]-self.B[0])/2.0#注意这里B是负数
        else:
            return self.A[0]
```

### 42 连续子数组最大和

> 输入一个整型数组，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。
>
> 要求时间复杂度为O(n)。

关键词：动态规划

dp [i-1]用于记录nums[i-1]以前的数组最大值。

计算dp[i]的方法：若先前元素的连续子数组最大值+当前num[i]比num[i]本身还要小，则证明，前面的连续子数组并不是最优解，需要从num[i]开始从新寻找子数组，则dp[i] = nums[i]

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        dp = [0]*len(nums)
        dp[0] = nums[0]
        for i in range(1,len(nums)):
            dp[i] = max(dp[i-1]+nums[i],nums[i])
        return max(dp)
```

### 43 1~n整数中1出现的次数

> 输入一个整数 n ，求1～n这n个整数的十进制表示中1出现的次数。
>
> 例如，输入12，1～12这些整数中包含1 的数字有1、10、11和12，1一共出现了5次。
>

关键词：脑筋急转弯，数学

当 **cur ==0 时：**出现1的次数

计算公式：$$high\times digit$$

![image-20210428114723127](E:\Leetcode刷题\刷题记录_剑指offer.assets\image-20210428114723127.png)

当 **cur==1时**：出现1的次数

计算公式：$$ high \times digit +low+1$$

![image-20210428114830096](E:\Leetcode刷题\刷题记录_剑指offer.assets\image-20210428114830096.png)

当 **cur == 2,3,4,...9时**，

计算公式：$$(high+1)\times digit$$

![image-20210428114947670](E:\Leetcode刷题\刷题记录_剑指offer.assets\image-20210428114947670.png)

```python
class Solution:
    def countDigitOne(self, n: int) -> int:
        digit = 1
        high = n//10
        cur = n%10
        low = 0
        res = 0
        while cur != 0 or high != 0:#当cur和high均为0的时候就代表结束循环
            if cur == 0: res += high*digit
            elif cur == 1: res += high*digit+low+1
            else:res += (high+1)*digit
            low += cur*digit
            cur = high % 10
            high //= 10
            digit *= 10
        return res
```

### 44 数字序列中的某一位数字

> 数字以0123456789101112131415…的格式序列化到一个字符序列中。在这个序列中，第5位（从下标0开始计数）是5，第13位是1，第19位是4，等等。
>
> 请写一个函数，求任意第n位对应的数字。
>

关键词：脑经急转弯，数学

确定 n 所在 数字 的 位数 ，记为 digit ；
确定 n 所在的 数字 ，记为 num ；
确定 n 是 num 中的哪一数位，并返回结果。

##### 1. 确定所求数位的所在数字的位数

![image-20210428142957577](E:\Leetcode刷题\刷题记录_剑指offer.assets\image-20210428142957577.png)

##### 2. 确定所求数位所在的数字

![image-20210428143037940](E:\Leetcode刷题\刷题记录_剑指offer.assets\image-20210428143037940.png)

##### 3. 确定所求数位在 num的哪一数位

![image-20210428143102713](E:\Leetcode刷题\刷题记录_剑指offer.assets\image-20210428143102713.png)

```python
class Solution:
    def findNthDigit(self, n: int) -> int:
        start,digit,count = 1,1,9
        while n > count:
            n -= count
            start *= 10
            digit += 1
            count = 9*start*digit #9*start是这个范围内数字的个数，*digit是总数位的个数
        num = start + (n-1)/digit #(n-1)/digit可以算出是start后第几位
        return int(str(num)[(n-1)%digit])
```

### 45 把数组排成最小的数

> 输入一个非负整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。

#### 方法一：快速排序

关键词：快速排序

此题求拼接起来的最小数字，本质上是一个排序问题。设数组 nums 中任意两数字的字符串为 x 和 y ，则规定 排序判断规则 为：

若拼接字符串 x + y > y + x  ，则 x “大于” y；
反之，若 x + y < y + x  ，则 x “小于” y ；

```python
class Solution:
    def minNumber(self, nums) -> str:
        def quick_sort(l,r):
            if l >= r: return
            i,j = l,r
            while i < j:
                #重点：此种快速排序j的循环一定要在i的前面，因为i是基准元素
                while i < j and nums[l]+nums[j] <= nums[j]+nums[l]:j -= 1
                while i < j and nums[l]+nums[i] >= nums[i]+nums[l]:i += 1
                nums[i],nums[j] = nums[j],nums[i]
            print(nums)
            nums[i],nums[l] = nums[l],nums[i]
            quick_sort(l,i-1)
            quick_sort(i+1,r)
        nums = [str(num) for num in nums]
        quick_sort(0,len(nums)-1)
        return ''.join(nums)

```



#### 方法二：内置函数

关键词：内置函数

利用内置函数比对大小，重新定义大小规则

```python
class Solution:
    def minNumber(self, nums: List[int]) -> str:
        def sort_rule(x, y):
            a, b = x + y, y + x
            if a > b: return 1
            elif a < b: return -1
            else: return 0
        
        strs = [str(num) for num in nums]
        strs.sort(key = functools.cmp_to_key(sort_rule))#注意内置函数用法
        return ''.join(strs)

```

### 46 把数字翻译成字符

> 给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。一个数字可能有多个翻译。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。
>

关键词：动态规划

根据题意，可按照下图的思路，总结出 “递推公式” （即转移方程）。
因此，此题可用动态规划解决，以下按照流程解题。

![image-20210429131721968](E:\Leetcode刷题\刷题记录_剑指offer.assets\image-20210429131721968.png)

```python
class Solution:
    def translateNum(self, num: int) -> int:
        num = str(num)
        dp = [0]*(len(num)+1)
        dp[0] = dp[1] = 1#这里dp[0]是从第0个元素开始
        for i in range(2,len(dp)):
            dp[i] = dp[i-1] + dp[i-2] if '10' <= num[i-2:i] <= '25' else dp[i-1]
        return dp[len(num)]
```

![image-20210429131816411](E:\Leetcode刷题\刷题记录_剑指offer.assets\image-20210429131816411.png)

dp和num的对应关系如图，注意有偏移量1！

### 47 礼物的最大价值

> 在一个 m*n 的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于 0）。你可以从棋盘的左上角开始拿格子里的礼物，并每次向右或者向下移动一格、直到到达棋盘的右下角。给定一个棋盘及其上面的礼物的价值，请计算你最多能拿到多少价值的礼物？
>

关键词：动态规划

动态规划的关键是填完dp矩阵，所以就利用迭代的方式把dp矩阵填写完毕，最后输出矩阵的\[-1][-1]就可以得出正确的结果

![image-20210505120234319](E:\Leetcode刷题\刷题记录_剑指offer.assets\image-20210505120234319.png)

```python
class Solution:
    def maxValue(self, grid: List[List[int]]) -> int:
        dp = [[0 for i in range(len(grid[0]))] for i in range(len(grid))]
        dp[0][0] = grid[0][0]
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if i == 0 and j == 0: continue 
                if i == 0 and j > 0:
                    dp[i][j] = dp[i][j-1] + grid[i][j]
                if i > 0 and j == 0:
                    dp[i][j] = dp[i-1][j] + grid[i][j]
                if i > 0 and j > 0:
                    dp[i][j] = max(dp[i-1][j],dp[i][j-1]) + grid[i][j]
        print(dp)
        return dp[-1][-1]
```

### 48 最长不含重复字符的字符串

> 请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。

#### 方法一：东林睿动态规划

关键词：动态规划，hash表

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if len(s)==0:return 0
        dp = [1 for i in range(len(s))]
        record = {}#hash表记录每个元素最后出现的位置
        start = 0#指针，记录非重复节点的起点
        for i,item in enumerate(s):
            if item not in record:#如果第一次出现
                record[item] = i#记入hash表
                if i>0:dp[i] = dp[i-1] + 1#dp值更新
            else:#如果已经出现过
                if start < record[item]:start = record[item]#更改start值
                dp[i] = i - start#更改dp值
                record[item] = i#更新hash表
        return max(dp)
```

####  方法二：双指针

关键词：hash表

![image-20210506212125320](E:\Leetcode刷题\刷题记录_剑指offer.assets\image-20210506212125320.png)

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        record,start,res = {},-1,0
        for i in range(len(s)):
            if s[i] in record:#如果s[i]已经出现过，则更换start
                start = max(record[s[i]],start)
            record[s[i]] = i#更新hash表
            res = max(res,i-start) #更新res
        return res
```

### 49 丑数

> 我们把只包含质因子 2、3 和 5 的数称作丑数（Ugly Number）。求按从小到大的顺序的第 n 个丑数。

关键词：数学

a,b,c记录了小于$$x_n$$的，且满足以下情况的最小值（首个乘以2，3，5大于$$x_n$$的丑数）

![image-20210506220804517](E:\Leetcode刷题\刷题记录_剑指offer.assets\image-20210506220804517.png)

```python
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        dp = [1]*n
        a,b,c = 0,0,0
        for i in range(1,n):
            n2,n3,n5 = dp[a]*2,dp[b]*3,dp[c]*5
            dp[i] = min(n2,n3,n5)#第i位元素等于n2，n3，n5中的最小值
            #如果与n2，n3，n5相等，则需要将下标+1
            if dp[i] == n2: a +=1
            if dp[i] == n3: b +=1
            if dp[i] == n5: c +=1
        return dp[-1]
```

### 50 第一个只出现一次的字符

> 在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。 s 只包含小写字母。

关键词：hash表

利用有序hash表，python中字典为有序

```python
class Solution:
    def firstUniqChar(self, s: str) -> str:
        dic = {}
        for i in range(len(s)):#记录一遍每个字母出现的次数
            if s[i] not in dic: dic[s[i]] = 0
            dic[s[i]] +=1
        for key,value in dic.items():#找到第一个满足条件的字符
            if value == 1:
                return key
        return ' '
```

