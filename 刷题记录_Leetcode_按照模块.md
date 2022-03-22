## 贪心算法

### 455 分发饼干

> 假设你是一位很棒的家长，想要给你的孩子们一些小饼干。但是，每个孩子最多只能给一块饼干。
>
> 对每个孩子 `i`，都有一个胃口值 `g[i]`，这是能让孩子们满足胃口的饼干的最小尺寸；并且每块饼干 `j`，都有一个尺寸 `s[j]` 。如果 `s[j] >= g[i]`，我们可以将这个饼干 `j` 分配给孩子 `i` ，这个孩子会得到满足。你的目标是尽可能满足越多数量的孩子，并输出这个最大数值。

因为饥饿度最小的孩子最容易吃饱，所以我们先考虑这个孩子。为了尽量使得剩下的饼干可 以满足饥饿度更大的孩子，所以我们应该把大于等于这个孩子饥饿度的、且大小最小的饼干给这 个孩子。满足了这个孩子之后，我们采取同样的策略，考虑剩下孩子里饥饿度最小的孩子，直到 没有满足条件的饼干存在。

简而言之，这里的贪心策略是，给剩余孩子里最小饥饿度的孩子分配最小的能饱腹的饼干。 至于具体实现，因为我们需要获得大小关系，一个便捷的方法就是把孩子和饼干分别排序。 这样我们就可以从饥饿度最小的孩子和大小最小的饼干出发，计算有多少个对子可以满足条件。

```python
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        #先排序
        g.sort()
        s.sort()
        #两个数组的index
        index_g = 0
        index_s = 0
        count = 0#记录有多少个小孩可以吃饱
        while (index_g < len(g)) and index_s< len(s):
            while g[index_g] > s[index_s]:#找到一个满足条件的饼干，若超出范围则直接返回
                if index_s+1 < len(s):index_s +=1
                else:break
            if g[index_g] <= s[index_s]:count +=1#找到满足条件的饼干后count+1
            #依次向后找
            index_g +=1
            index_s +=1
        return count
```

### 135 分发糖果

> 老师想给孩子们分发糖果，有 N 个孩子站成了一条直线，老师会根据每个孩子的表现，预先给他们评分。
>
> 你需要按照以下要求，帮助老师给这些孩子分发糖果：
>
> 每个孩子至少分配到 1 个糖果。
> 评分更高的孩子必须比他两侧的邻位孩子获得更多的糖果。
> 那么这样下来，老师至少需要准备多少颗糖果呢？

做完了题目 455，你会不会认为存在比较关系的贪心策略一定需要排序或是选择？虽然这一 道题也是运用贪心策略，但我们只需要简单的两次遍历即可：把所有孩子的糖果数初始化为 1； 先从左往右遍历一遍，如果右边孩子的评分比左边的高，则右边孩子的糖果数更新为左边孩子的 糖果数加 1；再从右往左遍历一遍，如果左边孩子的评分比右边的高，且左边孩子当前的糖果数 不大于右边孩子的糖果数，则左边孩子的糖果数更新为右边孩子的糖果数加 1。通过这两次遍历， 分配的糖果就可以满足题目要求了。这里的贪心策略即为，在每次遍历中，只考虑并更新相邻一 侧的大小关系。

```python
class Solution:
    def candy(self, ratings: List[int]) -> int:
        ret_1 = [1]*len(ratings)#从左到右遍历记录糖果数量
        ret_2 = ret_1[:]#从右向左遍历记录糖果数量
        for i in range(1,len(ratings)):
            if ratings[i-1] < ratings[i]:
                ret_1[i] = ret_1[i-1] + 1
        #因为下面的遍历中不会遍历到ret_1[-1],且ret_2[-1]一定为一，所以把ret_1[-1]直接加上
        count = ret_1[-1]
        for i in range(len(ratings)-2,-1,-1):
            if ratings[i+1] < ratings[i]:
                ret_2[i] = ret_2[i+1] + 1
            count += max(ret_1[i],ret_2[i])#左边遍历和右边遍历的最大值为该位孩子的真实获得的糖果数量
        return count
```

### 435 无重叠区间

> 给定一个区间的集合，找到需要移除区间的最小数量，使剩余区间互不重叠。

在选择要保留区间时，区间的结尾十分重要：选择的区间结尾越小，余留给其它区间的空间 就越大，就越能保留更多的区间。因此，我们采取的贪心策略为，优先保留结尾小且不相交的区 间。 具体实现方法为，先把区间按照结尾的大小进行增序排序，每次选择结尾最小且和前一个选 择的区间不重叠的区间。这里我们使用python中的sorted 函数进行排序。right用于记录当前遍历到的右边界。

```python
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        if len(intervals) == 0 : return  0
        intervals_sorted = sorted(intervals,key = lambda x:x[1])
        res = 0
        right = intervals_sorted[0][1]#初始化右边界
        for i in range(1,len(intervals_sorted)):
            if right > intervals_sorted[i][0]:#如果重叠则res+1
                res += 1
            else:
                right = intervals_sorted[i][1]#如果不重叠则更新右边界
        return res 
```

### 605 种花问题

> 假设有一个很长的花坛，一部分地块种植了花，另一部分却没有。可是，花不能种植在相邻的地块上，它们会争夺水源，两者都会死去。
>
> 给你一个整数数组  flowerbed 表示花坛，由若干 0 和 1 组成，其中 0 表示没种植花，1 表示种植了花。另有一个数 n ，能否在不打破种植规则的情况下种入 n 朵花？能则返回 true ，不能则返回 false。
>

先判断三种特殊情况

①n==0 直接返回True

②flowerbed长度为0 直接返回False

③flowerbed长度为1，判断内部元素是不是1，如果是1则返回F，如果是0则返回T

然后遍历花坛：

①第一个花坛

②最后一个花坛

③中间花坛

统计出可以种植的花盆的数量，然后和n对比

```python
class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        if n == 0: return True
        if len(flowerbed) == 0: return False
        max = 0
        if len(flowerbed) == 1:
            max = 1 if flowerbed[0] == 0 else 0
            return True if max >= n else False
        for i in range(len(flowerbed)):
            if flowerbed[i] == 0:
                if i == 0:
                    if flowerbed[i+1] == 0:
                        max += 1
                        flowerbed[i] = 1
                elif i == len(flowerbed)-1:
                    if flowerbed[i-1] == 0:
                        max += 1
                        flowerbed[i] = 1
                else:
                    if flowerbed[i-1] == 0 and flowerbed[i+1] == 0:
                        max += 1
                        flowerbed[i] = 1
        return True if max >= n else False
```

### 452 用最少数量的箭引爆气球

> 在二维空间中有许多球形的气球。对于每个气球，提供的输入是水平方向上，气球直径的开始和结束坐标。由于它是水平的，所以纵坐标并不重要，因此只要知道开始和结束的横坐标就足够了。开始坐标总是小于结束坐标。
>
> 一支弓箭可以沿着 x 轴从不同点完全垂直地射出。在坐标 x 处射出一支箭，若有一个气球的直径的开始和结束坐标为 xstart，xend， 且满足  xstart ≤ x ≤ xend，则该气球会被引爆。可以射出的弓箭的数量没有限制。 弓箭一旦被射出之后，可以无限地前进。我们想找到使得所有气球全部被引爆，所需的弓箭的最小数量。
>
> 给你一个数组 points ，其中 points [i] = [xstart,xend] ，返回引爆所有气球所必须射出的最小弓箭数。
>

```python
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        if len(points) == 0: return 0
        if len(points) == 1: return 1
        points.sort()
        record = points[0]#初始化第一个射箭的区间
        count = 1#需要射箭的次数
        for i in points[1:]:
            if record[1] >= i[0]:#如果可以在当前射箭区间里把当前气球射爆
                record = [max(record[0],i[0]),min(record[1],i[1])]#更新射箭区间（和当前区间去交集）
            else:#如果不能射爆，则需要一个新的箭，射箭区间改为当前区间
                record = i
                count += 1
        return count
```

### 763 划分字母区间

> 字符串 `S` 由小写字母组成。我们要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。返回一个表示每个字符串片段的长度的列表。

1.先 生成一个字典，字典内部是每一个字母对应的

2.在从0~last这个范围里，挨个查其他字母，看他们的最后位置是不是比刚才的last或一段的最后位置大。
如果没有刚才的last或一段的最后位置大，无视它继续往后找。
如果比刚才的大，说明这一段的分隔位置必须往后移动，所以我们把last或一段的最后位置更新为当前的字母的最后位置。

3.肯定到有一个时间，这个`last`就更新不了了，那么这个时候这个位置就是我们的分隔位置。

```python
class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        dic = {letter: index for index, letter in enumerate(s)}
        print(dic)
        last = dic[s[0]]
        num = 0
        ret = []
        for i in range(len(s)):
            num += 1
            if dic[s[i]] > last:
                last = dic[s[i]]
            if i == last:
                ret.append(num)
                num = 0
        return ret
```

### 122 买卖股票的最佳时机II

> 给定一个数组 prices ，其中 prices[i] 是一支给定股票第 i 天的价格。
>
> 设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。
>
> 注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
>

start 和end 记录买卖区间

ret 记录受益

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        start = 0
        end = 0
        ret = 0
        for i in range(len(prices)):
            if prices[end] < prices[i]:#若股票还在增长
                end = i
            else:#股票没在增长
                ret += prices[end] - prices[start]#卖出股票，计算累加收益
                start = i
                end = i
            if i == len(prices) -1:#若已经是最后一只股票，则卖出
                ret += prices[end] - prices[start]
        return ret
```

### 406 根据身高重建队列

同样地，我们也可以将每个人按照身高从大到小进行排序，处理身高相同的人使用的方法类似，即：按照 $$h_i$$
作为第一关键字降序，$$k_i$$作为第二关键字升序进行排序。如果我们按照排完序后的顺序，依次将每个人放入队列中，那么当我们放入第 i 个人时：

- 第 0,...,i−1 个人已经在队列中被安排了位置，他们只要站在第 i 个人的前面，就会对第 i 个人产生影响，因为他们都比第 i 个人高；

- 而第 i+1,...,n−1 个人还没有被放入队列中，并且他们无论站在哪里，对第 i个人都没有任何影响，因为他们都比第 i个人矮。

所以第i个人的位置的下标是$$k_i$$

但我们可以发现，后面的人既然不会对第 i个人造成影响，我们可以采用「插空」的方法，依次给每一个人在当前的队列中选择一个插入的位置。也就是说，当我们放入第 i 个人时，只需要将其插入队列中，使得他的前面恰好有 $$k_i$$个人即可。

```python
class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        people.sort(key = lambda x:(-x[0],x[1]))
        ans = list()
        for person in people:
            ans[person[1]:person[1]] = [person]
        return ans
```

### 665 非递减数列

**贪心算法**

本题是要维持一个非递减的数列，所以遇到递减的情况时（nums[i] > nums[i + 1]），要么将前面的元素缩小，要么将后面的元素放大。

但是本题唯一的易错点就在这，

- 如果将将nums[i]缩小，可能会导致其无法融入前面已经遍历过的非递减子数列；

- 如果将nums[i + 1]放大，可能会导致其后续的继续出现递减；

所以要采取贪心的策略，在遍历时，每次需要看连续的三个元素，也就是瞻前顾后，遵循以下两个原则：

- 需要尽可能不放大nums[i + 1]，这样会让后续非递减更困难；
- 如果缩小nums[i]，但不破坏前面的子序列的非递减性；

**算法步骤：**

遍历数组，如果遇到递减：
还能修改：
修改方案1：将nums[i]缩小至nums[i + 1]；
修改方案2：将nums[i + 1]放大至nums[i]；
不能修改了：直接返回false；。

```python
class Solution:
    def checkPossibility(self, nums: List[int]) -> bool:
        if len(nums) == 1 or len(nums) == 0: return True
        count = 0
        if nums[0] > nums[1] : count +=1#需要先对第一个第二个元素进行判断
        for i in range(1,len(nums)-1):
            if nums[i] > nums[i+1]:
                count +=1
                if count <=1:#若还有修改次数
                    if nums[i-1] > nums[i+1]:#此种情况下需要修改nums[i+1]
                        nums[i+1] = nums[i]
                    else:#nums[i-1] <= nums[i+1]此种情况下需要修改nums[i]
                        nums[i] = nums[i+1]
                else:#没有修改次数
                    return False
        return True
```

## 双指针

### 167 两数之和II-输入有序数组

> 给定一个已按照 升序排列  的整数数组 numbers ，请你从数组中找出两个数满足相加之和等于目标数 target 。
>
> 函数应该以长度为 2 的整数数组的形式返回这两个数的下标值。numbers 的下标 从 1 开始计数 ，所以答案数组应当满足 1 <= answer[0] < answer[1] <= numbers.length 。
>
> 你可以假设每个输入只对应唯一的答案，而且你不可以重复使用相同的元素。
>

因为数组已经排好序，我们可以采用方向相反的双指针来寻找这两个数字，一个初始指向最 小的元素，即数组最左边，向右遍历；一个初始指向最大的元素，即数组最右边，向左遍历。 如果两个指针指向元素的和等于给定值，那么它们就是我们要的结果。如果两个指针指向元 素的和小于给定值，我们把左边的指针右移一位，使得当前的和增加一点。如果两个指针指向元 素的和大于给定值，我们把右边的指针左移一位，使得当前的和减少一点。 可以证明，对于排好序且有解的数组，双指针一定能遍历到最优解。证明方法如下：假设最 优解的两个数的位置分别是 l 和 r。我们假设在左指针在 l 左边的时候，右指针已经移动到了 r； 此时两个指针指向值的和小于给定值，因此左指针会一直右移直到到达 l。同理，如果我们假设 在右指针在 r 右边的时候，左指针已经移动到了 l；此时两个指针指向值的和大于给定值，因此 右指针会一直左移直到到达 r。所以双指针在任何时候都不可能处于 (l,r) 之间，又因为不满足条 件时指针必须移动一个，所以最终一定会收敛在 l 和 r。

```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        index1 = 0#开头指针
        index2 = len(numbers)-1#末尾指针
        while numbers[index1] + numbers[index2] != target:
            if numbers[index1] + numbers[index2] < target:#若小于目标值
                index1 += 1
            if numbers[index1] + numbers[index2] > target:#若大于目标值
                index2 -= 1
        return [index1+1,index2+1]
```

### 88 合并有序数组

> 给你两个有序整数数组 nums1 和 nums2，请你将 nums2 合并到 nums1 中，使 nums1 成为一个有序数组。
>
> 初始化 nums1 和 nums2 的元素数量分别为 m 和 n 。你可以假设 nums1 的空间大小等于 m + n，这样它就有足够的空间保存来自 nums2 的元素。
>



```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        index1 = 0
        index2 = 0
        ret = []
        if len(nums1) == 0:
            for n in nums2:
                nums1.append(n)
                return
        while index1 < m and index2 < n:
            #比较两个元素的大小，较小的加入ret数组
            if nums1[index1] <= nums2[index2]:
                ret.append(nums1[index1])
                if index1+1 <= m:index1 += 1
            else:
                ret.append(nums2[index2])
                if index2+1 <= n:index2 += 1
           #最后将没有比较完的数组依次添加
        if index1 < m:
            for i in range(index1,m):
                ret.append(nums1[i])
        if index2 < n:
            for i in range(index2,n):
                ret.append(nums2[i])
        nums1.clear()
        for n in ret:
            nums1.append(n)
        return
```

### 142 环形链表II

> 给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
>
> 为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。注意，pos 仅仅是用于标识环的情况，并不会作为参数传递到函数中。
>

![image-20210918105226991](E:\Leetcode刷题\刷题记录_Leetcode_按照模块.assets\image-20210918105226991.png)

![image-20210918105257024](E:\Leetcode刷题\刷题记录_Leetcode_按照模块.assets\image-20210918105257024.png)

```python
class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        fast,slow = head,head
        #先让fast和slow各走一步
        if fast == None or fast.next == None or fast.next.next == None:
                return None
        else:
            fast = fast.next.next
        slow = slow.next
        #循环，找到slow和fast第一次相遇的点
        while fast != slow:
            if fast.next == None or fast.next.next == None:
                return None
            else:
                fast = fast.next.next
            slow = slow.next
        #fast指向head，然后再找他们第二次相遇的点
        fast = head
        while fast != slow:
            fast = fast.next
            slow = slow.next
        return slow
```

### 633 平方数之和

> 给定一个非负整数 `c` ，你要判断是否存在两个整数 `a` 和 `b`，使得$$a^2+b^2=c$$。

双指针法，一个指向$$c$$ 一个指向0

```python
class Solution:
    def judgeSquareSum(self, c: int) -> bool:
        a = 0
        b = int(c ** 0.5)+1
        while a<=b:
            if a**2 + b**2 < c:
                a += 1
            elif a**2 + b**2 > c:
                b -= 1
            else:
                return True
        return False
```

### 680 验证回文字符串II

> 给定一个非空字符串 `s`，**最多**删除一个字符。判断是否能成为回文字符串。

双指针+贪心算法

若不满足回文，则判断减少一个字符后是否满足

```python
class Solution:
    def validPalindrome(self, s: str) -> bool:
        def checkPalindrome(low, high):
            i, j = low, high
            while i < j:
                if s[i] != s[j]:
                    return False
                i += 1
                j -= 1
            return True

        low, high = 0, len(s) - 1
        while low < high:
            if s[low] == s[high]: 
                low += 1
                high -= 1
            else:#如果不满足回文，则判断删除low 或者删除high是否满足
                return checkPalindrome(low + 1, high) or checkPalindrome(low, high - 1)
        return True
```

### 524 通过删除字母匹配到字典里最长单词

> 给你一个字符串 `s` 和一个字符串数组 `dictionary` ，找出并返回 `dictionary` 中最长的字符串，该字符串可以通过删除 `s` 中的某些字符得到。
>
> 如果答案不止一个，返回长度最长且字母序最小的字符串。如果答案不存在，则返回空字符串。

关键词：贪心算法、双指针

```python
class Solution:
    def findLongestWord(self, s: str, dictionary: List[str]) -> str:
        def check(pattern,pp,ss):#贪心算法，匹配字符串。ss和pp是两个指针，指向s和pattern
            if pp == len(pattern):return True
            if ss == len(s):return False
            while pp < len(pattern):
                if s[ss] == pattern[pp]:#若相等，则两个指针都后移一位
                    return check(pattern,pp+1,ss+1)
                else:#若不相等，则只后移ss指针
                    return check(pattern,pp,ss+1)
        ret = ''
        for p in dictionary:
            if ret != '':#剪枝，如果下一个dictionary小于当前ret，则剪枝
                if len(ret) > len(p): continue
            if check(p,0,0) and len(p)>=len(ret):#字符串匹配
                if len(p) > len(ret):
                    ret = p
                elif len(p) == len(ret):#长度相等
                    if ret > p:
                        ret = p
        return ret
```

## 二分查找

### 69  x的平方根

> 给你一个非负整数 `x` ，计算并返回 `x` 的 **平方根** 。
>
> 由于返回类型是整数，结果只保留 **整数部分** ，小数部分将被 **舍去 。**

关键词：二分查找

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        low, high,ans = 0,x,-1
        while low <= high:#二分查找
            mid = (low + high) // 2
            if mid **2 <= x:#ans的平方一定是小于x的
                ans = mid
                low = mid+1
            else:
                high = mid-1
        return ans
```

### 34 在排序数组中查找元素的第一个和最后一个位置

> 给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。
>
> 如果数组中不存在目标值 target，返回 [-1, -1]。
>

关键词：二分查找

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if len(nums) == 0: return [-1,-1]
        low = 0
        high = len(nums) - 1
        ans = -1
        #利用二分查找找出元素所在位置
        while low <= high:
            mid = (low + high)//2
            if nums[mid] < target:
                low = mid + 1
            elif nums[mid] > target:
                high = mid-1
            elif nums[mid] == target:
                ans = mid
                break
        if ans == -1: return [-1,-1]
        print(ans)
        #找出第一个元素和最后一个元素的下标
        l = ans
        h = ans
        if l > 0:
            while nums[l-1] == target:
                l -= 1
                if l == 0: break
        if h < len(nums)-1:
            while nums[h+1] == target:
                h += 1
                if h == len(nums) -1:break
        return [l,h]

```

### 81 搜索旋转排序数组

> 已知存在一个按非降序排列的整数数组 nums ，数组中的值不必互不相同。
>
> 在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转 ，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,4,4,5,6,6,7] 在下标 5 处经旋转后可能变为 [4,5,6,6,7,0,1,2,4,4] 。
>
> 给你 旋转后 的数组 nums 和一个整数 target ，请你编写一个函数来判断给定的目标值是否存在于数组中。如果 nums 中存在这个目标值 target ，则返回 true ，否则返回 false 。
>

关键词：二分查找

![image-20210923211003991](E:\Leetcode刷题\刷题记录_Leetcode_按照模块.assets\image-20210923211003991.png)

![image-20210923211047010](E:\Leetcode刷题\刷题记录_Leetcode_按照模块.assets\image-20210923211047010.png)

```python
class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        if len(nums) == 0:return False
        if len(nums) == 1:return nums[0] == target
        low,high = 0,len(nums)-1
        while low <= high:
            mid = (low + high)//2
            if nums[mid] == target: return True #找到了target
            if nums[mid] == nums[low] and nums[mid] == nums[high]:#如果mid、low、和high 对应的元素相等则无法判断那一段是有序数组，则对左右区间进行缩小
                low += 1
                high -= 1
            elif nums[mid] <= nums[high]:#判断一个区间是否为增
                if nums[mid] <= target and target <= nums[high]:#如果是曾且target在nums[mid]和nums[high]之间，则在mid和high之间查找，否则在low和mid之间查找
                    low = mid +1
                else:
                    high = mid - 1
            else:
                if nums[low]<= target  and target < =nums[mid]:
                    high = mid - 1
                else:
                    low = mid + 1
        return False
```

### 154 寻找旋转排序数组中的最小值II

> 已知一个长度为 n 的数组，预先按照升序排列，经由 1 到 n 次 旋转 后，得到输入数组。例如，原数组 nums = [0,1,4,4,5,6,7] 在变化后可能得到：
> 若旋转 4 次，则可以得到 [4,5,6,7,0,1,4]
> 若旋转 7 次，则可以得到 [0,1,4,4,5,6,7]
> 注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。

关键词：二分查找

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        low,high = 0,len(nums)-1
        while low < high:
            pivot = (low + high)//2
            if nums[pivot] < nums[high]:
                high = pivot
            elif nums[pivot] > nums[high]:
                low = pivot+1
            else:
                high -= 1
        return nums[low]
```

### 540 有序数组中的单一元素

> 给定一个只包含整数的有序数组，每个元素都会出现两次，唯有一个数只会出现一次，找出这个数。
>

flag用于判断奇数的数组的位置，重点在于通过mid和flag更新区间

```python
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        low, high = 0,len(nums)-1
        while low < high:
            mid = (low+high)//2
            flag = (high - mid) % 2 == 0#mid~high为偶数
            if nums[mid] == nums[mid+1]:
                if flag:#奇数的数组在后半段
                    low = mid+2
                else:#奇数的数组在前半段
                    high = mid-1
            elif nums[mid] == nums[mid-1]:
                if flag:#奇数的数组在前半段
                    high = mid-2
                else:#奇数的数组在后半段
                    low = mid +1
            else:
                return nums[mid]
        return nums[low]
```

## 排序算法

### 215 数组中第K个最大元素

> 给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。
>
> 请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。
>

方法一：快速排序

练习一下快速排序写法，并不是最优解

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def Quicksort(l,r):
            if l >= r-1:return
            start,end = l,r-1
            #print(start,end)
            key = nums[start]
            while start < end:
                while start < end and nums[end] >= key:
                    end -= 1
                nums[start] = nums[end]
                while start < end and nums[start] <= key:
                    start += 1
                nums[end] = nums[start]
            nums[start] = key
            Quicksort(l,start)
            Quicksort(start+1,r)
            return
        Quicksort(0,len(nums))
        return nums[-k]
```

方法二：快速选择

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def p(l, r):#快速选择部分
            pivot = l
            low = l + 1
            high = r
            while (low <= high):
                while low <= high and nums[low] >= nums[pivot]:
                    low += 1#从左边找到第一个小于nums[pivot]的数
                while low <= high and nums[high] <= nums[pivot]:
                    high -= 1#从右边找到第一个大于nums[pivot]的数
                if low<=high and nums[low]<nums[pivot] and nums[high]>nums[pivot]:
                    nums[low], nums[high] = nums[high], nums[low]
                    low += 1
                    high -= 1
            nums[high], nums[pivot] = nums[pivot], nums[high]
        #这里为什么是high？
        #因为 如果high==low先是while low <= high and nums[low] >= nums[pivot]:这个语句在前，low的值先增加，high为要换的位置的值
        #如果high>low 即high = low + 1 则也是high-1为要换的位置
            return high
        if len(nums) ==0:
            return 0
        left,right = 0,len(nums)-1
        while(True):
            position = p(left,right)
            if position==k-1:return nums[position]
            #二分的思想
            elif position > k-1:right = position-1
            else:left = position +1
```

### 347 前K个高频元素

> 给你一个整数数组 `nums` 和一个整数 `k` ，请你返回其中出现频率前 `k` 高的元素。你可以按 **任意顺序** 返回答案。

利用桶排序

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        bucket1 = {}
        for num in nums:
            bucket1[num] = 1 if num not in bucket1 else bucket1[num] + 1
        return [i[0] for i in sorted(bucket1.items(),key=lambda x:x[1],reverse= True)[:k]]
```

  ### 451 根据字符出现频率排序

> 给定一个字符串，请将字符串里的字符按照出现的频率降序排列。

```python
class Solution:
    def frequencySort(self, s: str) -> str:
        d = {}
        for item in s:
            d[item] = d[item] + 1 if item in d else 1
        sorted_s = sorted(d.items(),key=lambda x:x[1],reverse= True)
        return ''.join([item[0] for item in sorted_s for i in range(item[1])])
```

### 75 颜色分类

> 给定一个包含红色、白色和蓝色，一共 n 个元素的数组，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。
>
> 此题中，我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。
>

我们也可以考虑使用指针 p_0来交换 0，p_2 来交换 2。此时，p_0 的初始值仍然为 00，而 p_2 的初始值为 n-1。在遍历的过程中，我们需要找出所有的 0 交换至数组的头部，并且找出所有的 2 交换至数组的尾部。

![image-20211019153024088](E:\Leetcode刷题\刷题记录_Leetcode_按照模块.assets\image-20211019153024088.png)

```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        p0,p2 = 0,len(nums)-1
        i = 0
        while i <= p2:
            while i<=p2 and nums[i] == 2:
                nums[i],nums[p2] = nums[p2],nums[i]
                p2 -= 1
            if nums[i] == 0:
                nums[i], nums[p0] = nums[p0], nums[i]
                p0 += 1
            i = i + 1
```

## 搜索算法

### 695 岛屿的最大面积

> 给你一个大小为 m x n 的二进制矩阵 grid 。
>
> 岛屿 是由一些相邻的 1 (代表土地) 构成的组合，这里的「相邻」要求两个 1 必须在 水平或者竖直的四个方向上 相邻。你可以假设 grid 的四个边缘都被 0（代表水）包围着。
>
> 岛屿的面积是岛上值为 1 的单元格的数目。
>
> 计算并返回 grid 中最大的岛屿面积。如果没有岛屿，则返回面积为 0 。
>

利用递归

```python
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        def DFS(i,j):#深度优先遍历
            if i == -1 or j == -1 or i == len(grid) or j == len(grid[0]) or grid[i][j] == 0:
                return 0
            grid[i][j] = 0#遍历之后修改当前节点，以保证不重复遍历同一个节点
            ret = 1 
            for n_i,n_j in [[0,1],[1,0],[-1,0],[0,-1]]:#在四个方向上做搜索
                ret += DFS(i+n_i,j+n_j)
            return ret
        ret = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                ret = max(ret,DFS(i,j))
        return ret
```

利用栈

```python
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        ret = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                stack = [[i,j]]
                cur = 0 #当前节点的岛屿大小
                while stack:
                    [cur_i,cur_j] = stack.pop()
                    if  cur_i == -1 or cur_j == -1 or cur_i == len(grid) or cur_j == len(grid[0]) or grid[cur_i][cur_j] == 0 :
                        continue
                    grid[cur_i][cur_j] = 0
                    cur += 1
                    for next_i,next_j in [[0,1],[1,0],[-1,0],[0,-1]]:
                        stack.append([cur_i+next_i,cur_j+next_j])
                ret = max(cur,ret)
        return ret
                
```

### 547 省份数量

> 有 n 个城市，其中一些彼此相连，另一些没有相连。如果城市 a 与城市 b 直接相连，且城市 b 与城市 c 直接相连，那么城市 a 与城市 c 间接相连。
>
> 省份 是一组直接或间接相连的城市，组内不含其他没有相连的城市。
>
> 给你一个 n x n 的矩阵 isConnected ，其中 isConnected [i] [j] = 1 表示第 i 个城市和第 j 个城市直接相连，而 isConnected[i][j] = 0 表示二者不直接相连。
>
> 返回矩阵中 省份 的数量。
>

 

```python
class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        def DFS(i):#记录所有访问过的节点
            visited[i] = True
            for index,value in enumerate(isConnected[i]):
                if value == 1 and i != index and visited[index] == False:
                    DFS(index)#深度优先遍历
        res = 0
        visited = [False]*len(isConnected)
        for i in range(len(isConnected)):
            if visited[i] == False:
                DFS(i)
                res += 1
        return  res
```

### 46 全排列

> 给定一个不含重复数字的数组 `nums` ，返回其 **所有可能的全排列** 。你可以 **按任意顺序** 返回答案。

回溯法

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def back(x):
            if x == lenth:
                res.append(nums[:])
            for i in range(x,lenth):#第i个位置开始，向后重新排列，如果i大于lenth则自动退出
                nums[x],nums[i] = nums[i],nums[x]
                back(x+1)
                nums[x], nums[i] = nums[i], nums[x]
        lenth = len(nums)
        res = []
        back(0)
        return res
```

### 77 组合

> 给定两个整数 `n` 和 `k`，返回范围 `[1, n]` 中所有可能的 `k` 个数的组合。
>
> 你可以按 **任何顺序** 返回答案。



```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        res = []
        ans = []
        def back(s,p):
            if p == 0:
                res.append(ans[:])
                return
            for i in range(s,n+2-p):#第s个位置的取值是[s,n+2-p]
                ans.append(i)
                back(i+1,p-1)#下一个位置的值，一定比本位置值i大1，p-1是为了确定上界
                ans.pop()
        back(1,k)
        return res
```

### 79 单词搜索

> 给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。
>
> 单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。
>

这种二维路径题的关键是

1. 建立一个d( d = [[0,1],[1,0],[0,-1],[-1,0]])
2. 回溯法，确定边界
3. visited 保存遍历过的结点

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        visted = set()
        d = [[0,1],[1,0],[0,-1],[-1,0]]
        def check(i,j,k):
            if board[i][j] != word[k]:
                return False
            if k == len(word)-1:
                return True

            visted.add((i,j))
            result = False
            for ii,jj in d:
                n_i,n_j = i+ii,j+jj
                if 0<=n_i<len(board) and 0<=n_j<len(board[0]):
                    if (n_i,n_j) not in visted:
                        if check(n_i,n_j,k+1):
                            result = True
                            break
            visted.remove((i,j))
            return result
        for i in range(len(board)):
            for j in range(len(board[0])):
                if check(i,j,0):return True
        return False
```

### 51 N皇后

> n 皇后问题 研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。
>
> 给你一个整数 n ，返回所有不同的 n 皇后问题 的解决方案。
>
> 每一种解法包含一个不同的 n 皇后问题 的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。
>

例子：

![image-20211107221420903](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211107221420903.png)

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        #返回数组
        solotions = []
        #记录第n行的Q的位置
        Queens = [-1]*n
        #列，斜线1，斜线2记录
        col = set()
        d1 = set()
        d2 = set()
        def generate_solotion():#用于生成一个solotion
            res = []
            for j in range(n):#共有n行
                row = []
                for i in range(n):#这个循环生成一个行
                    if i == Queens[j]:
                        row.append('Q')
                    else:
                        row.append('.')
                res.append(''.join(row))
            return res
        def back(k):#回溯算法
            if k == n:#k是行号，如果遍历完，则完成了一个solotion
                solotions.append(generate_solotion())
            else:
                for i in range(n):
                    if i not in col and i+k not in d1 and i-k not in d2:#如果符合条件
                        #皇后在第k行的位置是i，并记录
                        Queens[k] = i
                        col.add(i)
                        d1.add(i+k)
                        d2.add(i-k)
                        back(k+1)#下一行
                        #回溯回来后，删除记录
                        col.remove(i)
                        d1.remove(i+k)
                        d2.remove(i-k)
        back(0)
        return solotions
                        
```

### 934 最短的桥

> 在给定的二维二进制数组 A 中，存在两座岛。（岛是由四面相连的 1 形成的一个最大组。）
>
> 现在，我们可以将 0 变为 1，以使两座岛连接起来，变成一座岛。
>
> 返回必须翻转的 0 的最小数目。（可以保证答案至少是 1 。）
>

深度优先遍历找到一个岛屿的，然后进行层次遍历找另一个岛屿，层数为距离

```python
class Solution:
    def shortestBridge(self, grid: List[List[int]]) -> int:
        #初始化一些参数
        H,W = len(grid),len(grid[0])
        d = [[0,1],[1,0],[-1,0],[0,-1]]
        stack = []  # 用于进行层次遍历
        #深度遍历算法，用于把找到的第一个岛屿的1改为2
        def DFS(i,j):
            if i < 0 or j < 0 or i == H or j == W:
                return
            if grid[i][j] == 0 or grid[i][j] == 2:
                return
            grid[i][j] = 2
            stack.append((i,j))
            for ii,jj in d:
                DFS(i+ii,j+jj)
        #把找到的第一个岛屿的1改为2
        flag = False
        for i in range(H):
            if flag: break
            for j in range(W):
                if grid[i][j] == 1:
                    DFS(i,j)
                    flag = True
                    break
        #print(grid)
        #层次遍历算法
        #print(stack)
        level = 0 #记录下遍历层数
        while(len(stack) != 0):
            level += 1
            for _ in range(len(stack)):
                x,y = stack.pop(0)
                for xx,yy in d:
                    n_x,n_y = x+xx,y+yy
                    if 0<=n_x<H and 0<=n_y<W:
                        if grid[n_x][n_y] == 1:#找到了另一个岛屿直接返回
                            return level-1
                        if grid[n_x][n_y] == 0:
                            grid[n_x][n_y] = 2
                            stack.append((n_x,n_y))
                        if grid[n_x][n_y] == 2:
                            continue
            #print(grid)
        return -1
```

### 130 被围绕的区域

> 给你一个 `m x n` 的矩阵 `board` ，由若干字符 `'X'` 和 `'O'` ，找到所有被 `'X'` 围绕的区域，并将这些区域里所有的 `'O'` 用 `'X'` 填充。

被围绕的区间不会存在于边界上，换句话说，任何边界上的 'O' 都不会被填充为 'X'。 任何不在边界上，或不与边界上的 'O' 相连的 'O' 最终都会被填充为 'X'。如果两个元素在水平或垂直方向相邻，则称它们是“相连”的。

```python
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        d = [[1,0],[0,1],[-1,0],[0,-1]]
        set_O = set()#记录不应该被修改的O
        #深度遍历算法找到
        def DFS(i,j):
            print(i,j)
            if i<0 or j<0 or i>=len(board) or j >= len(board[0]) or (i,j) in set_O:
                return
            if board[i][j] == 'X':
                return
            set_O.add((i,j))
            for ii,jj in d:
                n_i,n_j = ii+i,jj+j
                DFS(n_i,n_j)
        #遍历所有边界元素
        for i in range(len(board)):
            DFS(i,0)
            DFS(i,len(board[0])-1)
        for i in range(1,len(board[0])-1):
            DFS(0,i)
            DFS(len(board)-1,i)
        #修改board
        for i in range(len(board)):
            for j in range(len(board[0])):
                if (i,j) not in set_O and board[i][j] == 'O':
                     board[i][j] = 'X'
```

### 257 二叉树的所有路径

> 给你一个二叉树的根节点 `root` ，按 **任意顺序** ，返回所有从根节点到叶子节点的路径。
>
> **叶子节点** 是指没有子节点的节点。

关键词：二叉树

```python
class Solution:
    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        ans = [str(root.val)]
        res = []
        def Back(node:TreeNode):#回溯算法
            if not root: return
            if node.right == None and node.left == None:#若左右皆为空，则为叶子节点返回结果
                res.append('->'.join(ans))
                return
            else:
                if node.left != None:#若左为不为空则遍历左边
                    ans.append(str(node.left.val))
                    Back(node.left)
                    ans.pop()#遍历完之后删除!!注意，这里不能用remove，会导致出错。应该用pop去删除掉最后一个元素
                if node.right != None:#若右边不为空则遍历右边
                    ans.append(str(node.right.val))
                    Back(node.right)
                    ans.pop()
        Back(root)
        return res
```

### 47 全排列II

> 给定一个可包含重复数字的序列 `nums` ，***按任意顺序*** 返回所有不重复的全排列。```

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
            result = []
            lenth = len(nums)
            def Back(x):
                if x == lenth:
                    if nums not in result:
                        result.append(nums[:])
                for i in range(x,lenth):#第i个位置开始，向后重新排列，如果i大于lenth则自动退出
                    nums[x],nums[i] = nums[i],nums[x]
                    Back(x+1)
                    nums[x], nums[i] = nums[i], nums[x]
                return
            Back(0)
            return result
```

### 40 组合总数II

> 给定一个候选人编号的集合 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
>
> candidates 中的每个数字在每个组合中只能使用 一次 。
>

关键词：回溯

```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        result = []
        ans = []
        candidates.sort()
        dic = dict()
        for i in candidates:
            if i not in dic:
                dic[i] = 1
            else:
                dic[i] += 1
        dic.items()
        freq = list(dic.items())
        def DFS(pos,rest):#pos是当前访问位置，rest是还剩多少数值
            nonlocal ans
            if rest == 0:
                result.append(ans[:])
                return
            if pos == len(freq) or rest < freq[pos][0]:
                return
            DFS(pos+1,rest)#若不放进去
            for i in range(freq[pos][1]):#若放进去
                ans.append(freq[pos][0])
                DFS(pos+1,rest-(i+1)*freq[pos][0])
            ans = ans[:-freq[pos][1]]
        DFS(0, target)
        return result
```

## 动态规划

### 70 爬楼梯

> 假设你正在爬楼梯。需要 `n` 阶你才能到达楼顶。
>
> 每次你可以爬 `1` 或 `2` 个台阶。你有多少种不同的方法可以爬到楼顶呢？

利用斐波那契数列阶梯，爬到第n阶楼梯的方法是爬到第n-1阶楼梯和的方法和爬到第n-2阶楼梯的方法之和

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        d1 = 1 
        d2 = 2
        if n <= 2: return n
        else:
            for i in range(n-2):
                d1,d2 = d2,d1+d2
        return d2
```

### 198 打家劫舍

> 你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
>
> 给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。
>

分两种情况，一种是抢劫第i个房屋，则为dp1+nums[i]，另一种是不抢劫这个房屋，则为dp2，然后依次更新即可

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 1: return nums[0]
        if len(nums) == 2: return max(nums[0],nums[1])
        dp1,dp2 = nums[0],max(nums[0],nums[1])
        for i in range(2,len(nums)):
            dp1,dp2 = dp2, max(dp1+nums[i],dp2)
        return dp2            
```

### 413 等差数列划分

> 如果一个数列 至少有三个元素 ，并且任意两个相邻元素之差相同，则称该数列为等差数列。
>
> 例如，[1,3,5,7,9]、[7,7,7,7] 和 [3,-1,-5,-9] 都是等差数列。
> 给你一个整数数组 nums ，返回数组 nums 中所有为等差数组的 子数组 个数。
>
> 子数组 是数组中的一个连续序列。
>

 

```python
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        if len(nums) < 3: return 0
        dp = [0 for i in range(len(nums))]
        for i in range(2,len(nums)):
            if nums[i-1]-nums[i] == nums[i-2]-nums[i-1]:#若为连续的等差数列，则以i为结尾的等差数列个数+1
                dp[i] = dp[i-1] + 1
            else:
                dp[i] = 0#若等差数列不连续了，则以i为结尾的等差数列为0
        return sum(dp)
```

### 64 最小路径和

> 给定一个包含非负整数的 `*m* x *n*` 网格 `grid` ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

二维动态规划

```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        dp = [[0 for i in range(len(grid[0]))]for i in range(len(grid))]
        dp[0][0] = grid[0][0]
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                #处理当遍历到的点在左边界和上边界的情况
                if i == 0 and j == 0: continue
                elif i == 0:dp[i][j] = dp[i][j-1]+grid[i][j]
                elif j == 0:dp[i][j] = dp[i-1][j]+grid[i][j]
                else:
                    dp[i][j] = min(dp[i][j-1],dp[i-1][j])+grid[i][j]
        return dp[-1][-1]
```

### 542 01矩阵

> 给定一个由 0 和 1 组成的矩阵 mat ，请输出一个大小相同的矩阵，其中每一个格子是 mat 中对应位置元素到最近的 0 的距离。
>
> 两个相邻元素间的距离为 1 。
>



```python
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        #记录到0的距离(需要更新的矩阵)
        dist = [[0 for _ in range(len(mat[0]))] for _ in range(len(mat))]
        #0的index队列#需要遍历的队列
        q = [(i,j) for i in range(len(mat)) for j in range(len(mat[0])) if mat[i][j]==0]
        #已经遍历过的坐标
        seen = set(q)
        while q:
            i,j = q.pop(0)
            for ii,jj in [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]:
                if 0 <= ii < len(mat) and 0 <= jj <len(mat[0]) and (ii,jj) not in seen:
                    dist[ii][jj] = dist[i][j] + 1
                    seen.add((ii,jj))
                    q.append((ii,jj))
        return dist
```

### 221 最大正方形

> 在一个由 `'0'` 和 `'1'` 组成的二维矩阵内，找到只包含 `'1'` 的最大正方形，并返回其面积。

可以使用动态规划降低时间复杂度。我们用$ \textit{dp}(i, j)$表示以 (i, j) 为右下角，且只包含 11 的正方形的边长最大值。如果我们能计算出所有$ \textit{dp}(i, j)$ 的值，那么其中的最大值即为矩阵中只包含 11 的正方形的边长最大值，其平方即为最大正方形的面积。 

```python
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        dp = [[0 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
        maxSide = 0#记录最大边长
        for i in range(len(dp)):
            for j in range(len(dp[0])):
                if matrix[i][j] == '1':maxSide = max(maxSide,1)
                if i == 0 or j == 0:
                    dp[i][j] = int(matrix[i][j])
                else:
                    if matrix[i][j] == '1':
                        dp[i][j] = min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])+1
                        maxSide = max(maxSide,dp[i][j])
                    else:
                        dp[i][j] = 0
        return maxSide*maxSide
```

### 279 完全平方数

> 给你一个整数 n ，返回 和为 n 的完全平方数的最少数量 。
>
> 完全平方数 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。
>

动态规划状态转移方程为

dp[i] = min(dp[i-1^2],dp[i-2^2],dp[i-3^2]...)+1 （需要保证下表大于0）

```python
class Solution:
    def numSquares(self, n: int) -> int:
        dp = [0 for _ in range(n+1)]
        dp[0] = 0
        for i in range(1,n+1):
            #状态转移方程
            dp[i] = min([dp[i-j*j] for j in range(1,int(n**0.5)+1) if i-j*j >= 0])+1
        return dp[-1]
```

### 91 解码方式

> 一条包含字母 A-Z 的消息通过以下映射进行了 编码 ：
>
> 'A' -> "1"
> 'B' -> "2"
> ...
> 'Z' -> "26"
> 要 解码 已编码的消息，所有数字必须基于上述映射的方法，反向映射回字母（可能有多种方法）。例如，"11106" 可以映射为：
>
> "AAJF" ，将消息分组为 (1 1 10 6)
> "KJF" ，将消息分组为 (11 10 6)
> 注意，消息不能分组为  (1 11 06) ，因为 "06" 不能映射为 "F" ，这是由于 "6" 和 "06" 在映射中并不等价。
>
> 给你一个只含数字的 非空 字符串 s ，请计算并返回 解码 方法的 总数 。
>
> 题目数据保证答案肯定是一个 32 位 的整数。
>

动态规划

```python
class Solution:
    def numDecodings(self, s: str) -> int:
        if s[0] == '0': return 0
        dp = [0 for _ in range(len(s))]
        dp[0] = 1
        for i in range(1,len(s)):
            if s[i] != '0':#当前位置不为0时，考虑一位数的情况，结果需要加上dp[i-1]
                dp[i] += dp[i-1]
            if int(s[i-1:i+1]) <=26 and s[i-1] !='0':#前一个位置不为0，则可以考虑两位数的情况，若满足两位数条件，则结果需要再加上dp[i-2]
                if i-2 >= 0:#保证i-2为整数
                    dp[i] += dp[i-2]
                else:
                    dp[i] += 1
        return dp[-1]
```

### 139 单词拆分

> 给你一个字符串 `s` 和一个字符串列表 `wordDict` 作为字典。请你判断是否可以利用字典中出现的单词拼接出 `s` 。



```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        dp = [False for _ in range(len(s))]
        for i in range(len(s)):
            flag = False #记录以第i位置为终点是否找到过在字典中的字符串
            for j in range(i,-1,-1):#从i开始向前找
                if s[j:i+1] in wordDict:#如果找到了
                    flag = flag or dp[j-1]#注意这里需要j-1，即为找到字符串开始节点的前一个
                    if j == 0:
                        flag = True
            dp[i] = flag
        return dp[-1]
```

### 300 最长递增子序列

> 给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
>
> 子序列 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。
>

注意返回的结果不一定为dp[-1]，也有可能为max(dp),一定要斟酌下这里

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        dp = [1] + [0]*(len(nums)-1)
        for i in range(1,len(nums)):
            if [dp[j] for j in range(i) if nums[j]<nums[i]] != []:#若nums[0:i]中存在比nums[i]更小的元素，则找出dp[0:i]中的最大值，并+1
                dp[i] = max([dp[j] for j in range(i) if nums[j]<nums[i]]) + 1
            else:
                dp[i] = 1
        print(dp)
        return max(dp)
```

