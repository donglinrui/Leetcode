import scipy.io
def rob(nums):
    def fun(start,end):
        ll = end - start
        if ll == 2: return max(nums[start],nums[end-1])#如果有两个元素（不可能有两个以上的元素的情况出现）
        dp = [0]*ll
        dp[0] = nums[start]
        dp[1] = max(nums[start],nums[start+1])
        for i in range(2,ll):
            dp[i] = max(dp[i-1],dp[i-2]+nums[start+i])
        return dp[-1]
    l = len(nums)
    if l == 1:
        return nums[0]
    elif l == 2:
        return max(nums[0],nums[1])
    else:
        return max(fun(0,l-1),fun(1,l))#由于第一家和最后一家不可以同时偷盗，所以要把最终结果分为这两种情况考虑
if __name__ == '__main__':
    nums = [1,2,3,1]
    print(rob(nums))