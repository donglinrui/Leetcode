def solution(nums):
    if len(nums) < 2: return False #若小于两个元素则不可能分成两个子集
    if sum(nums)%2 != 0: return False #若sum为奇数则不可能分成两个自己
    target = sum(nums) // 2
    #dp[i][j]的含义[0,i]的所有正整数中，是否存在一种选取方案是的被选取的正整数和等于j，若存在则为True
    dp = [[False for _ in range(target+1)] for _ in range(len(nums))]
    #dp为i行，target+1列
    for i in range(len(nums)):
        for j in range(target+1):
            if j == 0:
                dp[i][j] = True#如果选取正整数之和为0，则一定满足条件。
                continue
            if i == 0 and j == nums[0]:
                dp[i][j] = True#当i=0时，只有一个正整数可以被选取。
                continue
            if nums[i] <= j:
                dp[i][j] = dp[i-1][j] or dp[i-1][j-nums[i]]
            elif nums[i] > j:
                dp[i][j] = dp[i-1][j] #不符合条件，不放入
    return dp[-1][-1]

if __name__ == '__main__':
    nums = [1,5,11,5]
    print(solution(nums))
