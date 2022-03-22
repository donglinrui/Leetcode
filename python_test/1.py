def solution(nums):
    dp = [1] + [0]*(len(nums)-1)
    for i in range(1,len(nums)):
        if [dp[j] for j in range(i) if nums[j]<nums[i]] != []:
            dp[i] = max([dp[j] for j in range(i) if nums[j]<nums[i]]) + 1
        else:
            dp[i] = 1
    print(dp)
    return dp[-1]
if __name__ == '__main__':
    nums = [10,9,2,5,3,7,101,18]
    print(solution( [0,1,0,3,2,3]))
