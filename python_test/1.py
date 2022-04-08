def solution(n):
    if n < 2: return 0
    dp = [0]*2 + [n+1]*(n-1)
    for i in range(2,n+1):
        dp[i] = i
        for j in range(2,int(n**0.5)):#由于 i 肯定同时拥有因数 j 和 i/j,两者必有一个小于sqrt(n)
            if i%j == 0:
                dp[i] = min(dp[j]+i//j,dp[i//j]+j,dp[i])
                break
    return dp[-1]
if __name__ == '__main__':
    n = 3
    print(solution(n))
