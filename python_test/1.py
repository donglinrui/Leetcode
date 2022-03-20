def solution(s):
    dp = [0 for _ in range(len(s))]
    dp[0] = 1
    for i in range(1,len(s)):
        if s[i] != '0':
            dp[i] += dp[i-1]
        if int(s[i-1:i+1]) <=26 and s[i-1] !='0':
            if i-2 > 0:
                dp[i] += dp[i-2]
            else:
                dp[i] += 1
    return dp[-1]
if __name__ == '__main__':
    #print(int(5**0.5))
    print(solution('11106'))
