def solution(text1,text2):
    if len(text1) == 0 or len(text2) == 0: return 0
    dp = [[0 for _ in range(len(text1)+1)] for _ in range(len(text2)+1)]
    for i in range(1,len(text2)+1):
        for j in range(1,len(text1)+1):
            if text1[j-1] == text2[i-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j],dp[i][j-1])
    return dp[-1][-1]
if __name__ == '__main__':
    text1 = "abcde"
    text2 = "ace"
    print(solution(text1,text2))
