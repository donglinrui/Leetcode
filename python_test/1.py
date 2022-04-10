def solution(s,p):
    m, n = len(s), len(p)
    def match(i, j):  # match函数用于判断地s[i-1]和p[j-1]是否匹配
        if i == 0:
            return False
        if p[j - 1] == '.':
            return True
        return s[i - 1] == p[j - 1]

    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    for i in range(m + 1):
        for j in range(1, n + 1):  # 当p为空，且s不为空时，一定匹配失败
            if p[j - 1] == '*':
                dp[i][j] = [i][j - 2]  # 匹配0个p[j-1]元素时
                if match(i, j - 1):
                    dp[i][j] = dp[i - 1][j] 
            else:
                if match(i, j):
                    dp[i][j] = dp[i - 1][j - 1]
    print(dp)
    return dp[-1][-1]

if __name__ == '__main__':
    s = "aa"
    p = "a*"
    print(solution(s,p))
