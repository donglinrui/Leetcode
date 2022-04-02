def solution(coins,amount):
    dp = [amount + 2] * (amount + 1)
    for i in range(amount + 1):
        for coin in coins:
            if i >= coin:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    print(dp)
    return dp[-1]
if __name__ == '__main__':
    coins = [1,2,5]
    amount = 11
    print(solution(coins,amount))
