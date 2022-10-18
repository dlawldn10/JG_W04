#복습하면서 코드 다시 구현해보기
# #2748
# n = int(input())
# dp = [0]*(n+1)
# dp[1] = 1
# dp[2] = 1

# for i in range(3, n+1):
#     dp[i] = dp[i-1] + dp[i-2]

# print(dp[n])


# #1904
# # #이렇게 마지막에 나눠주면 수가 너무 커져서 메모리 초과가 나므로
# # #아래 처럼 dp테이블에 넣을 때 15746으로 나눈 것을 넣어주면 메모리 초과를 막을 수 있다.
# # n = int(input())
# # dp = [0]*(10**6+1)
# # dp[1] = 1
# # dp[2] = 2

# # for i in range(3, n+1):
# #     dp[i] = dp[i-1] + dp[i-2]

# # print(dp[n]%15746)

#이렇게 했더니 100%에서 인덱스 에러 발생
# #dp테이블 크기->dp[2]까지 초기화 불가능하여 오류.
# n = int(input())
# dp = [0]*(n+1)
# dp[1] = 1
# dp[2] = 2

# for i in range(3, n+1):
#     dp[i] = (dp[i-1] + dp[i-2])%15746

# print(dp[n])


# #최종 코드
# n = int(input())
# dp = [0]*(10**6+1)
# dp[1] = 1
# dp[2] = 2

# for i in range(3, n+1):
#     dp[i] = (dp[i-1] + dp[i-2])%15746

# print(dp[n])


#9084
# for _ in range(int(input())):
#     n = int(input())
#     coins = list(map(int, input().split()))
#     m = int(input())
#     dp = [0]*(m+1)
#     dp[0] = 1

#     for k in coins:
#         for i in range(m+1):
#             if i >= k:
#                 dp[i] += dp[i-k]

#     print(dp[m])


# #9251
# w1 = input()
# w2 = input()
# dp = [[0]*(len(w2)+1) for _ in range(len(w1)+1)]
# # print(dp)

# for i in range(1, len(w1)+1):
#     for j in range(1, len(w2)+1):
#         if w1[i-1] == w2[j-1]:
#             dp[i][j] = dp[i-1][j-1] + 1
#         else:
#             dp[i][j] = max(dp[i-1][j], dp[i][j-1])

# print(dp[-1][-1])


# #12865
# import sys
# input = sys.stdin.readline
# N, K = map(int, input().split())
# dp = [0]*(K+1)
# products = []

# for _ in range(1, N+1):
#     W, V = map(int, input().split())
#     products.append((W, V))

# for w, v in products:
#     for i in range(K, 0, -1):
#         if w <= i:
#             dp[i] = max(dp[i], dp[i-w] + v)

# print(dp[-1])


# #11053번과 비슷한 문제
# #11055
# N = int(input())
# A = list(map(int, input().split()))
# dp = [0]*N
# dp[0] = A[0]
# for i in range(1, N):
#     for j in range(i):
#         if A[j] < A[i]:
#             dp[i] = max(dp[i], dp[j])
    
#     dp[i] += A[i]

# print(max(dp))

