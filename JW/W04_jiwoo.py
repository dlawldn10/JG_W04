# #다이나믹 프로그래밍(Dynamic Programming) 동적 계획법
# #메모리를 적절히 사용하여 수행 시간 효율성을 비약적으로 향상시키는 방법
# #이미 계산된 결과를 별도의 메모리 영역에 저장해놨다가 나중에 해당 결과가 필요할대 메뫼 영역에 기록된 정보를 그대로 사용한다.
# #한번 해결한 문제는 다시 해결하지 않도록하는 것.
# #완탐일때 시간 복잡도를 효과적으로 줄일 수 있다.

# #탑다운 -> 하향식
# #          구현과정에서 재귀함수를 이용함.
# #바텀업 -> 상향식
# #          반복문을 이용하여 구현함.
# #          결과 저장용 리스트 = dp 테이블
# #          dp의 전형적인 형태는 바텀업 방식임.

# #메모이제이션은 dp에 국한된 개념은 아님. 메모이제이션은 기법!
# #또한, dp에 활용하지 않을 수도 있다.

# #자료구조에서 동적 할당 : 프로그램이 실행되는 도중에 필요한 메모리를 할당하는 기법
# #다이나믹 프로그래밍 : 별다른 의미 없음.

# #다이나믹 프로그래밍을 사용할 수 있는 문제의 조건
# #1. 최적 부분 구조(Optimal Substructure)
# #   큰 문제를 작은 문제로 나눌 수 있으며 작은 문제의 답을 모아서 큰 문제를 해결할 수 있다.
# #2. 중복되는 부분 문제(Overlaping Subproblem)
# #   동일한 작은 문제를 반복적으로 해결해야한다.

# #대표적인 문제
# #피보나치 수열 문제.


# #메모이제이션(Memoization)
# #한번 계산된 결과를 메모리 공간에 메모하는 기법.
# #같은 문제를 호출하면 메모했던 결과를 그대로 가져온다.
# #값을 게록해놓는다는 점에서 캐싱(caching)이라고도 함.

# d = [0]*100

# #탑다운 방식
# def fibo(x):
#     if x == 1 or x == 2:
#         return 1

#     if d[x] != 0:
#         return d[x]

#     d[x] = fibo(x-1) + fibo(x-2)

#     return d[x]

# print(fibo(99))


# #바텀업 방식
# d[1] = 1
# d[2] = 1
# n = 99

# for i in range(3, n+1):
#     d[i] = d[i-1] + d[i-2]

# print(d[n])


# #다이나믹 프로그래밍 VS 분할정복
# #공통점: 두개 모두 최적 부분 구조를 가질 때 사용할 수 있다.
# #즉, 큰 문제를 작은 문제로 나눌 수 있으며 작은 문제의 답을 모아서 큰 문제를 해결할 수 있는 상황.

# #차이점: 부분 문제의 중복
# #dp에서는 각 부분 문제들이 서로 영향을 미치며 부분 문제가 반복적으로 일어난다.
# #분할 정복 문제에서는 동일한 부분 문제가 반복적으로 일어나지 않는다.

# #분할정복 ex) 퀵정렬
# #한번 분할이 이루어지고 나면 피벗은 다른 부분 문제에 포함되지 않고 위치가 더이상 변경되지 않기 때문에 
# #동일한 부분 문제가 반복적으로 일어나지 않는다고 볼 수 있다.
# #즉 분할 이후에 해당 피벗을 다시 처리하는 부분 문제는 호출하지 않는다.


# #dp문제에 접근하는 방법:
# #주어진 문제가 dp유형임을 파악하는 것이 첫번째.
# #그리디, 구현, 완탐 등의 아이디어로 문제를 해결할 수 있는지 검토.
# #다른 알고리즘 풀이 방법이 떠오르지 않으면 dp를 고려해본다.
# #일단 재귀함수로 완탐 프로그램을 작성한 뒤에, 작은 문제에서 구한 답이 큰 문제에서 그대로 사용될 수 있다면
# #메모이제이션 기법을 추가하여 코드를 개선하는 방법을 사용할 수 있다.
# #일반적인 코테 수준에서는 기본 유형의 dp프로그래밍 문제가 출제되는 경우가 많다. 
# #면접의 경우 쉬운 난이도로 알려진 문제 나오는 경우 있음. 
# #점화식 구하는거 때문에.
# #난이도 제한 없이 매우 어렵게 출제될 수도 있음.


# #2748
# n = int(input())
# d = [0]*(n+1)

# def fibo(x):

#     if x == 1 or x == 2:
#         return 1

#     if d[x] != 0:
#         return d[x]

#     d[x] = fibo(x-1) + fibo(x-2)
#     return d[x]

# print(fibo(n))


#1904
#재귀로 푸니 메모리 초과 났음
# import sys
# sys.setrecursionlimit(10**6)
# input = sys.stdin.readline
# n = int(input())
# d = [0]*(n+1)

# def fibo(x):

#     if x == 1:
#         return 1
#     elif x == 2:
#         return 2

#     if d[x] != 0:
#         return d[x]

#     d[x] = (fibo(x-1) + fibo(x-2))%15746
#     return d[x]

# print(fibo(n))


# #정답 코드
# import sys
# input = sys.stdin.readline

# n = int(input())
# dp = [0] * 1000001
# dp[1] = 1
# dp[2] = 2

# for k in range(3,n+1):
#     dp[k] = (dp[k-1]+ dp[k-2])%15746
# print(dp[n])


# #9084
# import sys

# input = sys.stdin.readline

# t = int(input())
# for _ in range(t):
#     n = int(input())
#     coins = list(map(int, input().split()))
#     m = int(input())

#     # memoization을 위한 리스트 선언
#     d = [0] * (m + 1)
#     d[0] = 1


#     for coin in coins:
#         for i in range(m + 1):
#             # a_(i-k) 를 만드는 방법이 존재한다면 
#             # 이전 경우의 수에 현재 동전으로 만들 수 있는 경우의 수를 더한다.
#             if i >= coin:
#                 d[i] = d[i] + d[i - coin]

#     print(d[m])


#9251
# #시간초과
# from itertools import combinations
# import sys
# input = sys.stdin.readline
# str1 = input().rstrip()
# str2 = input().rstrip()
# res1 = []
# res2 = []

# for i in range(len(str1), 0, -1):
#     for set1 in combinations(str1, i):
#         res1.append((''.join(set1)))
# print(res1)

# for i in range(len(str2), 0, -1):
#     for set2 in combinations(str2, i):
#         res2.append((''.join(set2)))
# print(res2)

# for l in res1:
#     if l in res2:
#         print(len(l))
#         break


#정답코드
# import sys
# read = sys.stdin.readline

# word1, word2 = read().strip(), read().strip()
# h, w = len(word1), len(word2)
# cache = [[0] * (w+1) for _ in range(h+1)]

# for i in range(1, h+1):
#     for j in range(1, w+1):
#         if word1[i-1] == word2[j-1]:
#             cache[i][j] = cache[i-1][j-1] + 1
#         else:
#             cache[i][j] = max(cache[i][j-1], cache[i-1][j])
# print(cache[-1][-1])


# # #12865
# import sys
# input = sys.stdin.readline


# # 가방싸기 함수
# def knapsack(N,K,items):
#     dp = [[0]*(K+1) for _ in range(N+1)]

#     # 가방에 담을 수 있는 물건의 개수를 1개부터 하나씩 늘려 나간다
#     for i in range(1,N+1): # i: item
#         weight, value = map(int, items[i-1])
#         # 가방에 담을 수 있는 최대 무게를 1부터 차례대로 증가시켜 나가면서
#         for j in range(1,K+1): # j:가방에 담을 수 있는 무게
#             # 현재 물건이 가방이 담을 수 있는 무게보다 작으면
#             if weight <= j:
#                 # 현재 물건을 넣지 않았을 때와 현재 물건을 넣었을 때의 가치를 비교한다.
#                 dp[i][j] = max(dp[i-1][j],dp[i-1][j-weight]+value)
#             # 크면 이 물건을 담지 않고 이전 물건까지 담았을 때 가방에 담을 수 있는 최고 가치를 저장
#             else:
#                 dp[i][j] = dp[i-1][j]

#     # 가방에 담을 수 있는 최대 무게에서 모든 물건을 고려했을 때의 최대값을 출력
#     print(dp[N][K])



# # N: 물건 개수 K:가방에 담을 수 있는 최대 무게
# N, K = map(int, input().split())
# # 각 물건의 무게와 가치
# items = [list(map(int, input().split())) for _ in range(N)] 
# # 주어진 조건으로 가방싸기!
# knapsack(N,K,items)


#11049
# #pypy로만 통과
# import sys
# input = sys.stdin.readline

# N = int(input())
# matrix = [list(map(int, input().split())) for _ in range(N)]
# DP = [[0]*N for _ in range(N)]

# # 분할된 그룹의 크기를 1부터 N-1까지 돎
# for size in range(1, N):
# 	# 크기 size인 그룹의 모든 경우의 수 돎
#     for start in range(N - size):
#         end = start + size
        
#         # 어떤 그룹의 최소 곱셈 횟수는 분할한 두 그룹의 최소 곱셈 횟수 + 각 그룹의 곱셈 다 끝나고 남은 행렬끼리의 곱셈 횟수
#         result = float("inf")
#         for cut in range(start, end):
#             result = min(result, DP[start][cut] + DP[cut+1][end] + matrix[start][0]*matrix[cut][1]*matrix[end][1])
#         DP[start][end] = result

# print(DP[0][-1])



# #11053
# N = int(input())
# A = [0] + list(map(int, input().split()))
# dp = [0]*(N+1)
# #dp에 dp[n]이 마지막 요소로 있는 가장 긴 수열의 길이를 저장해 줄 것임.
 
# for i in range(1, N+1):
#     for j in range(1, N+1):
#         #나보다 작은 요소를 발견하면
#         #그 뒤에 나를 얹어줄 것임
#         if A[i]>A[j] and dp[i]<dp[j]:
#             #그 마지막 요소가 있는 수열의 길이를 복사해오고
#             dp[i] = dp[j]
#     #나를 얹어준다.
#     dp[i] += 1

# print(max(dp))


#11047
# N, K = map(int, input().split())
# coins = list(int(input()) for _ in range(N))
# coins.reverse()

# ans = 0
# for coin in coins:
#     ans += K // coin
#     K %= coin
# print(ans)


# #for문을 뒤로 돌린 코드
# N, K = map(int, input().split())
# coins = list(int(input()) for _ in range(N))

# ans = 0
# for i in range(N-1, -1, -1):
#     ans += K // coins[i]
#     K %= coins[i]
# print(ans)


#1541
# s = input().split('-')

# sum = 0
# for i in s[0].split('+'):
#     sum += int(i)

# for i in s[1:]:
#     for j in i.split('+'):
#         sum -= int(j)

# print(sum)


# #1931
# #활동 선택 문제
# import sys
# input = sys.stdin.readline
# N = int(input())
# m = []
# for _ in range(N):
#     start, end = map(int, input().split())
#     m.append((end, start))

# m.sort()

# res = 0
# # tmp = []
# time = 0
# for end, start in m:
    
#     if time <= start:
#         res += 1
#         # tmp.append((start, end))
#         time = end

# # print(tmp)
# print(res)



# #1946
# import sys
# input = sys.stdin.readline
# T = int(input())
# for _ in range(T):
#     N = int(input())
#     scores = [list(tuple(map(int, input().split()))) for _ in range(N)]
#     scores.sort()

#     top = 0
#     result = 1

#     for i in range(1, N):
#         if scores[i][1] < scores[top][1]:
#             top = i
#             result += 1

#     print(result)


# #1700
# N, K = map(int, input().split())
# order = list(map(int, input().split()))

# multi = [0] * N
# count = 0
# scheduling_idx = 0
# tmp = 0
# tmp_i = 0

# for i in order:
#     # 멀티탭에 같은 전기용품이 있을 때
#     if i in multi:
#         pass
#     # 멀티탭이 아직 채워지지 않았을 때
#     elif 0 in multi:
#         multi[multi.index(0)] = i
#     # 멀티탭에 빈자리 없고 현재 꽂혀 있는 전기용품들과 다를 때
#     else:
#         for j in multi:
#             # 현재 꽂혀있는 전기용품이 더 이상 사용되지 않는다면
#             if j not in order[scheduling_idx:]:
#                 tmp = j
#                 break
#             #현재 꽂혀있는 전기용품이 이후에도 사용될 때
#             elif order[scheduling_idx:].index(j) > tmp_i:  # 꽂혀있는 것들 중 여러 개가 다시 사용될 때, 더 나중에 사용되는 것을 뽑는다.
#                 tmp = j
#                 tmp_i = order[scheduling_idx:].index(j)
#         multi[multi.index(tmp)] = i
#         tmp = tmp_i = 0
#         count += 1
#     scheduling_idx += 1

# print(count)


# #2098
# n = int(input())

# INF = int(1e9)
# dp = [[INF] * (1 << n) for _ in range(n)]

# def dfs(x, visited):
#     if visited == (1 << n) - 1:     # 모든 도시를 방문했다면
#         if graph[x][0]:             # 출발점으로 가는 경로가 있을 때
#             return graph[x][0]
#         else:                       # 출발점으로 가는 경로가 없을 때
#             return INF

#     if dp[x][visited] != INF:       # 이미 최소비용이 계산되어 있다면
#         return dp[x][visited]

#     for i in range(1, n):           # 모든 도시를 탐방
#         if not graph[x][i]:         # 가는 경로가 없다면 skip
#             continue
#         if visited & (1 << i):      # 이미 방문한 도시라면 skip
#             continue

#         # 점화식 부분(위 설명 참고)
#         dp[x][visited] = min(dp[x][visited], dfs(i, visited | (1 << i)) + graph[x][i])
#     return dp[x][visited]


# graph = []
# for i in range(n):
#     graph.append(list(map(int, input().split())))

# print(dfs(0, 1))


# #2253
# from sys import stdin

# N, stone_n = map(int, stdin.readline().split())

# stone = set()
# for _ in range(stone_n):
#     stone.add(int(stdin.readline().rstrip()))

# dp  = [[10001]* (int((2*N)**0.5)+2)  for _ in range(N+1)]

# dp[1][0] = 0
# for i in range(2, N+1):
#     if i in stone:
#         continue
#     for v in range(1,int((2*i)**0.5)+1):
#         dp[i][v] = min(dp[i-v][v-1],dp[i-v][v],dp[i-v][v+1]) +1


# ans = min(dp[N])
# if ans == 10001:
#     print(-1)
# else:
#     print(ans)