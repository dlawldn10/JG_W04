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
#7:50
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


import sys
read = sys.stdin.readline

word1, word2 = read().strip(), read().strip()
h, w = len(word1), len(word2)
cache = [[0] * (w+1) for _ in range(h+1)]

for i in range(1, h+1):
    for j in range(1, w+1):
        if word1[i-1] == word2[j-1]:
            cache[i][j] = cache[i-1][j-1] + 1
        else:
            cache[i][j] = max(cache[i][j-1], cache[i-1][j])
print(cache[-1][-1])