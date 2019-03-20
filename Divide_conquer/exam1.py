import numpy as np



n,k = input('개수와 K를 입력하시오').split()
n = int(n)
k = int(k)
list = np.zeros(5,dtype="i")
for i in range(n):
    list[i] = int(input('{}번째 입력 : '.format(i)))
print('입력 배열 : ',list)

list.sort()
print(list)


