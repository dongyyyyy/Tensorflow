def makeFun(N):
    Matrix = [[0 for i in range(pow(2,N))] for row in range(pow(2,N))]
    return Matrix

def function(x,y,Matrix):
    count = 0
    for i in range(pow(2,N)):
        for j in range(pow(2,N)):
            if j == 0:
                if i == 0:
                    Matrix[i][j] = 1
                elif i%2 == 0:
                    Matrix[i][j] = Matrix[i-1][pow(2,N)-1]+1
                else:
                    Matrix[i][j] = int(Matrix[i - 1][j]) + 2
            elif j%2 == 0: # 2 , 4 , 6 ...
                Matrix[i][j] = int(Matrix[i][j-1])+3
            else: # 1 , 3 , 5 ...
                Matrix[i][j] = int(Matrix[i][j-1])+1
    print()
    return Matrix




list = (input("N,x,y 값을 입력하시오 : ").split())
N = int(list[0])
x = int(list[1])
y = int(list[2])
Matrix = makeFun(N) # 배열 생성
Matrix = function(x,y,Matrix)

for i in range(pow(2,N)):
    for j in range(pow(2, N)):
        print(Matrix[i][j],end=' ')
    print()
print('Matrix[{}][{}] = {}'.format(x,y,Matrix[x][y]))
