def makeFun(N):
    Matrix = [[0 for i in range(pow(2,N))] for row in range(pow(2,N))]
    return Matrix

def function(x,y,Matrix):
    count = 0
    for i in range(0,pow(2,N),2):
        for j in range(pow(2,N)):
            if j % 2 == 0:
                count += 1
                Matrix[i][j] = count
                if i == x and j == y :
                    return Matrix
            else :
                count +=1
                Matrix[i][j] = count
                if i == x and j == y :
                    return Matrix
                count += 1
                Matrix[i+1][j-1] = count
                if i+1 == x and j-1 == y :
                    return Matrix
                count += 1
                Matrix[i+1][j] = count
                if i+1 == x and j == y :
                    return Matrix
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
