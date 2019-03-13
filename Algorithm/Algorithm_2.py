import random
def makeFun(N,M):
    Matrix = [['.'for i in range(M)] for j in range(N)]
    for i in range(N):
        for j in range(M):
            if random.randrange(1,5)!=1:
                Matrix[i][j] = '.';
            else:
                Matrix[i][j] = 'X';
    return Matrix

def function(N,M,Matrix):
    count2 = 0
    count1 = 0
    for i in range(N):
        for j in range(M):
            if(j < M-1 and i < N - 1) :
                if Matrix[i][j] == '.' and Matrix[i][j+1]=='.' :
                    count2 += 1
                    Matrix[i][j] = '2'
                    Matrix[i][j+1] = '2'
                elif Matrix[i][j] == '.' and Matrix[i+1][j]=='.':
                    count2 +=1
                    Matrix[i][j] = '2'
                    Matrix[i+1][j] = '2'
            elif j == M-1 and i < N-1:
                if Matrix[i][j] == '.' and Matrix[i+1][j] == '.' :
                    count2 +=1
                    Matrix[i][j] = '2'
                    Matrix[i + 1][j] = '2'
            elif j < M - 1 and i == N-1:
                if Matrix[i][j] == '.' and Matrix[i][j+1]=='.' :
                    count2 += 1
                    Matrix[i][j] = '2'
                    Matrix[i][j+1] = '2'
    for i in range(N):
        for j in range(M):
            if Matrix[i][j] =='.':
                Matrix[i][j] = '1'
                count1 += 1
    return count1 + count2




list = (input("N,M 값을 입력하시오 : ").split())
N = int(list[0])
M = int(list[1])
Matrix = makeFun(N,M) # 배열 생성
print('타일 깔기 전 학회방')
for i in range(N):
    for j in range(M):
        print(Matrix[i][j],end=' ')
    print()
print('타일의 개수 : ',function(N,M,Matrix))
print('타일 깐 후 학회방')
for i in range(N):
    for j in range(M):
        print(Matrix[i][j],end=' ')
    print()