def fib(n):
    return fibAux(n,1,0)

def fibAux(n,next,result):
    if(n==0):
        return result
    else:
        print('fibAux({},{},{})'.format(n-1,next + result,next))
        return fibAux(n - 1 , next + result , next)

try:
    number = int(input('정수를 입력하시오 : '))
    fib(number)
except ValueError:
    print('정수외의 문자를 입력했습니다.')
