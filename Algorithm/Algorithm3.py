number = input()

arr = list(number)
#
# for i in range(len(arr)):
#     arr[i] = int(arr[i])
arr = [int(i) for i in arr]

arr.sort(reverse = True)

sum = 0
for i in range(len(arr)):
    sum += arr[i]

if sum % 3 != 0:
    print(-1)
elif 0 not in arr:
    print(-1)
else :
    for i in range(len(arr)):
        print(arr[i],end='')