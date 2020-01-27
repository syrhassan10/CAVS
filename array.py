
N = int(input())
stg = input()
ary = []
for y in range(len(stg)-1):
    if stg[y] == " ":
        ary.append(int(stg[y - 1]))
    
d = 0
j = 0
while (j < (N - 2*d)):
    if (ary[j] == 8):
        ary.pop(j)
        ary.pop(j)
        d = d + 1
        j = j - 2
    j = j + 1
for i in ary:
    print(i, " ")