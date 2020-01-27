#include <stdio.h>
int main(void){
    int n , m;
    int j , k, h;
    scanf("%d %d", &n, &m);
    scanf("%d %d", &j, &k);
    scanf("%d", &h);
    int mult, m2, m1;
    int ans;
    if (k > m){
        mult = k/m;
    }
    if (m > k){
        mult = m/k;
    }
    
    m2 = mult * j;
    m1 = ((float)h) / ((float)n);
    ans = (int)(m2 * m1);

    printf("%d\n", ans);
    return 0;
}