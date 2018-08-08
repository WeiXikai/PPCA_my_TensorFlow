#include <stdio.h>
#include <stdlib.h>

// *(a + f(i, j, k, h, i_size, j_size, k_size)) is for the a[i][j][k][h])
inline int find(const int &i, const int &j, const int &k, const int &w, const int &j_size, const int &k_size, const int &w_size)
{
    return i * j_size * k_size * w_size + j * k_size * w_size + k * w_size + w;
}

inline int mat_find(const int &i, const int &j, const int &j_size)
{
    return i * j_size + j;
}

inline float max(const float &a, const float &b)
{
    return a > b ? a : b;
}

void test_matmul(const int m, const int n, const int k, const float alpha, float *a,
                const int lda, float *b, const int ldb, const float beta, float *c, const int ldc)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int w = 0; w < k; w++)
                *(c + mat_find(i, j, n)) += *(a + mat_find(i, w, k)) * (*(b + mat_find(w, j, n))) * alpha
                                            + (*(c + mat_find(i, j, n))) * beta;
        }
    }
}

int main()
{
    float *a = (float*)malloc(6 * sizeof(float));
    float *b = (float*)malloc(6 * sizeof(float));
    float *c = (float*)malloc(4 * sizeof(float));
    for (int i = 0; i < 6; i++)
    {
        *(a + i) = i;
        *(b + i) = i;
    }
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
            printf("%f ", *(a + mat_find(i, j, 3)));
        printf("\n");
    }
    printf("\n\n\n\n");
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 2; j++)
            printf("%f ", *(b + mat_find(i, j, 2)));
        printf("\n");
    }
    test_matmul(2, 2, 3, 1, a, 3, b, 2, 0, c, 2);

    printf("\n\n\n\n");
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
            printf("%f ", *(c + mat_find(i, j, 2)));
        printf("\n");
    }
    free(a);
    free(b);
    free(c);
    return 0;
}
