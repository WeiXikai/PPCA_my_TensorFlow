#include <cstdio>

// *(a + f(i, j, k, h, i_size, j_size, k_size)) is for the a[i][j][k][h])
int f(const int &i, const int &j, const int &k, const int &w, const int &j_size, const int &k_size, const int &w_size)
{
    return i * j_size * k_size * w_size + j * k_size * w_size + k * w_size + w;
}

extern "C"
void test_func(int a)
{
    printf("%d\n", a);
    printf("test sync\n");
}

extern "C"
void test_array(float *a, int i_size, int j_size, int k_size)
{
    for (int k = 0; k < k_size; k++)
    {
        for (int i = 0; i < i_size; i++)
        {
            for (int j = 0; j < j_size; j++)
                printf("%f ", *(a + k * j_size * i_size + i * j_size + j));
            printf("\n");
        }
        printf("\n");
    }
}
