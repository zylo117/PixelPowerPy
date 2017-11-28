# coding=utf-8
def is_primenumber(n):
    if n <= 1:
        return False
    if n == 2:
        return True
    if n == 3:
        return False
    for i in range(2, n // 2 + 1):
        if n % i == 0:
            return True
    return False
