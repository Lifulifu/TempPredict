

def mae(a, b):
    if a.shape[0] != b.shape[0]:
        return -1
    N = a.shape[0]
    absSum = 0
    for n in range(N):
        absSum = absSum + abs(a[n] - b[n])
    return absSum / N


def mse(a, b):
    N = a.shape[0]
    sqrSum = 0
    for n in range(N):
        sqrSum = sqrSum + (a[n] - b[n])**2
    return sqrSum / N

    
    