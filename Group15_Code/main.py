import numpy as np
def generate_key(w,m,n):
    S = (np.random.rand(m,n) * w / (2 ** 16))
    return S
def encrypt(x,S,m,n,w):
    e = (np.random.rand(m))
    c = np.linalg.inv(S).dot((w * x) + e) #
    return c
def decrypt(c,S,w):
    x_recon = (S.dot(c) / w).astype('int')
    return x_recon

if __name__ == '__main__':
    x = np.array([0, 1, 2, 5])
    m = len(x)
    n = m
    w = 16
    S = generate_key(w, m, n)
    c = encrypt(x, S, m, n, w)
    x_recon = decrypt(c, S, w)
    print(x_recon)


