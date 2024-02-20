import numpy as np
from numpy.polynomial import polynomial as poly

def polymul(x, y, modulus, poly_mod):
    return np.int64(
        np.round(poly.polydiv(poly.polymul(x, y) % modulus, poly_mod)[1] % modulus)
    )
def polyadd(x, y, modulus, poly_mod):
    return np.int64(
        np.round(poly.polydiv(poly.polyadd(x, y) % modulus, poly_mod)[1] % modulus)
    )
def gen_binary_poly(size):
    return np.random.randint(0, 2, size, dtype=np.int64)

def gen_uniform_poly(size, modulus):
    return np.random.randint(0, modulus, size, dtype=np.int64)

def gen_normal_poly(size):
    return np.int64(np.random.normal(0, 2, size=size))
def keygen(size, modulus, poly_mod):
    sk = gen_binary_poly(size)
    a = gen_uniform_poly(size, modulus)
    e = gen_normal_poly(size)
    b = polyadd(polymul(-a, sk, modulus, poly_mod), -e, modulus, poly_mod)
    return (b, a), sk
def encrypt(pk, size, q, t, poly_mod, pt):
    m = np.array([pt] + [0] * (size - 1), dtype=np.int64) % t
    delta = q // t
    scaled_m = delta * m % q
    e1 = gen_normal_poly(size)
    e2 = gen_normal_poly(size)
    u = gen_binary_poly(size)
    ct0 = polyadd(
            polyadd(
                polymul(pk[0], u, q, poly_mod),
                e1, q, poly_mod),
            scaled_m, q, poly_mod
        )
    ct1 = polyadd(
            polymul(pk[1], u, q, poly_mod),
            e2, q, poly_mod
        )

    return (ct0, ct1)
def decrypt(sk, size, q, t, poly_mod, ct):
    scaled_pt = polyadd(
            polymul(ct[1], sk, q, poly_mod),
            ct[0], q, poly_mod
        )
    decrypted_poly = np.round(scaled_pt * t / q) % t
    return int(decrypted_poly[0])
def add_plain(ct, pt, q, t, poly_mod):
    size = len(poly_mod) - 1
    m = np.array([pt] + [0] * (size - 1), dtype=np.int64) % t
    delta = q // t
    scaled_m = delta * m  % q
    new_ct0 = polyadd(ct[0], scaled_m, q, poly_mod)
    return (new_ct0, ct[1])
def mul_plain(ct, pt, q, t, poly_mod):
    size = len(poly_mod) - 1
    m = np.array([pt] + [0] * (size - 1), dtype=np.int64) % t
    new_c0 = polymul(ct[0], m, q, poly_mod)
    new_c1 = polymul(ct[1], m, q, poly_mod)
    return (new_c0, new_c1)
if __name__ == '__main__':
    n_degree= int(input('Please input the polynomial coefficient:'))
    n = 2 ** n_degree
    # ciphertext modulus
    q = 2 ** 15
    # plaintext modulus
    t = 2 ** 8
    # polynomial modulus
    poly_mod = np.array([1] + [0] * (n - 1) + [1])
    # Keygen
    pk, sk = keygen(n, q, poly_mod)
    # Encryption
    pt1=int(input('Please input the plaintext:'))
    pt2=int(input('Please input the plaintext:'))

    ct1 = encrypt(pk, n, q, t, poly_mod, pt1)
    ct2 = encrypt(pk, n, q, t, poly_mod, pt2)
    decrypted_ct1 = decrypt(sk, n, q, t, poly_mod, ct1)
    decrypted_ct2 = decrypt(sk, n, q, t, poly_mod, ct2)



    print("[+] Ciphertext ct1({}):".format(pt1))
    print("")
    print("\t ct1_0:", ct1[0])
    print("\t ct1_1:", ct1[1])
    print("")
    print("[+] Ciphertext ct2({}):".format(pt2))
    print("")
    print("\t ct1_0:", ct2[0])
    print("\t ct1_1:", ct2[1])
    print("")
    print("[+] Ciphertext ct1({}):".format(decrypted_ct1))
    print("[+] Ciphertext ct2({}):".format(decrypted_ct2))
    print("")


    cst1=int(input('Please input the addition factor:'))
    cst2=int(input('Please input the multiplication factor:'))
    # Evaluation
    ct3 = add_plain(ct1, cst1, q, t, poly_mod)
    ct4 = mul_plain(ct2, cst2, q, t, poly_mod)

    # Decryption
    decrypted_ct3 = decrypt(sk, n, q, t, poly_mod, ct3)
    decrypted_ct4 = decrypt(sk, n, q, t, poly_mod, ct4)

    print("[+] Decrypted ct3(ct1 + {}): {}".format(cst1, decrypted_ct3))
    print("[+] Decrypted ct4(ct2 * {}): {}".format(cst2, decrypted_ct4))


    cst3 = int(input('Please input the addition factor:'))
    cst4 = int(input('Please input the multiplication factor:'))

    ct5 = add_plain(ct3, cst3, q, t, poly_mod)
    ct6 = mul_plain(ct4, cst4, q, t, poly_mod)

    # Decryption
    decrypted_ct3 = decrypt(sk, n, q, t, poly_mod, ct5)
    decrypted_ct4 = decrypt(sk, n, q, t, poly_mod, ct6)

    print("[+] Decrypted ct3(ct1 + {}): {}".format(cst3, decrypted_ct3))
    print("[+] Decrypted ct4(ct2 * {}): {}".format(cst4, decrypted_ct4))