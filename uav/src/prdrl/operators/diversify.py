from .local import local


def diversify(X_region, n_offspring, xl, xu, sigma, rng):
    """
    DIVERSIFY (eq. 47): x' = Pi_Omega[x + sigma_D*(u-l)*xi] -- CUNG cong
    thuc voi LOCAL (eq.44), chi khac do lon sigma (rang buoc sigma_D >
    sigma_P > sigma_L do CALLER dam bao khi truyen sigma vao, khong
    phai operator tu enforce). Goi thang local() la DUNG, khong phai
    stub -- da xac nhan, khong sua. (Phan "tron mot ty le nho mau
    global/LHS" trong dac ta la optional ("co the"), CHUA implement --
    khong phai bug, chi la tinh nang mo rong chua lam.)
    """
    return local(X_region, n_offspring, xl, xu, sigma, rng)