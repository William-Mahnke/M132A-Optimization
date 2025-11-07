import numpy as np
import matplotlib.pyplot as plt

def hog20new(Im, d, B):
    """
    Input:
        Im: input image. 
        d: signifies size of basic image patch to perform HOG.
           Typically d is between 3 and 7.
        B: number of histogram bins. Typically B is set between 7 and 9. 
    Output:
        h: HOG feature vector of input image.

    Usage:
    >>> X2 = scipy.io.loadmat('X2.mat')['X2']
    >>> hog20new(X2[:, 0].reshape(28, 28).T, 7, 7).sum()

    Verify on matlab:
    >>> hog20new( reshape(X2(:, 1),28,28), 7, 7)

    The result should be exactly the same.
    """
    Im = Im.T
    t = d // 2
    N, M = Im.shape

    k1 = (M - d) / t
    c1 = np.ceil(k1)
    k2 = (N - d) / t
    c2 = np.ceil(k2)

    if c1 - k1 > 0:
        M1 = int(d + t * c1)
        Im = np.hstack((Im, np.fliplr(Im[:, (2 * M - M1):M])))

    if c2 - k2 > 0:
        N1 = int(d + t * c2)
        Im = np.vstack((Im, np.flipud(Im[(2 * N - N1):N, :])))

    N, M = Im.shape
    nx1 = np.arange(0, M - d + 1, t)
    nx2 = nx1 + d - 1
    ny1 = np.arange(0, N - d + 1, t)
    ny2 = ny1 + d - 1

    Lx = len(nx1)
    Ly = len(ny1)
    h = np.zeros(Lx * Ly * B)

    Im = Im.astype(float)

    Gx = np.hstack((Im[:, 1:], np.zeros((N, 1)))) - np.hstack((np.zeros((N, 1)), Im[:, :-1]))
    Gy = np.vstack((np.zeros((1, M)), Im[:-1, :])) - np.vstack((Im[1:, :], np.zeros((1, M))))

    mag = np.sqrt(Gx**2 + Gy**2)
    ang = np.arctan2(Gy, Gx)

    c3 = (B - 1e-6) / (2 * np.pi)
    I = np.round((ang + np.pi) * c3 + 0.5).astype(int)
    I[I == 0] = 1

    k = 0
    zb = np.zeros(B)
    Lt = d**2

    for m in range(Lx):
        for n in range(Ly):
            ht = np.copy(zb)
            mag_patch = mag[ny1[n]:ny2[n]+1, nx1[m]:nx2[m]+1].flatten()
            ang_patch = I[ny1[n]:ny2[n]+1, nx1[m]:nx2[m]+1].flatten()

            for i in range(Lt):
                ai = ang_patch[i] - 1
                ht[ai] += mag_patch[i]

            norm_ht = np.linalg.norm(ht)
            if norm_ht != 0:
                ht /= (norm_ht + 0.01)

            h[k*B:(k+1)*B] = ht
            k += 1

    return h

def sgd_rlog(X, y, Xte1, Xte2, mu, bt, gm, m, st, iter, f_rlog, g_rlog):
    """
    Input:
     - f_rlog : loss function
     - g_rlog : gradient of loss function
    Usage:
    >>> ws, f, rtm = sgd_rlog(Xhog,y,T2_hog,T7_hog,0.002,13,0.01,8,9,1176,f_rlog, g_rlog)

    """
    N, P = X.shape
    Xh = np.vstack([X, np.ones(P)])
    t1, t2 = Xte1.shape[1], Xte2.shape[1]
    A1 = np.vstack([Xte1, np.ones(t1)])
    A2 = np.vstack([Xte2, np.ones(t2)])
    w0 = np.zeros(N + 1)
    wk = w0
    print(wk.shape)
    print(Xh.shape)
    print(y.shape)
    f = [f_rlog(wk, Xh, y, 0)] 
    y1 = np.dot(wk.T, A1)
    y2 = np.dot(wk.T, A2)
    
    C = np.zeros((2, 2))
    C[0, 0] = np.sum(y1 > 0)
    C[1, 0] = t1 - C[0, 0]
    C[1, 1] = np.sum(y2 < 0)
    C[0, 1] = t2 - C[1, 1]
    
    rtm = [(1 - ((C[0, 0] + C[1, 1]) / np.sum(C))) * 100]
    aw = np.array([bt / (1 + gm * k) for k in range(1, iter + 1)])
    
    np.random.seed(st)
    for k in range(iter):
        r = np.random.permutation(P)
        Xw = Xh[:, r[:m]]
        yw = y[r[:m]]
        
        gk = g_rlog(wk, Xw, yw, mu)
        dk = -gk
        adk = aw[k] * dk
        wk = wk + adk
        
        fk = f_rlog(wk, Xh, y, 0)
        f.append(fk)
        
        y1 = np.dot(wk.T, A1)
        y2 = np.dot(wk.T, A2)
        
        C[0, 0] = np.sum(y1 > 0)
        C[1, 0] = t1 - C[0, 0]
        C[1, 1] = np.sum(y2 < 0)
        C[0, 1] = t2 - C[1, 1]
        
        rk = (1 - ((C[0, 0] + C[1, 1]) / np.sum(C))) * 100
        rtm.append(rk)
    
    print(fk)
    ws = wk
    print(C)
    print("rate of misclassification in percentage: ", rk)
    
    plt.figure(figsize=(8, 6))
    plt.semilogy(range(iter + 1), f, 'b-', linewidth=1.4)
    plt.xlabel('Iterations $k$', fontsize=13)
    plt.ylabel('Objective $E_L(w_k)$', fontsize=13)
    plt.grid(True)
    plt.show()
    
    return ws, f, rtm

def SVRG_rlog(X, y, Xte1, Xte2, mu, a, T, st, K):
    """
    SVRG algorithm for logistic regression.
    - X: training data, shape (n_features, n_samples)
    - y: labels, shape (n_samples,), with values +1 or -1.
    - Xte1, Xte2: two test sets, with shapes (n_features, n_test1) and (n_features, n_test2)
    - mu: regularization parameter.
    - a: step size.
    - T: number of inner iterations.
    - st: initial random seed.
    - K: number of outer iterations.
    """
    n_features, P = X.shape
    # Augment training data with a row of ones for bias term
    Xh = np.vstack((X, np.ones((1, P))))
    
    t1 = Xte1.shape[1]
    t2 = Xte2.shape[1]
    A1 = np.vstack((Xte1, np.ones((1, t1))))
    A2 = np.vstack((Xte2, np.ones((1, t2))))
    
    w0 = np.zeros(n_features + 1)
    wt = w0.copy()
    
    f0 = f_rlog(wt, Xh, y, 0)  # Objective computed with mu=0
    f_vals = [f0]
    
    # Initial prediction and confusion matrix
    y1 = np.dot(wt, A1)
    y2 = np.dot(wt, A2)
    C = np.zeros((2,2), dtype=int)
    C[0,0] = np.sum(y1 > 0)
    C[1,0] = t1 - C[0,0]
    C[1,1] = np.sum(y2 < 0)
    C[0,1] = t2 - C[1,1]
    rtm = [(1 - ((C[0,0] + C[1,1]) / C.sum())) * 100]
    
    print(f'iter 0: obj = {f0:.2e}')
    
    tt = 1  # mini-batch size (selecting one sample per inner iteration)
    for k in range(1, K+1):
        gt = g_rlog(wt, Xh, y, mu)
        wk = wt.copy()
        for t in range(1, T+1):
            # Reset the random seed (note: frequent seed resetting is uncommon and mainly used here for reproducibility)
            np.random.seed(st + (k-1)*T + t)
            ind = np.random.permutation(P)
            it = ind[:tt]
            xi = Xh[:, it]   # shape: (n_features+1, tt)
            yi = y[it]       # shape: (tt,)
            gik = g_rlog(wk, xi, yi, mu)
            git = g_rlog(wt, xi, yi, mu)
            gk = gik - git + gt
            wk = wk - a * gk
        wt = wk.copy()
        fk = f_rlog(wt, Xh, y, 0)
        f_vals.append(fk)
        
        y1 = np.dot(wt, A1)
        y2 = np.dot(wt, A2)
        C[0,0] = np.sum(y1 > 0)
        C[1,0] = t1 - C[0,0]
        C[1,1] = np.sum(y2 < 0)
        C[0,1] = t2 - C[1,1]
        rk = (1 - ((C[0,0] + C[1,1]) / C.sum())) * 100
        rtm.append(rk)
        
        print(f'iter {k}: obj = {fk:.2e}')
        
    ws = wt
    print('Objective function at the solution point:')
    fs = f_vals[-1]
    print(fs)
    print('Confusion matrix:')
    print(C)
    print('Rate of misclassification in percentage:')
    print(rtm[-1])
    
    # Plot the objective function values (semilog plot)
    plt.figure(1)
    plt.semilogy(range(0, K+1), f_vals, 'b-', linewidth=1.5)
    plt.xlabel('Iterations k', fontsize=15, fontname='times')
    plt.ylabel('Objective E_L(w_k)', fontsize=15, fontname='times')
    plt.axis([0, K, 1.5e-2, 1])
    plt.grid(True)
    plt.show()
    
    return ws, f_vals, rtm