"""
Functions for fits to be used when the problem can be transformed into a linear regression solved by singular value
decomposition.
"""

import numpy as np
import matplotlib.pyplot as plt
import src.fit_nonseparable as fit_nonseparable


def fit_hyperplane(x, y, add_bias=True):

    """
    Fits an m-dimensional hyperplane w to x and y using singular value decomposition, where w is the least squares fit
    for the system wx=y, x is an m-by-n matrix, and y is an m-by-1 column vector. If add_bias is True, then x is
    augmented to be an n-by-(m+1) and w becomes an (m+1)-by-1 column vector.
    """

    if add_bias:
        x = append_bias_col(x)
    ata = np.matmul(x.T, x)
    atb = np.matmul(x.T, y)

    u, s, v_t = np.linalg.svd(ata)
    s_inv = np.diag((1 / s))
    w = np.matmul(v_t.T, np.matmul(s_inv, np.matmul(u.T, atb)))

    return w


def eval_linear_fit(w, x, y, add_bias=True):
    y_predict = linear_predict(w, x, add_bias=add_bias)
    correlation = np.corrcoef(np.ravel(y_predict), np.ravel(y))[0, 1]
    return correlation


def linear_predict(w, x, add_bias=True):
    if add_bias:
        x = append_bias_col(x)
    y_predict = np.matmul(x, w)
    return y_predict


def append_bias_col(x):
    n, m = np.shape(x)
    bias_col = np.ones((n, 1))
    x = np.append(x, bias_col, axis=1)
    return x


def linear_transform(distortion_vector, distortion_ids):
    """
    Used for compatibility with fit functions that can also use non-linear axis transforms.
    """
    return distortion_vector


def nonlinear_transform_0(distortion_vector, distortion_ids=('res', 'blur', 'noise')):

    if distortion_ids != ('res', 'blur', 'noise'):
        raise Exception('distortion_ids must == (res, blur, noise)')

    n, m = np.shape(distortion_vector)

    res = distortion_vector[:, 0]
    blur = distortion_vector[:, 1]
    noise = distortion_vector[:, 2]

    f_res = np.log10(res)
    f_blur = np.log10(blur)
    f_noise = noise
    f_res_blur = blur / res

    transformed_distortion_vector = np.zeros((n, 4))

    transformed_distortion_vector[:, 0] = f_res
    transformed_distortion_vector[:, 1] = f_blur
    transformed_distortion_vector[:, 2] = f_noise
    transformed_distortion_vector[:, 3] = f_res_blur

    return transformed_distortion_vector


def nonlinear_transform_1(distortion_vector, distortion_ids=('res', 'blur', 'noise')):

    if distortion_ids != ('res', 'blur', 'noise'):
        raise Exception('distortion_ids must == (res, blur, noise)')

    n, m = np.shape(distortion_vector)

    res = distortion_vector[:, 0]
    blur = distortion_vector[:, 1]
    noise = distortion_vector[:, 2]

    f_res = np.log10(res)
    f_blur = blur
    f_noise = noise
    f_res_blur = blur / res

    transformed_distortion_vector = np.zeros((n, 4))

    transformed_distortion_vector[:, 0] = f_res
    transformed_distortion_vector[:, 1] = f_blur
    transformed_distortion_vector[:, 2] = f_noise
    transformed_distortion_vector[:, 3] = f_res_blur

    return transformed_distortion_vector


def fit(x, y, distortion_ids=('res', 'blur', 'noise'), add_bias=True, fit_key='linear'):

    try:
        transform = _transforms[fit_key]
    except KeyError:
        return fit_nonseparable.fit(x, y, distortion_ids=distortion_ids, fit_key=fit_key)
    else:
        x = transform(x, distortion_ids=distortion_ids)
        w = fit_hyperplane(x, y, add_bias=add_bias)
        return w


def apply_fit(w, x, distortion_ids=('res', 'blur', 'noise'), fit_key='linear', add_bias=True):

    try:
        transform = _transforms[fit_key]
    except KeyError:
        return fit_nonseparable.apply_fit(w, x, fit_key)
    else:
        x = transform(x, distortion_ids=distortion_ids)
        if add_bias:
            x = append_bias_col(x)
        y_predict = np.matmul(x, w)
        return y_predict


def evaluate_fit(w, x, y, distortion_ids=('res', 'blur', 'noise'), fit_key='linear', add_bias=True):
    y_predict = apply_fit(w, x, distortion_ids=distortion_ids, fit_key=fit_key, add_bias=add_bias)
    correlation = np.corrcoef(np.ravel(y_predict), np.ravel(y))[0, 1]
    return correlation


_transforms = {
    'linear': linear_transform,
    'nonlinear_0': nonlinear_transform_0,
    'nonlinear_1': nonlinear_transform_1,
}


# if __name__ == '__main__':
#
#     n_pts = 100
#     x = np.random.rand(n_pts, 3)
#     xt = distortion_transform(x)
#     w = np.atleast_2d(np.array([1, 2, 3, 4, 1])).T
#     y = linear_predict(w, xt)
#     y_hat = y + np.atleast_2d(np.random.randn(len(y))).T * 10
#
#     w_hat = nonlinear_fit(x, y_hat)
#
#     y_fit = linear_predict(w_hat, xt)
#     y_fit_compare = nonlinear_predict(w_hat, x)
#     assert np.array_equal(y_fit, y_fit_compare)
#
#     plt.figure()
#     plt.plot(y, label='y')
#     # plt.plot(y_hat, label='y + noise')
#     plt.plot(y_fit, label='fit')
#     plt.legend()
#     plt.show()
#
#     corr = eval_nonlinear_fit(w_hat, x, y)
# #
#
#
#
