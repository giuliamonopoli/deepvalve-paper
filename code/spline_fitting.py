import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate


def bspline_from_points(pts, plotit=False):
    """
    Fit a B-spline to a set of points.

    Parameters
    ----------
    pts : array_like
        Array of points to fit a B-spline to.
    plotit : bool, optional
        Plot the B-spline if True.

    Returns
    -------
    bspline : scipy.interpolate.BSpline
        The fitted B-spline.
    xx : array_like
        The x-coordinates of the B-spline.
    c : array_like
        The coefficients of the B-spline.
    """
    x = pts[:, 0]
    y = pts[:, 1]
    distance = get_cumulative_distance(pts)
    N_pts_path = 50
    alpha = np.linspace(0, 1, N_pts_path)

    # t, c, k = interpolate.splrep(x, y, s=0)
    # bspline = interpolate.BSpline(t, c, k, extrapolate=False)
    # N_x = 50
    # x_min, x_max = x.min(), x.max()
    # xx = np.linspace(x_min, x_max, N_x)

    spline, u = interpolate.splprep(pts.T, u=distance, s=0)
    interpolated_pts = interpolate.splev(alpha, spline)

    fig, ax = plt.subplots()
    ax.plot(x, y, "bo", label="Original points")
    ax.plot(interpolated_pts[0], interpolated_pts[1], "r", label="BSpline")
    ax.grid()
    ax.legend(loc="best")

    if plotit:
        plt.show()
    else:
        plt.savefig("results/bspline.png")
        plt.close()

    # return bspline(xx), xx, c, t
    return spline, interpolated_pts


def univariate_spline_fit(pts, plotit=False):
    N_pts_path = 50
    distance = get_cumulative_distance(pts)
    splines = [
        interpolate.UnivariateSpline(distance, coords, k=3, s=0) for coords in pts.T
    ]

    alpha = np.linspace(0, 1, N_pts_path)
    pts_fitted = np.vstack(spl(alpha) for spl in splines).T

    plt.plot(pts[:, 0], pts[:, 1], "bo", label="Original points")
    plt.plot(pts_fitted[:, 0], pts_fitted[:, 1], "r", label="Fitted spline")
    plt.grid()
    plt.legend(loc="best")
    if plotit:
        plt.show()
    else:
        plt.savefig("results/univariate.png")
        plt.close()
    return splines


def cubic_interp_from_points(pts, plotit=False):
    """
    Fit a cubic interpolation to a set of points.

    Parameters
    ----------
    pts : array_like
        Array of points to fit a cubic interpolation to.
    plotit : bool, optional
        Plot the cubic interpolation if True.

    Returns
    -------
    interpolated_pts : array_like
        The interpolated points.
    """
    N_pts_path = 50
    distance = get_cumulative_distance(pts)
    alpha = np.linspace(0, 1, N_pts_path)
    interpolator = interpolate.interp1d(distance, pts, kind="cubic", axis=0)
    interpolated_pts = interpolator(alpha)

    fig, ax = plt.subplots()
    ax.plot(pts[:, 0], pts[:, 1], "bo", label="Original points")
    ax.plot(
        interpolated_pts[:, 0],
        interpolated_pts[:, 1],
        "r",
        label="Cubic curve",
    )
    ax.grid()
    ax.legend(loc="best")
    if plotit:
        plt.show()
    else:
        plt.savefig("results/interp1d.png")
        plt.close()

    return interpolated_pts, interpolator


def get_cumulative_distance(pts):
    """
    Return the cumulative distance between points in pts.
    """
    distance = np.cumsum(np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1)))
    return np.insert(distance, 0, 0) / distance[-1]


# x = np.array([0, 1, 2, 3, 4, 5, 6, 7])
# y = np.random.randint(0, 10, size=8)

# x = np.array([0.0, 1.2, 1.9, 3.2, 4.0, 6.5])
# y = np.array([0.0, 2.3, 3.0, 4.3, 2.9, 3.1])

# data = np.array(
#     [
#         [0.68526786, 0.76116071],
#         [0.66676431, 0.75341259],
#         [0.64907075, 0.74172166],
#         [0.63488989, 0.72649754],
#         [0.62680648, 0.70817047],
#         [0.62367381, 0.68782212],
#         [0.62038781, 0.66720553],
#         [0.61687596, 0.64696359],
#         [0.61652162, 0.62698436],
#         [0.62276786, 0.60714286],
#         [0.59151786, 0.46205357],
#         [0.59199159, 0.47686914],
#         [0.58936325, 0.49371672],
#         [0.5868426, 0.51121608],
#         [0.58763943, 0.52798698],
#         [0.59486833, 0.54268],
#         [0.60630384, 0.55567588],
#         [0.61168912, 0.56996009],
#         [0.60928909, 0.58638441],
#         [0.61830357, 0.60044643],
#     ]
# )

# data = np.vstack((data[0:10], np.flipud(data[10:])))
# splines = univariate_spline_fit(data, plotit=False)
# bspline_from_points(data, plotit=False)
# _, interpolator = cubic_interp_from_points(data, plotit=False)
