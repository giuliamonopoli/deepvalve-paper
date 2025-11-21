import pprint

import numpy as np
import torch
from icp import iterative_closest_point
from my_procrustes import procrustes
from plotter import plot_all_splines
from plotter import plot_results_by_flag
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.transform import Rotation
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from spline_fitting import bspline_from_points


class EvaluateRegression(object):
    """
    Evaluate regression model performance using the following metrics:
    - Mean Absolute Error (mae)
    - Mean Squared Error (mse)
    - Root Mean Squared Error (rmse)
    - Mean Absolute Percentage Error (mape)
    - Cosine Similarity (cosine_similarity)
    - Procrustes distance (pro_dist)
    """

    def __init__(self, y_true, y_pred, weights=None):
        self.y_true = y_true
        self.y_pred = y_pred
        if weights is None:
            self.weights = np.ones(y_true.shape)
        else:
            self.weights = weights
            assert (
                self.weights.shape == self.y_true.shape
            ), "Weights must match shape of data"

    def evaluate(self):
        return {
            "mae": self._mae(),
            "mse": self._mse(),
            "rmse": self._rmse(),
            "mape": self._mape(),
            "cosine_similarity": self._cos_sim(),
            "pro_dist": self._procrustes_dist(),
            "pro_rotation_angle": self._orthog_procrustes(),
            # "align_vectors_angle": self._align_vectors(),
        }

    def compare_spline_fit(self):
        shape = self.y_true.shape
        c_true = []
        c_pred = []
        for i in range(shape[0]):
            spline_y_true, xx_true, c_t, t_true = bspline_from_points(
                self.y_true[i, :, :],
            )
            spline_y_pred, xx_pred, c_p, t_pred = bspline_from_points(
                self.y_pred[i, :, :],
            )
            c_true.append(c_t)
            c_pred.append(c_p)
        return c_true, c_pred

    def _mae(self):
        # return mean_absolute_error(self.y_true, self.y_pred, multioutput="raw_values")
        return np.multiply(np.abs(self.y_true - self.y_pred), self.weights).mean()

    def _mse(self):
        return np.multiply((self.y_true - self.y_pred) ** 2, self.weights).mean()

    def _rmse(self):
        # Does not use weights
        return np.sqrt(((self.y_true - self.y_pred) ** 2).mean())

    def _mape(self):
        # Does not use weights
        return (np.abs(self.y_true - self.y_pred) / self.y_true).mean()

    def _cos_sim(self):
        shape = self.y_true.shape
        return np.array(
            [
                cosine_similarity(
                    self.y_true[i, :, :].reshape(1, -1),
                    self.y_pred[i, :, :].reshape(1, -1),
                )
                for i in range(shape[0])
            ],
        ).mean()

    def _procrustes_dist(self):
        shape = self.y_true.shape
        pro_dist = []
        for i in range(shape[0]):  # iterate over all cases
            d, Z, tForm = procrustes(
                self.y_pred[i, :, :],
                self.y_true[i, :, :],
                scaling=False,
                reflection=False,
            )
            pro_dist.append(np.sqrt(d))
        return np.array(pro_dist).mean()

    def _orthog_procrustes(self):
        shape = self.y_true.shape
        pro_rotate = []
        for i in range(shape[0]):
            R, scale = orthogonal_procrustes(
                self.y_pred[i, :, :] - self.y_pred[i, :, :].mean(axis=0, keepdims=True),
                self.y_true[i, :, :] - self.y_true[i, :, :].mean(axis=0, keepdims=True),
            )
            pro_rotate.append(self._angle_from_rot_matrix(R))
        return np.array(pro_rotate).mean()

    def _align_vectors(self):
        """
        Returns the rotation angle from scipy Rotation.align_vectors.
        Uses the Kabsch algorithm to find the rotation matrix that aligns the
        predicted points to the ground truth points. Same output as _orthog_procrustes().
        """
        shape = self.y_true.shape
        rotations = []
        for i in range(shape[0]):
            rotation, rssd = Rotation.align_vectors(
                np.hstack(
                    (
                        self.y_true[i, :, :]
                        - self.y_true[i, :, :].mean(axis=0, keepdims=True),
                        np.zeros((shape[1], 1)),
                    ),
                ),
                np.hstack(
                    (
                        self.y_pred[i, :, :]
                        - self.y_true[i, :, :].mean(axis=0, keepdims=True),
                        np.zeros((shape[1], 1)),
                    ),
                ),
            )
            rotations.append(self._angle_from_rot_matrix(rotation.as_matrix()))
        return np.array(rotations).mean()

    def _angle_from_rot_matrix(self, R):
        """
        Return the rotation angle in degrees from a rotation matrix.
        """
        return np.abs(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


class EvaluateSegmentation(object):
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def evaluate(self):
        return {
            "iou": self._iou(),
            "dice": self._dice(),
            "icp": self._icp(),
        }

    def _iou(self):
        return jaccard_score(self.y_true, self.y_pred, average="micro")

    def _dice(self):
        iou_score = float(self._iou())  # Convert iou() result to float
        return 2.0 * iou_score / (1.0 + iou_score)

    def _icp(self):
        """
        Iterative Closest Point (ICP) algorithm to find the best transformation matrix
        that maps the predicted points to the ground truth points. Returns the final error.
        """

        y_pred_coords = np.argwhere(self.y_pred)
        y_true_coords = np.argwhere(self.y_true)
        T, finalA, final_error, i = iterative_closest_point(
            y_pred_coords,
            y_true_coords,
            max_iterations=100,
            tolerance=1e-5,
        )
        return final_error


def load_model_reg(model_path):
    model_data = torch.load(model_path)
    results = {}  # combine batches
    for key in [
        "ground_truth_septal",
        "ground_truth_lateral",
        "predicted_septal",
        "predicted_lateral",
    ]:
        tmp = []
        # for i in range(len(model_data[key])):
        tmp.append(model_data[key][0])
        results[key] = torch.cat(tuple(tmp), 0).cpu().detach().numpy()
    check_shape_of_variables(results)  # check shape of GTs and predictions
    for key in ["images"]:
        tmp = []
        # for i in range(len(model_data[key])):
        tmp.append(model_data[key][0])
        results[key] = torch.cat(tuple(tmp), 0).cpu().detach().numpy()
    for key in ["fp", "ff"]:
        if key == "ff":
            for i in range(len(model_data[key])):
                if model_data["ff"][i] == [0]:
                    model_data["ff"][i] = []
        mlb = MultiLabelBinarizer()  # new instance for each flag set (ff, fp)
        results[key] = mlb.fit_transform(model_data[key])
    results["flags"] = np.hstack((results["fp"], results["ff"]))
    return results


def load_model_seg(model_path):
    model_data = torch.load(model_path)
    results = {}
    for key in [
        "ground_truth_mask",
        "predicted_mask",
    ]:
        tmp = []
        for i in range(len(model_data[key])):
            tmp.append(np.array(model_data[key][i], dtype=int))
        results[key] = np.array(tmp)[:, :, :, 1]
    for key in ["images"]:
        tmp = []
        for i in range(len(model_data[key])):
            tmp.append(np.array(model_data[key][i][0]))
        results[key] = np.array(tmp)
    return results


def check_shape_of_variables(results):  # index of GT/predicted is hard-coded here
    shapes = [data.shape for data in results.values()]
    if not all(shape == shapes[0] for shape in shapes):
        print(shapes)
        raise AssertionError("Shapes of variables do not match")
    else:
        print("Variable size check passed.")


def evaluate_regression(model_name):
    fname_out = f"results/METRICS_{model_name}.txt"
    with open(fname_out, "w") as f:
        f.write(model_name + "\n")

    results = load_model_reg(f"models/{model_name}.pth")

    print("#####################")
    print(model_name)
    print("#####################")
    # Regression - all test cases, each leaflet
    names = ["lateral", "septal"]
    for name in names:
        y_true = results[f"ground_truth_{name}"]
        y_pred = results[f"predicted_{name}"]
        print("\n" + name + "\n")
        pprint.pprint(EvaluateRegression(y_true, y_pred).evaluate())
        with open(fname_out, "a") as f:
            f.write("\n" + name + "\n")
            f.write(
                pprint.pformat(EvaluateRegression(y_true, y_pred).evaluate()),
            )
            f.write("\n")

    # Regression - all test cases, combined leaflets
    y_true = np.hstack(
        (results["ground_truth_lateral"], results["ground_truth_septal"]),
    )
    y_pred = np.hstack((results["predicted_lateral"], results["predicted_septal"]))
    RegResults = EvaluateRegression(y_true, y_pred)
    print("\nCombined leaflets")
    pprint.pprint(RegResults.evaluate())
    # pprint.pprint(RegResults.compare_spline_fit())
    with open(fname_out, "a") as f:
        f.write("\nCombined leaflets\n")
        f.write(pprint.pformat(RegResults.evaluate()))
        f.write("\n")

    # Regression - metrics by test case
    metrics_by_sample = {}
    y_true = np.hstack(
        (results["ground_truth_lateral"], results["ground_truth_septal"]),
    )
    y_pred = np.hstack((results["predicted_lateral"], results["predicted_septal"]))
    n_test_cases = results["images"].shape[0]
    for i in range(n_test_cases):
        metrics_by_sample[f"test_{str(i)}"] = EvaluateRegression(
            np.expand_dims(y_true[i], 0),  # class method calculates average
            np.expand_dims(y_pred[i], 0),
            weights=np.ones_like(np.expand_dims(y_true[i], 0)),
        ).evaluate()
        plot_all_splines(results, i, model_name)
    print("\n")
    pprint.pprint(metrics_by_sample, sort_dicts=False)
    with open(fname_out, "a") as f:
        f.write("\n")
        f.write(pprint.pformat(metrics_by_sample, sort_dicts=False))
        f.write("\n")

    plot_results_by_flag(metrics_by_sample, results["flags"], model_name)


def evaluate_segmentation(model_name):
    fname_out = f"results/METRICS_{model_name}.txt"
    with open(fname_out, "w") as f:
        f.write(model_name + "\n")

    results = load_model_seg(f"models/{model_name}.pth")

    metrics_by_sample = {}
    metrics = np.zeros((results["images"].shape[0], 3))
    n_test_cases = results["images"].shape[0]
    for i in range(n_test_cases):
        metrics_by_sample[f"test_{str(i)}"] = EvaluateSegmentation(
            results["ground_truth_mask"][i],
            results["predicted_mask"][i],
        ).evaluate()
        metrics[i, 0] = metrics_by_sample[f"test_{str(i)}"]["iou"]
        metrics[i, 1] = metrics_by_sample[f"test_{str(i)}"]["dice"]
        metrics[i, 2] = metrics_by_sample[f"test_{str(i)}"]["icp"]
    print("\n")
    pprint.pprint(metrics_by_sample, sort_dicts=False)
    with open(fname_out, "a") as f:
        f.write("\n")
        f.write(pprint.pformat(metrics_by_sample, sort_dicts=False))
        f.write("\n")


def main():
    model_name_reg = "{PyTorch_pth_state_dictionary_name}"
    model_name_seg = "{PyTorch_pth_state_dictionary_name}"
    # example: model_name_reg = "model_name" corresponding to "model_name.pth"

    # Regression
    evaluate_regression(model_name_reg)

    # Segmentation
    evaluate_segmentation(model_name_seg)


if __name__ == "__main__":
    main()
