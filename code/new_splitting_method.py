from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

# def load_data(file_path):
#     df = pd.read_csv(file_path)
#     return df['patient_name'].tolist(), df[['fp_1', 'fp_2', 'fp_3', 'ff_1', 'ff_2', 'ff_3']].values.tolist()

# def get_combinations_counts(flags_list):
#     class_counts = Counter(map(tuple, flags_list))
#     filtered_classes = [list(key) for key, value in class_counts.items() if value > 1]
#     one_class = [list(key) for key, value in class_counts.items() if value == 1]
#     return filtered_classes, one_class

# def split_data(patient_names, flags_list,filtered_classes, one_class, test_size=0.3, random_state=47):
#     flags_list = np.array(flags_list)
#     patient_names = np.array(patient_names)

#     indices = [index for value in filtered_classes for index, name in enumerate(flags_list) if np.array_equal(name, np.array(value))]
#     indices_one = [index for value in one_class for index, name in enumerate(flags_list) if np.array_equal(name, np.array(value))]

#     selected_flags = flags_list[indices]
#     selected_patients = patient_names[indices]

#     patients_one = patient_names[indices_one]
#     flags_one = flags_list[indices_one]

#     X_train, X_test, y_train, y_test = train_test_split(selected_patients, selected_flags, test_size=test_size, random_state=random_state, stratify=selected_flags)

#     X_train = np.append(X_train, patients_one)
#     y_train = np.append(y_train, flags_one, axis=0)

#     return X_train, X_test, y_train, y_test

# def process_and_split_data(file_path, test_size=0.3, random_state=47):
#     patient_names, flags_list = load_data(file_path)
#     filtered_classes, one_class = get_combinations_counts(flags_list)
#     return split_data(patient_names, flags_list,filtered_classes, one_class, test_size=test_size, random_state=random_state)

# def main():
#     X_train, X_test, y_train, y_test = process_and_split_data('/Users/giuliamonopoli/Desktop/PhD /deepvalve/data/data_new.csv')
#     filtered_classes_t, one_class_t = get_combinations_counts(y_train)
#     X_train, X_val, y_train, y_val = split_data(X_train, y_train,filtered_classes_t, one_class_t, test_size=0.3, random_state=47)
#     return X_train, X_val, X_test
# # print(f"The data has been split into {len(X_train)/68} for training, {len(X_val)/68} for validation, and {len(X_test)/68} for test.")


from sklearn.model_selection import train_test_split
from collections import Counter

from sklearn.model_selection import train_test_split
from collections import Counter

" Method 1"


def load_data(file_path):
    df = pd.read_csv(file_path)
    return (
        df["patient_name"].tolist(),
        df[["fp_1", "fp_3", "ff_1", "ff_2", "ff_3"]].values.tolist(),
    )


def get_combinations_counts(flags_list):
    class_counts = Counter(map(tuple, flags_list))
    filtered_classes = [list(key) for key, value in class_counts.items() if value > 1]
    one_class = [list(key) for key, value in class_counts.items() if value == 1]
    return filtered_classes, one_class


def split_data(
    patient_names,
    flags_list,
    filtered_classes,
    one_class,
    test_size=0.3,
    step="one",
    random_state=47,
):
    flags_list = np.array(flags_list)
    patient_names = np.array(patient_names)

    indices = [
        index
        for value in filtered_classes
        for index, name in enumerate(flags_list)
        if np.array_equal(name, np.array(value))
    ]
    indices_one = [
        index
        for value in one_class
        for index, name in enumerate(flags_list)
        if np.array_equal(name, np.array(value))
    ]

    selected_flags = flags_list[indices]
    selected_patients = patient_names[indices]

    patients_one = patient_names[indices_one]
    flags_one = flags_list[indices_one]

    X_train, X_test, y_train, y_test = train_test_split(
        selected_patients,
        selected_flags,
        test_size=test_size,
        random_state=random_state,
        stratify=selected_flags,
    )

    if step == "one":
        X_test = np.append(X_test, patients_one)
        y_test = np.append(y_test, flags_one, axis=0)
    elif step == "second":
        np.random.shuffle(patients_one)
        split_index = len(patients_one) // 2
        X_train = np.append(X_train, patients_one[:split_index])
        y_train = np.append(y_train, flags_one[:split_index], axis=0)
        X_test = np.append(X_test, patients_one[:split_index])
        y_test = np.append(y_test, flags_one[split_index:], axis=0)

    return X_train, X_test, y_train, y_test


def process_and_split_data(file_path, test_size=0.3, random_state=47):
    patient_names, flags_list = load_data(file_path)
    filtered_classes, one_class = get_combinations_counts(flags_list)
    return split_data(
        patient_names,
        flags_list,
        filtered_classes,
        one_class,
        test_size=test_size,
        random_state=random_state,
    )


def split_sets():
    X_train, X_test, _, y_test = process_and_split_data(
        "/Users/giuliamonopoli/Desktop/PhD /deepvalve/data/data_new.csv"
    )

    filtered_classes_t, one_class_t = get_combinations_counts(y_test)

    X_test, X_val, y_test, _ = split_data(
        X_test,
        y_test,
        filtered_classes_t,
        one_class_t,
        test_size=0.3,
        step="second",
        random_state=47,
    )

    print(
        f"The data has been splitted into {len(X_train)/68} for training, {len(X_val)/68} for validation, and {len(X_test)/68} for test."
    )
    print(
        f"The data has been splitted into {len(X_train)} for training, {len(X_val)} for validation, and {len(X_test)} for test."
    )

    return X_train, X_val, X_test


# print(f"The data has been split into {len(X_train)/68} for training, {len(X_val)/68} for validation, and {len(X_test)/68} for test.")


if __name__ == "__main__":
    split_sets()
