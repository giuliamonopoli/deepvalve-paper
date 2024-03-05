import json
import random


def load_and_sort(daniel_file, ashay_file):
    """Load and sort data from two JSON files.

    Args:
        daniel_file (str): File path of the JSON file containing Daniel's data.
        ashay_file (str): File path of the JSON file containing Ashay's data.

    Returns:
        tuple: A tuple containing two lists, sorted data from Daniel's file and Ashay's file.

    Raises:
        AssertionError: If the two JSON files don't have the same number of patients.

    """

    with open(daniel_file) as json_file:
        data_daniel = json.load(json_file)

    with open(ashay_file) as json_file:
        data_ashay = json.load(json_file)

    # assert they are the same length else throw error
    assert len(data_daniel) == len(
        data_ashay
    ), "The two json files don't have the same number of patients!"

    # sort them by patient_name
    data_daniel = sorted(data_daniel, key=lambda k: k["patient_name"])
    data_ashay = sorted(data_ashay, key=lambda k: k["patient_name"])

    # assert that they are the same
    for i in range(len(data_daniel)):
        assert (
            data_daniel[i]["patient_name"] == data_ashay[i]["patient_name"]
        ), "The two json files don't have the same patients!"

    return data_daniel, data_ashay


def merge_data(data_daniel, data_ashay):
    """Merge data from two sources into a single data structure.

    Args:
        data_daniel (list): Data from Daniel's source.
        data_ashay (list): Data from Ashay's source.

    Returns:
        list: Merged data containing information from both sources.

    """

    merged_data = []

    # iter over patients
    for i in range(len(data_daniel)):

        patient_dict = {}
        patient_dict["patient_name"] = data_daniel[i]["patient_name"]

        frames_to_keep = list(
            set(data_ashay[i]["key_frames"] + data_daniel[i]["key_frames"])
        )
        frames_to_keep = [int(x) for x in frames_to_keep]
        frames_to_keep.sort()
        copy_frames_to_keep = frames_to_keep.copy()

        # if two key frames have a distance of less than 3, keep only one, randomly
        for j in range(1, len(copy_frames_to_keep)):
            if abs(copy_frames_to_keep[j - 1] - copy_frames_to_keep[j]) <= 3:
                frame_to_remove = random.choice(
                    [copy_frames_to_keep[j - 1], copy_frames_to_keep[j]]
                )
                try:
                    frames_to_keep.remove(frame_to_remove)
                except:
                    pass  # might have been removed already

        frames_to_keep = [str(x) for x in frames_to_keep]

        errors_to_keep = list(set(data_ashay[i]["errors"] + data_daniel[i]["errors"]))

        errors_to_keep = [
            x for x in errors_to_keep if x.split("_")[0][1:] in frames_to_keep
        ]
        patient_dict["errors"] = errors_to_keep

        patient_dict["key_frames"] = frames_to_keep

        annotations_dict = {}
        first_data = random.choice([data_daniel[i], data_ashay[i]])
        second_data = data_daniel[i] if first_data == data_ashay[i] else data_ashay[i]
        for key, value in first_data["annotations"].items():
            if key in frames_to_keep:
                annotations_dict.update({key: value})
                patient_dict["annotations"] = annotations_dict

        for key, value in second_data["annotations"].items():
            if key in frames_to_keep:
                annotations_dict.update({key: value})
                patient_dict["annotations"] = annotations_dict

        # if there are two errors in patient_dict["errors"] that have the same frame, keep the one from the second data
        error_frames = [x.split("_")[0][1:] for x in patient_dict["errors"]]
        for error in patient_dict["errors"]:
            # count the occurences of the error in the error_frames list
            if error_frames.count(error.split("_")[0][1:]) > 1:
                # remove if it's in the first data
                if error in first_data["errors"]:
                    patient_dict["errors"].remove(error)

        if len(patient_dict["errors"]) == 0:  # make this smarter after
            patient_dict["errors"] = ["no_error"]

        patient_dict["bounding_box"] = data_daniel[i]["bounding_box"]
        merged_data.append(patient_dict)

    return merged_data


def main():

    data_daniel, data_ashay = load_and_sort(
        "../data/annotations_daniel.json", "../data/annotations_ashay.json"
    )
    merged_data = merge_data(data_daniel, data_ashay)

    with open("../data/annotations.json", "w") as json_file:
        json.dump(merged_data, json_file, indent=4)


if __name__ == "__main__":
    main()
