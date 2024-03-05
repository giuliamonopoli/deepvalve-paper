import json
import os
from collections import Counter
from dataclasses import InitVar, dataclass
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import config
import utils as ut


@dataclass
class Annotation:
    mv_insert_septal: List[List[int]]
    mv_insert_lateral: List[List[int]]
    lv_base_septal: List[List[int]]
    lv_base_lateral: List[List[int]]
    leaflet_septal: List[List[int]]
    leaflet_lateral: List[List[int]]


@dataclass
class PatientData:
    patient_name: str
    key_frames: List[str]
    annotations: Dict[str, Annotation]
    bounding_box: List[int]
    flags: InitVar[List[int]] = None

    def __post_init__(self, flags):
        if self.flags is None:
            self.flags = self.encoded_flag(self.patient_name)

    def encoded_flag(self, patient_name):
        file_path = config.data_frame_path
        result_df = pd.read_csv(file_path)
        patient_df = result_df[result_df["patient_name"] == patient_name]
        flags = patient_df[["fp_1", "fp_3", "ff_1", "ff_2", "ff_3"]].values.tolist()
        return flags[0]


class PreprocessDataset:
    def __init__(
        self,
        raw_imgs_path=config.raw_imgs_path,
        annotation_folder=config.annotation_folder,
    ):
        """
        Initialize the dataset.

        Args:
            data_path (str): Path to the folder containing the data.
            annotation_path (str): Path to the annotation file.
        """

        self.raw_imgs_path = raw_imgs_path
        self.annotation_folder = annotation_folder

    def get_annotation_json(self):
        """
        Parse all the text files in the specified folder and return the annotation data.

        Args:
            folder_path (str): Path to the folder containing annotation files.

        Returns:
            list: List of annotation data dictionaries.
        """
        with open(config.annotation_json, "r") as json_file:
            annotation_data_list = json.load(json_file)
            return annotation_data_list

    def rescale_annotations(self, annotation_data_list, normalize_keypts=True):
        """
        Rescale the annotations based on the maximum bounding box dimensions and
        between 0 and 1.

        Args:
            annotation_data_list (list): List of annotation data dictionaries.
            normalize_keypts (bool): Whether to normalize the keypoints between 0 and 1.

        Returns:
            list: List of rescaled annotation data dictionaries.
        """

        max_width = 448  # max([x["bounding_box"][2] for x in annotation_data_list])
        max_height = 448  # max([x["bounding_box"][3] for x in annotation_data_list])

        for annotation_data in annotation_data_list:
            for frame in annotation_data["annotations"]:
                for key in annotation_data["annotations"][frame]:
                    width = annotation_data["bounding_box"][2]
                    height = annotation_data["bounding_box"][3]

                    scale_x = max_width / width
                    scale_y = max_height / height

                    # translation scalling (the one done in cv2.resize)
                    for i, annot in enumerate(
                        annotation_data["annotations"][frame][key]
                    ):
                        # zoom scalling
                        annotation_data["annotations"][frame][key][i][1] = (
                            annot[1] - annotation_data["bounding_box"][0] + width // 2
                        )  # x
                        annotation_data["annotations"][frame][key][i][2] = (
                            annot[2] - annotation_data["bounding_box"][1] + height // 2
                        )  # y

                        # scale scalling
                        annotation_data["annotations"][frame][key][i][1] = int(
                            annotation_data["annotations"][frame][key][i][1] * scale_x
                        )  # x
                        annotation_data["annotations"][frame][key][i][2] = int(
                            annotation_data["annotations"][frame][key][i][2] * scale_y
                        )  # y

                        # normalize scalling
                        if normalize_keypts:
                            annotation_data["annotations"][frame][key][i][1] = (
                                annotation_data["annotations"][frame][key][i][1]
                                / max_width
                            )
                            annotation_data["annotations"][frame][key][i][2] = (
                                annotation_data["annotations"][frame][key][i][2]
                                / max_height
                            )

        return annotation_data_list

    def standardize_annotations(self, annotation_data_list):
        keys = [
            "leaflet_septal",
            "leaflet_lateral",
            "mv_insert_septal",
            "mv_insert_lateral",
            "lv_base_lateral",
            "lv_base_septal",
        ]
        for annotation_data in annotation_data_list:
            for frame in annotation_data["annotations"]:
                for key in keys:
                    annotation_data["annotations"][frame][key] = np.array(
                        [
                            sublist[1:]
                            for sublist in annotation_data["annotations"][frame][key]
                        ]
                    )

                    if "leaflet" in key:
                        annotation_data["annotations"][frame][key] = ut.get_spline_pts(
                            annotation_data["annotations"][frame][key], 10
                        )  # 10 points per spline

        return annotation_data_list

    def get_patient_data_list(
        self, patient_names: List[str], data_list: List[PatientData]
    ) -> List[PatientData]:
        return [
            patient_data
            for patient_data in data_list
            if patient_data.patient_name in patient_names
        ]

    def load_dataset(self, normalize_keypts=True):

        json_data = self.get_annotation_json()
        json_data = self.rescale_annotations(
            json_data, normalize_keypts=normalize_keypts
        )
        json_data = self.standardize_annotations(json_data)
        patient_data = [PatientData(**json_data[i]) for i, _ in enumerate(json_data)]

        return patient_data

    @staticmethod
    def get_kframes_and_annot_from_mhd(list_p):
        lst_of_matrix_imgs = []
        lst_of_annotation_imgs = []
        lst_of_patient_names = []
        raw_imgs_path = config.raw_imgs_path
        lst_of_image_ids = []
        for patient in list_p:
            key_frames = patient.key_frames
            center_x, center_y, width, height = patient.bounding_box
            max_width = 448
            max_height = 448

            for structure_folder in os.listdir(
                os.path.join(raw_imgs_path, patient.patient_name[:-2])
            ):
                if structure_folder.startswith("LA"):
                    for file in os.listdir(
                        os.path.join(
                            raw_imgs_path, patient.patient_name[:-2], structure_folder
                        )
                    ):
                        file_frame = file.split("_")[1].split(".")[0]

                        if file.endswith(".mhd") and file_frame in key_frames:
                            image_id = f"{patient.patient_name}_{file_frame}"  # Unique identifier for each image
                            lst_of_image_ids.append(image_id)
                            itkimage = sitk.ReadImage(
                                os.path.join(
                                    raw_imgs_path,
                                    patient.patient_name[:-2],
                                    structure_folder,
                                    file,
                                )
                            )
                            array_img = sitk.GetArrayFromImage(itkimage)
                            cropped_img = array_img[
                                center_y - height // 2 : center_y + height // 2,
                                center_x - width // 2 : center_x + width // 2,
                            ]
                            resized_img = cv2.resize(
                                cropped_img,
                                (max_width, max_height),
                                interpolation=cv2.INTER_CUBIC,
                            )

                            lst_of_matrix_imgs.append(resized_img)
                            lst_of_annotation_imgs.append(
                                patient.annotations[file_frame]
                            )
                            lst_of_patient_names.append(patient.patient_name)

        return (
            lst_of_matrix_imgs,
            lst_of_annotation_imgs,
            lst_of_patient_names,
            lst_of_image_ids,
        )


class DeepValveDataset(Dataset):
    def __init__(self, mode="train", transform=None, normalize_keypts=True):
        assert mode in [
            "train",
            "val",
            "test",
        ], "Mode should be either 'train', 'val', or 'test'"

        self.transform = transform

        # Load and preprocess dataset
        preprocess = PreprocessDataset()
        patient_data = preprocess.load_dataset(normalize_keypts=normalize_keypts)

        # Split data
        data_splitter = DataSplitter()
        X_train, X_val, X_test = data_splitter.split_data_balanced()

        # Select the appropriate dataset based on the mode
        if mode == "train":
            self.patient_data_list = preprocess.get_patient_data_list(
                X_train, patient_data
            )
        elif mode == "val":
            self.patient_data_list = preprocess.get_patient_data_list(
                X_val, patient_data
            )
        else:  # mode == "test"
            self.patient_data_list = preprocess.get_patient_data_list(
                X_test, patient_data
            )

        # Extract images, annotations, and patient names
        (
            self.images,
            self.landmarks,
            self.patient_names,
            self.image_ids,
        ) = PreprocessDataset.get_kframes_and_annot_from_mhd(self.patient_data_list)
        print(f"{mode.capitalize()} set: {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        annotation = self.landmarks[idx]
        patient_name = self.patient_names[idx]
        image_id = self.image_ids[idx]
        sample = {
            "image": image,
            "landmarks": annotation,
            "patient_name": patient_name,
            "image_id": image_id,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class DataSplitter:
    def __init__(
        self, file_path=config.data_frame_path, test_size=0.25, random_state=47
    ):
        self.file_path = file_path
        self.test_size = test_size
        self.random_state = random_state

    def load_data(self):
        df = pd.read_csv(self.file_path)
        return (
            df["patient_name"].tolist(),
            df[["fp_1", "fp_3", "ff_1", "ff_2", "ff_3"]].values.tolist(),
        )

    def get_combinations_counts(self, flags_list):
        class_counts = Counter(map(tuple, flags_list))
        filtered_classes = [
            list(key) for key, value in class_counts.items() if value > 1
        ]
        one_class = [list(key) for key, value in class_counts.items() if value == 1]
        return filtered_classes, one_class

    def split_data(
        self,
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
        elif step == "two":
            split_index = (len(patients_one) + 1) // 2
            X_train = np.append(X_train, patients_one[:split_index])
            y_train = np.append(y_train, flags_one[:split_index], axis=0)
            X_test = np.append(X_test, patients_one[split_index:])
            y_test = np.append(y_test, flags_one[split_index:], axis=0)

        return X_train, X_test, y_train, y_test

    def process_and_split_data(self, test_size=0.25, random_state=47):

        patient_names, flags_list = self.load_data()
        filtered_classes, one_class = self.get_combinations_counts(flags_list)
        return self.split_data(
            patient_names,
            flags_list,
            filtered_classes,
            one_class,
            test_size=test_size,
            random_state=random_state,
        )

    def split_data_balanced(self):
        X_train, X_test, _, y_test = self.process_and_split_data()

        filtered_classes_t, one_class_t = self.get_combinations_counts(y_test)
        X_test, X_val, y_test, _ = self.split_data(
            X_test,
            y_test,
            filtered_classes_t,
            one_class_t,
            test_size=0.3,
            step="two",
            random_state=47,
        )

        return X_train, X_val, X_test


def get_data_loader(
    mode="train", batch_size=4, transform=None, normalize_keypts=True, num_workers=8
):
    dataset = DeepValveDataset(
        mode=mode, transform=transform, normalize_keypts=normalize_keypts
    )
    shuffle = mode == "train"
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
