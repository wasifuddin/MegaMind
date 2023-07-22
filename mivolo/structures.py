import math
import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from mivolo.data.misc import assign_faces, box_iou, cropout_black_parts
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils.plotting import Annotator, colors

# because of ultralytics bug it is important to unset CUBLAS_WORKSPACE_CONFIG after the module importing
os.unsetenv("CUBLAS_WORKSPACE_CONFIG")


class PersonAndFaceCrops:
    def __init__(self):
        # int: index of person along results
        self.crops_persons: Dict[int, np.ndarray] = {}

        # int: index of face along results
        self.crops_faces: Dict[int, np.ndarray] = {}

        # int: index of face along results
        self.crops_faces_wo_body: Dict[int, np.ndarray] = {}

        # int: index of person along results
        self.crops_persons_wo_face: Dict[int, np.ndarray] = {}

    def _add_to_output(
        self, crops: Dict[int, np.ndarray], out_crops: List[np.ndarray], out_crop_inds: List[Optional[int]]
    ):
        inds_to_add = list(crops.keys())
        crops_to_add = list(crops.values())
        out_crops.extend(crops_to_add)
        out_crop_inds.extend(inds_to_add)

    def _get_all_faces(
        self, use_persons: bool, use_faces: bool
    ) -> Tuple[List[Optional[int]], List[Optional[np.ndarray]]]:
        """
        Returns
            if use_persons and use_faces
                faces: faces_with_bodies + faces_without_bodies + [None] * len(crops_persons_wo_face)
            if use_persons and not use_faces
                faces: [None] * n_persons
            if not use_persons and use_faces:
                faces: faces_with_bodies + faces_without_bodies
        """

        def add_none_to_output(faces_inds, faces_crops, num):
            faces_inds.extend([None for _ in range(num)])
            faces_crops.extend([None for _ in range(num)])

        faces_inds: List[Optional[int]] = []
        faces_crops: List[Optional[np.ndarray]] = []

        if not use_faces:
            add_none_to_output(faces_inds, faces_crops, len(self.crops_persons) + len(self.crops_persons_wo_face))
            return faces_inds, faces_crops

        self._add_to_output(self.crops_faces, faces_crops, faces_inds)
        self._add_to_output(self.crops_faces_wo_body, faces_crops, faces_inds)

        if use_persons:
            add_none_to_output(faces_inds, faces_crops, len(self.crops_persons_wo_face))

        return faces_inds, faces_crops

    def _get_all_bodies(
        self, use_persons: bool, use_faces: bool
    ) -> Tuple[List[Optional[int]], List[Optional[np.ndarray]]]:
        """
        Returns
            if use_persons and use_faces
                persons: bodies_with_faces + [None] * len(faces_without_bodies) + bodies_without_faces
            if use_persons and not use_faces
                persons: bodies_with_faces + bodies_without_faces
            if not use_persons and use_faces
                persons: [None] * n_faces
        """

        def add_none_to_output(bodies_inds, bodies_crops, num):
            bodies_inds.extend([None for _ in range(num)])
            bodies_crops.extend([None for _ in range(num)])

        bodies_inds: List[Optional[int]] = []
        bodies_crops: List[Optional[np.ndarray]] = []

        if not use_persons:
            add_none_to_output(bodies_inds, bodies_crops, len(self.crops_faces) + len(self.crops_faces_wo_body))
            return bodies_inds, bodies_crops

        self._add_to_output(self.crops_persons, bodies_crops, bodies_inds)
        if use_faces:
            add_none_to_output(bodies_inds, bodies_crops, len(self.crops_faces_wo_body))

        self._add_to_output(self.crops_persons_wo_face, bodies_crops, bodies_inds)

        return bodies_inds, bodies_crops

    def get_faces_with_bodies(self, use_persons: bool, use_faces: bool):
        """
        Return
            faces: faces_with_bodies, faces_without_bodies, [None] * len(crops_persons_wo_face)
            persons: bodies_with_faces, [None] * len(faces_without_bodies), bodies_without_faces
        """

        bodies_inds, bodies_crops = self._get_all_bodies(use_persons, use_faces)
        faces_inds, faces_crops = self._get_all_faces(use_persons, use_faces)

        return (bodies_inds, bodies_crops), (faces_inds, faces_crops)

    def save(self, out_dir="output"):
        ind = 0
        os.makedirs(out_dir, exist_ok=True)
        for crops in [self.crops_persons, self.crops_faces, self.crops_faces_wo_body, self.crops_persons_wo_face]:
            for crop in crops.values():
                if crop is None:
                    continue
                out_name = os.path.join(out_dir, f"{ind}_crop.jpg")
                cv2.imwrite(out_name, crop)
                ind += 1


class PersonAndFaceResult:
    def __init__(self, results: Results):

        self.yolo_results = results
        names = set(results.names.values())
        assert "person" in names and "face" in names

        # initially no faces and persons are associated to each other
        self.face_to_person_map: Dict[int, Optional[int]] = {ind: None for ind in self.get_bboxes_inds("face")}
        self.unassigned_persons_inds: List[int] = self.get_bboxes_inds("person")

        n_objects = len(self.yolo_results.boxes)
        self.ages: List[Optional[float]] = [None for _ in range(n_objects)]
        self.genders: List[Optional[str]] = [None for _ in range(n_objects)]
        self.gender_scores: List[Optional[float]] = [None for _ in range(n_objects)]

    @property
    def n_objects(self) -> int:
        return len(self.yolo_results.boxes)

    def get_bboxes_inds(self, category: str) -> List[int]:
        bboxes: List[int] = []
        for ind, det in enumerate(self.yolo_results.boxes):
            name = self.yolo_results.names[int(det.cls)]
            if name == category:
                bboxes.append(ind)

        return bboxes

    def get_distance_to_center(self, bbox_ind: int) -> float:
        """
        Calculate euclidian distance between bbox center and image center.
        """
        im_h, im_w = self.yolo_results[bbox_ind].orig_shape
        x1, y1, x2, y2 = self.get_bbox_by_ind(bbox_ind).cpu().numpy()
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        dist = math.dist([center_x, center_y], [im_w / 2, im_h / 2])
        return dist

    def plot(
        self,
        conf=False,
        line_width=None,
        font_size=None,
        font="Arial.ttf",
        pil=False,
        img=None,
        labels=True,
        boxes=True,
        probs=True,
        ages=True,
        genders=True,
        gender_probs=False,
    ):
        """
        Plots the detection results on an input RGB image. Accepts a numpy array (cv2) or a PIL Image.
        Args:
            conf (bool): Whether to plot the detection confidence score.
            line_width (float, optional): The line width of the bounding boxes. If None, it is scaled to the image size.
            font_size (float, optional): The font size of the text. If None, it is scaled to the image size.
            font (str): The font to use for the text.
            pil (bool): Whether to return the image as a PIL Image.
            img (numpy.ndarray): Plot to another image. if not, plot to original image.
            labels (bool): Whether to plot the label of bounding boxes.
            boxes (bool): Whether to plot the bounding boxes.
            probs (bool): Whether to plot classification probability
        Returns:
            (numpy.ndarray): A numpy array of the annotated image.
        """

        # return self.yolo_results.plot()
        colors_by_ind = {}
        for face_ind, person_ind in self.face_to_person_map.items():
            if person_ind is not None:
                colors_by_ind[face_ind] = face_ind + 2
                colors_by_ind[person_ind] = face_ind + 2
            else:
                colors_by_ind[face_ind] = 0
        for person_ind in self.unassigned_persons_inds:
            colors_by_ind[person_ind] = 1

        names = self.yolo_results.names
        annotator = Annotator(
            deepcopy(self.yolo_results.orig_img if img is None else img),
            line_width,
            font_size,
            font,
            pil,
            example=names,
        )
        pred_boxes, show_boxes = self.yolo_results.boxes, boxes
        pred_probs, show_probs = self.yolo_results.probs, probs

        if pred_boxes and show_boxes:
            for bb_ind, (d, age, gender, gender_score) in enumerate(
                zip(pred_boxes, self.ages, self.genders, self.gender_scores)
            ):

                c, conf, id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
                name = ("" if id is None else f"id:{id} ") + names[c]
                label = (f"{name} {conf:.2f}" if conf else name) if labels else None
                if ages and age is not None:
                    label += f" {age:.1f}"
                if genders and gender is not None:
                    label += f" {'F' if gender == 'female' else 'M'}"
                if gender_probs and gender_score is not None:
                    label += f" ({gender_score:.1f})"
                annotator.box_label(d.xyxy.squeeze(), label, color=colors(colors_by_ind[bb_ind], True))

        if pred_probs is not None and show_probs:
            text = f"{', '.join(f'{names[j] if names else j} {pred_probs.data[j]:.2f}' for j in pred_probs.top5)}, "
            annotator.text((32, 32), text, txt_color=(255, 255, 255))  # TODO: allow setting colors

        return annotator.result()

    def get_bbox_by_ind(self, ind: int) -> torch.tensor:
        return self.yolo_results.boxes[ind].xyxy.squeeze().type(torch.int32)

    def set_age(self, ind: Optional[int], age: float):
        if ind is not None:
            self.ages[ind] = age

    def set_gender(self, ind: Optional[int], gender: str, gender_score: float):
        if ind is not None:
            self.genders[ind] = gender
            self.gender_scores[ind] = gender_score

    def associate_faces_with_persons(self):
        face_bboxes_inds: List[int] = self.get_bboxes_inds("face")
        person_bboxes_inds: List[int] = self.get_bboxes_inds("person")

        face_bboxes: List[torch.tensor] = [self.get_bbox_by_ind(ind) for ind in face_bboxes_inds]
        person_bboxes: List[torch.tensor] = [self.get_bbox_by_ind(ind) for ind in person_bboxes_inds]

        self.face_to_person_map = {ind: None for ind in face_bboxes_inds}
        assigned_faces, unassigned_persons_inds = assign_faces(person_bboxes, face_bboxes)

        for face_ind, person_ind in enumerate(assigned_faces):
            face_ind = face_bboxes_inds[face_ind]
            person_ind = person_bboxes_inds[person_ind] if person_ind is not None else None
            self.face_to_person_map[face_ind] = person_ind

        self.unassigned_persons_inds = [person_bboxes_inds[person_ind] for person_ind in unassigned_persons_inds]

        # print(f"face_to_person_map: {self.face_to_person_map}")
        # print(f"unassigned_persons_inds: {self.unassigned_persons_inds}")

    def crop_object(
        self, full_image: np.ndarray, ind: int, cut_other_classes: Optional[List[str]] = None
    ) -> Optional[np.ndarray]:

        IOU_THRESH = 0.000001
        MIN_PERSON_CROP_AFTERCUT_RATIO = 0.4
        CROP_ROUND_RATE = 0.3
        MIN_PERSON_SIZE = 50

        obj_bbox = self.get_bbox_by_ind(ind)
        x1, y1, x2, y2 = obj_bbox
        cur_cat = self.yolo_results.names[int(self.yolo_results.boxes[ind].cls)]
        # get crop of face or person
        obj_image = full_image[y1:y2, x1:x2].copy()
        crop_h, crop_w = obj_image.shape[:2]

        if cur_cat == "person" and (crop_h < MIN_PERSON_SIZE or crop_w < MIN_PERSON_SIZE):
            return None

        if not cut_other_classes:
            return obj_image

        # calc iou between obj_bbox and other bboxes
        other_bboxes: List[torch.tensor] = [
            self.get_bbox_by_ind(other_ind) for other_ind in range(len(self.yolo_results.boxes))
        ]

        iou_matrix = box_iou(torch.stack([obj_bbox]), torch.stack(other_bboxes)).cpu().numpy()[0]

        # cut out other objects in case of intersection
        for other_ind, (det, iou) in enumerate(zip(self.yolo_results.boxes, iou_matrix)):
            other_cat = self.yolo_results.names[int(det.cls)]
            if ind == other_ind or iou < IOU_THRESH or other_cat not in cut_other_classes:
                continue
            o_x1, o_y1, o_x2, o_y2 = det.xyxy.squeeze().type(torch.int32)

            # remap current_person_bbox to reference_person_bbox coordinates
            o_x1 = max(o_x1 - x1, 0)
            o_y1 = max(o_y1 - y1, 0)
            o_x2 = min(o_x2 - x1, crop_w)
            o_y2 = min(o_y2 - y1, crop_h)

            if other_cat != "face":
                if (o_y1 / crop_h) < CROP_ROUND_RATE:
                    o_y1 = 0
                if ((crop_h - o_y2) / crop_h) < CROP_ROUND_RATE:
                    o_y2 = crop_h
                if (o_x1 / crop_w) < CROP_ROUND_RATE:
                    o_x1 = 0
                if ((crop_w - o_x2) / crop_w) < CROP_ROUND_RATE:
                    o_x2 = crop_w

            obj_image[o_y1:o_y2, o_x1:o_x2] = 0

        obj_image, remain_ratio = cropout_black_parts(obj_image, CROP_ROUND_RATE)
        if remain_ratio < MIN_PERSON_CROP_AFTERCUT_RATIO:
            return None

        return obj_image

    def collect_crops(self, image) -> PersonAndFaceCrops:

        crops_data = PersonAndFaceCrops()
        for face_ind, person_ind in self.face_to_person_map.items():
            face_image = self.crop_object(image, face_ind, cut_other_classes=[])

            if person_ind is None:
                crops_data.crops_faces_wo_body[face_ind] = face_image
                continue

            person_image = self.crop_object(image, person_ind, cut_other_classes=["face", "person"])

            crops_data.crops_faces[face_ind] = face_image
            crops_data.crops_persons[person_ind] = person_image

        for person_ind in self.unassigned_persons_inds:
            person_image = self.crop_object(image, person_ind, cut_other_classes=["face", "person"])
            crops_data.crops_persons_wo_face[person_ind] = person_image

        # uncomment to save preprocessed crops
        # crops_data.save()
        return crops_data
