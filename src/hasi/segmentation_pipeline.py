#!/usr/bin/env python3

# Copyright NumFOCUS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Purpose: Python functions for shape classification and distance
#           analysis with DWD classifier
import itk
import numpy as np
import os
from pathlib import Path


def sorted_file_list(folder, extension):
    file_list = []
    for filename in os.listdir(folder):
        if filename.endswith(extension):
            filename = os.path.splitext(filename)[0]
            filename = Path(filename).stem
            file_list.append(filename)

    file_list = list(set(file_list))  # remove duplicates
    file_list.sort()
    return file_list


def read_slicer_fiducials(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    lines.pop(0)  # Markups fiducial file version = 4.11

    coordinate_system = lines[0][-4:-1]
    if coordinate_system == 'RAS' or coordinate_system[-1:] == '0':
        ras = True
    elif coordinate_system == 'LPS' or coordinate_system[-1:] == '1':
        ras = False
    elif coordinate_system == 'IJK' or coordinate_system[-1:] == '2':
        raise RuntimeError('Fiducials file with IJK coordinates is not supported')
    else:
        raise RuntimeError('Unrecognized coordinate system: ' + coordinate_system)

    lines.pop(0)  # CoordinateSystem = 0
    lines.pop(0)  # columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID

    fiducials = []
    for line in lines:
        e = line.split(',', 4)
        p = itk.Point[itk.D, 3]()
        for i in range(3):
            p[i] = float(e[i + 1])
        fiducials.append(p)

    return fiducials


rigid_transform_type = itk.VersorRigid3DTransform[itk.D]

def register_landmarks(atlas_landmarks, input_landmarks):
    transform_type = itk.Transform[itk.D, 3, 3]
    landmark_transformer = itk.LandmarkBasedTransformInitializer[transform_type].New()
    rigid_transform = rigid_transform_type.New()
    landmark_transformer.SetFixedLandmarks(atlas_landmarks)
    landmark_transformer.SetMovingLandmarks(input_landmarks)
    landmark_transformer.SetTransform(rigid_transform)
    landmark_transformer.InitializeTransform()

    # force rotation to be around center of femur head
    rigid_transform.SetCenter(atlas_landmarks[0])
    # and make sure that the other corresponding point maps to it perfectly
    rigid_transform.SetTranslation(input_landmarks[0] - atlas_landmarks[0])

    return rigid_transform


def main_processing(root_dir, bone, atlas):
    data_list = sorted_file_list(root_dir + 'Data', '.nrrd')
    if atlas not in data_list:
        raise RuntimeError("Missing data file for the atlas")
    data_list.remove(atlas)

    landmarks_list = sorted_file_list(root_dir + bone, '.fcsv')
    if atlas not in landmarks_list:
        raise RuntimeError("Missing landmarks file for the atlas")
    landmarks_list.remove(atlas)
    if 'Pose' not in landmarks_list:
        raise RuntimeError("Missing Pose.fcsv file")
    landmarks_list.remove('Pose')

    # check if there are any discrepancies
    if data_list != landmarks_list:
        print('There is a discrepancy between data_list and landmarks_list')
        print('data_list:', data_list)
        print('landmarks_list:', landmarks_list)

    pose = read_slicer_fiducials(root_dir + bone + '/Pose.fcsv')

    # now load atlas landmarks, axis-aligning transform, image, and segmentation
    atlas_landmarks = read_slicer_fiducials(root_dir + bone + '/' + atlas + '.fcsv')
    atlas_aa_transform = itk.transformread(root_dir + bone + '/' + atlas + '-landmarks.tfm')
    atlas_aa_transform = atlas_aa_transform[0]  # turn this from a list into a transform
    atlas_aa_inverse_transform = rigid_transform_type.New()
    atlas_aa_transform.GetInverse(atlas_aa_inverse_transform)
    atlas_aa_landmarks = [atlas_aa_inverse_transform.TransformPoint(l) for l in atlas_landmarks]

    atlas_aa_image = itk.imread(root_dir + bone + '/' + atlas + '-AA.nrrd')
    atlas_aa_segmentation = itk.imread(root_dir + bone + '/' + atlas + '-AA.seg.nrrd',
                                       pixel_type=itk.VariableLengthVector[itk.UC])

    # now go through all the cases, doing main processing
    for case in data_list:
        case_landmarks = read_slicer_fiducials(root_dir + bone + '/' + case + '.fcsv')
        transform = register_landmarks(pose, case_landmarks)


# main code
main_processing('../../', 'Tibia', '901-R')
main_processing('../../', 'Femur', '907-L')
