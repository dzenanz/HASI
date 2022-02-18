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

# Purpose: Overall segmentation pipeline

import itk
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
    print(f'List of cases to process: {data_list}')

    pose = read_slicer_fiducials(root_dir + bone + '/Pose.fcsv')

    # now load atlas landmarks, axis-aligning transform, image, and segmentation
    # atlas_landmarks = read_slicer_fiducials(root_dir + bone + '/' + atlas + '.fcsv')  # not needed
    atlas_aa_transform = itk.transformread(root_dir + bone + '/' + atlas + '-landmarks.tfm')
    atlas_aa_transform = atlas_aa_transform[0]  # turn this from a list into a transform
    atlas_aa_inverse_transform = rigid_transform_type.New()
    atlas_aa_transform.GetInverse(atlas_aa_inverse_transform)
    # atlas_aa_landmarks = pose

    atlas_aa_image = itk.imread(root_dir + bone + '/' + atlas + '-AA.nrrd', pixel_type=itk.F)
    # atlas_aa_segmentation = itk.imread(root_dir + bone + '/' + atlas + '-AA.seg.nrrd',
    #                                    pixel_type=itk.VariableLengthVector[itk.UC])
    atlas_aa_segmentation = itk.imread(root_dir + bone + '/' + atlas + '-AA-label.nrrd', pixel_type=itk.F)

    # now go through all the cases, doing main processing
    for case in data_list:
        print(f'Processing case {case}')
        case_landmarks = read_slicer_fiducials(root_dir + bone + '/' + case + '.fcsv')
        case_to_pose = register_landmarks(pose, case_landmarks)
        pose_to_case = rigid_transform_type.New()
        case_to_pose.GetInverse(pose_to_case)
        # atlas_to_case = register_landmarks(case_landmarks, atlas_landmarks)
        case_image = itk.imread(root_dir + 'Data/' + case + '.nrrd', pixel_type=itk.F)

        # write atlas_to_case transform to file - needed for initializing Elastix registration
        affine_pose_to_case = itk.AffineTransform[itk.D, 3].New()
        affine_pose_to_case.SetCenter(pose_to_case.GetCenter())
        affine_pose_to_case.SetMatrix(pose_to_case.GetMatrix())
        affine_pose_to_case.SetOffset(pose_to_case.GetOffset())
        atlas_to_case_filename = root_dir + bone + '/' + case + '-' + atlas + '.tfm'
        itk.transformwrite([affine_pose_to_case], atlas_to_case_filename)
        out_elastix_transform = open(atlas_to_case_filename + '.txt', "w")
        out_elastix_transform.writelines(['(Transform "File")\n',
                                          '(TransformFileName "' + case + '-' + atlas + '.tfm")'])
        out_elastix_transform.close()

        # Construct elastix parameter map
        parameter_object = itk.ParameterObject.New()
        resolutions = 4
        parameter_map_rigid = parameter_object.GetDefaultParameterMap('rigid', resolutions)
        parameter_object.AddParameterMap(parameter_map_rigid)
        parameter_map_bspline = parameter_object.GetDefaultParameterMap("bspline", resolutions, 1.0)
        parameter_object.AddParameterMap(parameter_map_bspline)
        parameter_object.SetParameter("DefaultPixelValue", "-1024")

        print('Starting atlas registration')
        registered, elastix_transform = itk.elastix_registration_method(
            case_image, atlas_aa_image,
            parameter_object=parameter_object,
            initial_transform_parameter_file_name=atlas_to_case_filename + '.txt',
            log_to_console=True,
        )
        registered_filename = root_dir + bone + '/' + case + '-' + atlas + '-reg.nrrd'
        print(f'Writing registered image to file {registered_filename}')
        itk.imwrite(registered.astype(itk.SS), registered_filename)
        print(elastix_transform)
        # serialize each parameter map to a file.
        for index in range(elastix_transform.GetNumberOfParameterMaps()):
            parameter_map = elastix_transform.GetParameterMap(index)
            elastix_transform.WriteParameterFile(
                parameter_map,
                root_dir + bone + '/' + case + '-' + atlas + '.{0}.txt'.format(index))

        print('Running transformix')
        elastix_transform.SetParameter('FinalBSplineInterpolationOrder', '0')
        result_image_transformix = itk.transformix_filter(
            atlas_aa_segmentation,
            elastix_transform,
            # reference image?
        )
        result_image = result_image_transformix.astype(itk.UC)
        registered_label_file = root_dir + bone + '/' + case + '-' + atlas + '-label.nrrd'
        print(f'Writing deformed atlas to {registered_label_file}')
        itk.imwrite(result_image, registered_label_file, compression=True)

        # # now use the transform to transfer atlas labels to the case under observation
        # nearest_interpolator = itk.NearestNeighborInterpolateImageFunction.New(atlas_aa_segmentation)
        # atlas_labels_transformed = itk.resample_image_filter(atlas_aa_segmentation,
        #                                                      use_reference_image=True,
        #                                                      reference_image=case_image,
        #                                                      transform=elastix_transform,
        #                                                      interpolator=nearest_interpolator)
        # itk.imwrite(atlas_labels_transformed, 'case-label.nrrd', compression=True)

        print('Computing morphometry features')
        morphometry_filter = itk.BoneMorphometryFeaturesFilter[type(atlas_aa_image)].New(case_image)
        morphometry_filter.SetMaskImage(result_image)
        morphometry_filter.Update()
        print('BVTV', morphometry_filter.GetBVTV())
        print('TbN', morphometry_filter.GetTbN())
        print('TbTh', morphometry_filter.GetTbTh())
        print('TbSp', morphometry_filter.GetTbSp())
        print('BSBV', morphometry_filter.GetBSBV())

        print('Generate the mesh from the segmented case')
        padded_segmentation = itk.constant_pad_image_filter(
            result_image,
            PadUpperBound=1,
            PadLowerBound=1,
            Constant=0
        )

        mesh = itk.cuberille_image_to_mesh_filter(padded_segmentation)
        mesh_filename = root_dir + bone + '/' + case + '-' + atlas + '.vtk'
        print(f'Writing the mesh to file {mesh_filename}')
        itk.meshwrite(mesh, mesh_filename)

        canonical_pose_mesh = itk.transform_mesh_filter(
            mesh,
            transform=pose_to_case
        )
        canonical_pose_filename = root_dir + bone + '/' + case + '-' + atlas + '.obj'
        print(f'Writing canonical pose mesh to {canonical_pose_filename}')
        itk.meshwrite(canonical_pose_mesh, canonical_pose_filename)
        print(f'Done processing case {case}')


# main code
main_processing('../../', 'Tibia', '901-R')
main_processing('../../', 'Tibia', '901-L')
main_processing('../../', 'Femur', '907-L')
