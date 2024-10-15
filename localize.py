#!/usr/bin/env python3

# Copyright (c) 2022, Vojtech Panek and Zuzana Kukelova and Torsten Sattler
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import yaml
import immatch
# import pycolmap

import numpy as np
from immatch.utils import plot_matches

import collections
import os
import struct

from tqdm import tqdm

import scipy.sparse
import scipy.spatial.distance

import meshloc

import argparse

# Defines helper function

#### Code taken from Colmap:
# from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras



def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                # xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       # tuple(map(float, elems[1::3]))])
                # point3D_ids = np.array(tuple(map(int, elems[2::3])))
                # images[image_id] = Image(
                    # id=image_id, qvec=qvec, tvec=tvec,
                    # camera_id=camera_id, name=image_name,
                    # xys=xys, point3D_ids=point3D_ids)
                images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys={}, point3D_ids={})
    return images


def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            # images[image_id] = Image(
                # id=image_id, qvec=qvec, tvec=tvec,
                # camera_id=camera_id, name=image_name,
                # xys=xys, point3D_ids=point3D_ids)
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys={}, point3D_ids={})
    return images

def detect_model_format(path, ext):
    if os.path.isfile(os.path.join(path, "cameras"  + ext)) and \
       os.path.isfile(os.path.join(path, "images"   + ext)):
        print("Detected model format: '" + ext + "'")
        return True

    return False


def read_model(path, ext=""):
    # try to detect the extension automatically
    if ext == "":
        if detect_model_format(path, ".bin"):
            ext = ".bin"
        elif detect_model_format(path, ".txt"):
            ext = ".txt"
        else:
            print("Provide model format: '.bin' or '.txt'")
            return

    if ext == ".txt":
        cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        images = read_images_text(os.path.join(path, "images" + ext))
    else:
        cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
        images = read_images_binary(os.path.join(path, "images" + ext))
    return cameras, images


####
# Main function


def main():
    parser = argparse.ArgumentParser(description="Localization against 3D model")
    parser.add_argument("--db_image_dir", type=str, help="Directory with database images")
    parser.add_argument("--colmap_model_dir", type=str, help="Directory with colmap model")
    parser.add_argument("--db_depth_image_dir", type=str, help="Directory with database image depth maps")
    parser.add_argument("--method_name", type=str, help="Method name")
    parser.add_argument("--method_config", type=str, help="Method config name")
    parser.add_argument("--method_string", type=str, help="String name for method used for output")
    parser.add_argument("--out_prefix", type=str, help="Prefix of output file (including path)")
    parser.add_argument("--query_list", type=str, help="Text file containing the query names and intrinsics")
    parser.add_argument("--query_dir", type=str, help="Directory with query images")
    parser.add_argument("--retrieval_pairs", type=str, help="Text file with results of retrieval")
    parser.add_argument("--top_k", type=int, help="Number of top-ranked images to use")
    parser.add_argument("--reproj_error", type=float, default=12.0, help="Reprojection error threshold for RANSAC")
    parser.add_argument("--use_orig_db_images", action="store_true", help="Use real or rendered database images.")
    parser.add_argument("--triangulate", action="store_true", help="Use triangulation instead of 3D points from depth maps.")
    parser.add_argument("--merge_3D_points", action="store_true", help="If multiple 3D points are available per query feature, whether to select one or not.")
    parser.add_argument("--cluster_keypoints", action="store_true", help="Whether to cluster keypoints. Only applicable for patch2pix at the moment.")
    parser.add_argument("--covisibility_filtering", action="store_true", help="Use covisibility filtering or not.")
    parser.add_argument("--all_matches_ransac", action="store_true", help="Use all possible 2D-3D matches in RANSAC.")
    parser.add_argument("--min_ransac_iterations", type=int, default=1000, required=False, help="Minimum number of RANSAC iterations.")
    parser.add_argument("--max_ransac_iterations", type=int, default=100000, required=False, help="Maximum number of RANSAC iterations.")
    parser.add_argument("--max_side_length", type=int, default=800, required=False, help="Maximum side length to use for queries, -1 for original resolution")
    parser.add_argument("--ransac_type", type=str, default="MSAC", required=False, help="RANSAC type: MSAC, EFFSAC, or PYCOLMAP")
    parser.add_argument("--match_prefix", type=str, required=True, help="Contains the directory name and a prefix for the filenames that will be used to write out matches")
    parser.add_argument("--rendering_postfix", type=str, required=False, help="Ending for the images")
    parser.add_argument("--refinement_range", type=float, default=1.0, help="Range for the +REF refinement")
    parser.add_argument("--refinement_step", type=float, default=0.25, help="Step size for the +REF refinement")
    parser.add_argument("--bias_x", type=float, default=0.0, help="Bias term for x-direction for feature detections")
    parser.add_argument("--bias_y", type=float, default=0.0, help="Bias term for y-direction for feature detections")
    args = parser.parse_args()
    print(args)

    assert os.path.exists(args.db_image_dir), "Given db_image_dir does not exist: {}".format(args.db_image_dir)
    assert os.path.exists(args.colmap_model_dir), "Given colmap_model_dir does not exist: {}".format(args.colmap_model_dir)
    assert os.path.exists(args.db_depth_image_dir), "Given db_depth_image_dir does not exist: {}".format(args.db_depth_image_dir)
    assert os.path.exists(args.query_list), "Given query_list file does not exist: {}".format(args.query_list)
    assert os.path.exists(args.query_dir), "Given query_dir does not exist: {}".format(args.query_dir)
    assert os.path.exists(args.retrieval_pairs), "Given retrieval_pairs file does not exist: {}".format(args.retrieval_pairs)

    # get the query information (names, intrinsics calibrations)
    query_infos = open(args.query_list, "r").readlines()
    print(len(query_infos))

    # load the retrieval pairs
    retrieval_pairs = open(args.retrieval_pairs, "r").readlines()
    retrieval_results = {}
    for i in range(0, len(retrieval_pairs)):
        if ',' in retrieval_pairs[i]:
            # AP-GeM retrieval pairs format
            q_name, db_name, score = retrieval_pairs[i].split(', ')
        else:
            # NetVLAD retrieval pairs format
            q_name, db_name = retrieval_pairs[i].split(' ')

        if q_name not in retrieval_results:
            retrieval_results[q_name] = []
        retrieval_results[q_name].append(db_name)

    # maximum side length of the query images
    max_side_length = np.float32(args.max_side_length)

    # load the reference camera poses
    print('Loading the reference poses')
    cameras, images = read_model(args.colmap_model_dir)
    print('Found %d images and %d cameras' % (len(images), len(cameras)))

    # maps from image names to image ids
    map_db_name_to_id = {}
    for img in images:
        map_db_name_to_id[images[img].name] = images[img].id

    reproj_error = args.reproj_error

    db_image_dir = args.db_image_dir
    use_orig_images = args.use_orig_db_images
    rendering_postfix = args.rendering_postfix  #'_rendered_no_color.png'
    db_depth_dir = args.db_depth_image_dir

    # main code: matching and pose estimation
    poses = {}
    best_inliers = {}
    num_top_ranked = args.top_k
    print('Using the %d top-ranked images' % num_top_ranked)

    # initialize matcher
    if "#" not in args.method_name:
        config_file = f'configs/{args.method_name}.yml'
        print(config_file)
        print(args.method_config)
        with open(config_file, 'r') as f:
            imm_args = yaml.load(f, Loader=yaml.FullLoader)[args.method_config]
            if 'ckpt' in imm_args:
                imm_args['ckpt'] = os.path.join('.', imm_args['ckpt'])
            class_name = imm_args['class']
        # Init model
        model = immatch.__dict__[class_name](imm_args)
        matcher = lambda im1, im2: model.match_pairs(im1, im2)
    method = args.method_string

    for i in tqdm(range(0, len(query_infos))):
        print('   ')
        q_name = query_infos[i].split(' ')[0]
        q_data = query_infos[i].split(' ')[1:]
        print(' Trying to localize query image ' + q_name)
        if q_name not in retrieval_results:
            print('   Could not find retrieval results, skipping')
            continue
        width = np.float32(q_data[1])
        height = np.float32(q_data[2])
        if max_side_length > 0:
            scaling_factor = max_side_length / max(width, height)
        else:
            scaling_factor = 1.0
        # We are currently assuming the SIMPLE_RADIAL camera model
        camera_dict = {'model': q_data[0], 'width' : int(width * scaling_factor),
                       'height' : int(height * scaling_factor),
                       'params' : [np.float32(q_data[3]) * scaling_factor,
                                   np.float32(q_data[4]) * scaling_factor,
                                   np.float32(q_data[5]) * scaling_factor,
                                   np.float32(q_data[6])]}

        retrieved_db = retrieval_results[q_name]

        best_inliers[q_name] = 0

        top_ranked_cameras = []

        matches_per_feat = {}


        for j in range(0, num_top_ranked):
            q_name_base = q_name.split('/')[-1]
            db_name_underscore = retrieved_db[j].replace('/', '_')

            img1_name = os.path.join(args.query_dir, q_name)
            if not os.path.exists(img1_name):
                img1_name = os.path.join(args.query_dir, q_name_base)

            assert os.path.exists(img1_name), "Query image does not exist at: {} or at: {}".format(
                    os.path.join(args.query_dir, q_name),
                    os.path.join(args.query_dir, q_name_base))

            if use_orig_images:
                img2_name = os.path.join(db_image_dir, retrieved_db[j])
                if not os.path.exists(img2_name):
                    img2_name = os.path.join(db_image_dir, db_name_underscore)

                assert os.path.exists(img2_name), "Database image does not exist at: {} or at: {}".format(
                        os.path.join(db_image_dir, retrieved_db[j]),
                        os.path.join(db_image_dir, db_name_underscore))

            else:
                img2_name = os.path.join(db_image_dir, retrieved_db[j].split('.')[0] + rendering_postfix)
                if not os.path.exists(img2_name):
                    img2_name = os.path.join(db_image_dir, db_name_underscore.split('.')[0] + rendering_postfix)

                assert os.path.exists(img2_name), "Database image does not exist at: {} or at: {}".format(
                        os.path.join(db_image_dir, retrieved_db[j].split('.')[0] + rendering_postfix),
                        os.path.join(db_image_dir, db_name_underscore.split('.')[0] + rendering_postfix))

            print('   Matching against ' + img2_name)

            # Loads the depth map
            img2_depth = os.path.join(db_depth_dir, retrieved_db[j].split('.')[0] + '_depth.npz')
            if not os.path.exists(img2_depth):
                img2_depth = os.path.join(db_depth_dir, db_name_underscore.split('.')[0] + '_depth.npz')

            assert os.path.exists(img2_depth), "Database depth image does not exist at: {} or at: {}".format(
                    os.path.join(db_depth_dir, retrieved_db[j].split('.')[0] + '_depth.npz'),
                    os.path.join(db_depth_dir, db_name_underscore.split('.')[0] + '_depth.npz'))

            depth_map = np.load(img2_depth)['depth'].astype(np.float32)

            # Get the transformation from reference camera to world coordinates
            img2_id = map_db_name_to_id[retrieved_db[j].strip()]
            T = np.identity(4)
            R = np.asmatrix(qvec2rotmat(images[img2_id].qvec)).transpose()
            T[0:3,0:3] = R
            T[0:3,3] = -R.dot(images[img2_id].tvec)
            P = np.zeros((3,4))
            P[0:3,0:3] = R.transpose()
            P[0:3,3] = images[img2_id].tvec

            colmap_cam = cameras[images[img2_id].camera_id]

            top_ranked_cameras.append({'model': colmap_cam.model,
                                       'width' : colmap_cam.width,
                                       'height' : colmap_cam.height,
                                       'params' : colmap_cam.params,
                                       'q' : images[img2_id].qvec,
                                       't' : images[img2_id].tvec})

            if "#" not in args.method_name:
                # Tries to load the 2D-2D matches from disk to save time.
                match_file_name = args.match_prefix + str(q_name_base.split('.')[0]) \
                                  + '_-_' + str(db_name_underscore.split('.')[0]) + '_-_'\
                                  + args.method_name + '_' + args.method_config\
                                  + '_-_' + str(max_side_length) + '.npy'
                match_file_exists = os.path.exists(match_file_name)

                if match_file_exists:
                    matches = np.load(match_file_name)
                else:
                    # computes the matches
                    matches, _, _, _ = matcher(img1_name, img2_name)
                    np.save(match_file_name, matches)
            else:
                method_names = args.method_name.split("#")
                method_configs = args.method_config.split("#")
                match_dirs = args.match_prefix.split("#")
                matches = np.empty((0, 4))
                for method_idx in range(0, len(method_names)):
                    # Tries to load the 2D-2D matches from disk to save time.
                    match_file_name = match_dirs[method_idx] + str(q_name_base.split('.')[0]) \
                                      + '_-_' + str(db_name_underscore.split('.')[0]) + '_-_'\
                                      + method_names[method_idx] + '_' + method_configs[method_idx]\
                                      + '_-_' + str(max_side_length) + '.npy'
                    match_file_exists = os.path.exists(match_file_name)

                    if match_file_exists:
                        matches_method_idx = np.load(match_file_name)
                        matches = np.append(matches, matches_method_idx, axis=0)



            print('    Number of matches with %s: %d.' % (method, matches.shape[0]))
            if matches.shape[0] < 3:
                continue

            matches[:, 2:] += np.array([args.bias_x, args.bias_y])

            # kpts1 = matches[:, :2]
            kpts2 = matches[:, 2:]

            # Get the corresponding depth values.
            kpts2_int = np.rint(kpts2).astype(np.int64)
            kpts2_int[:,0] = np.clip(kpts2_int[:,0], 0, colmap_cam.width - 1)
            kpts2_int[:,1] = np.clip(kpts2_int[:,1], 0, colmap_cam.height - 1)
            depth_values = depth_map[kpts2_int[:,1], kpts2_int[:,0]]

            # Assumes that the images use the PINHOLE camera model
            fx = colmap_cam.params[0]
            fy = colmap_cam.params[1]
            cx = colmap_cam.params[2]
            cy = colmap_cam.params[3]
            P[0,:] *= fx
            P[1,:] *= fy

            rays = kpts2 - np.array([cx, cy])
            rays = np.append(rays, np.ones(depth_values.reshape((-1,1)).shape), axis=1)
            rays[:,0] /= fx
            rays[:,1] /= fy
            points3D = rays * depth_values.reshape((-1,1))
            num_points = points3D.shape[0]
            points3D_world = np.matmul(T, np.append(points3D, np.ones([num_points, 1]), axis=1).transpose()).transpose()[:, :3]

            for m in range(0, matches.shape[0]):
                m_key = tuple([matches[m, 0], matches[m, 1]])
                xr = np.arange(max(0, np.floor(matches[m, 2])), min(colmap_cam.width - 1, np.floor(matches[m, 2]) + 2)).astype(int)
                yr = np.arange(max(0, np.floor(matches[m, 3])), min(colmap_cam.height - 1, np.floor(matches[m, 3]) + 2)).astype(int)
                xx, yy = np.meshgrid(xr, yr)
                D = depth_map[yy, xx]
                delta_x = matches[m, 2] - np.floor(matches[m, 2])
                delta_y = matches[m, 3] - np.floor(matches[m, 3])
                if len(xr) == 2 and len(yr) == 2:
                    depth_m = (D[0, 0] * (1.0 - delta_x) + D[0, 1] * delta_x) * (1.0 - delta_y) + (D[1, 0] * (1.0 - delta_x) + D[1, 1] * delta_x) * delta_y
                elif len(xr) == 2 and len(yr) == 1:
                    depth_m = D[0, 0] * (1.0 - delta_x) + D[0, 1] * delta_x
                elif len(xr) == 1 and len(yr) == 2:
                    depth_m = D[0, 0] * (1.0 - delta_y) + D[1, 0] * delta_y
                else:
                    depth_m = 0.0

                rays_ = np.array([(matches[m, 2] - cx) / fx, (matches[m, 3] - cy) / fy, 1.0]).transpose()
                points_3D_m = rays_ * depth_m
                points_3D_world_m = np.matmul(T, np.array([points_3D_m[0], points_3D_m[1], points_3D_m[2], 1.0]).transpose()).transpose()[:3]

                m_key = tuple([matches[m, 0], matches[m, 1]])
                pt = points3D_world[m, :]
                pt = points_3D_world_m

                if m_key not in matches_per_feat:
                    matches_per_feat[m_key] = {'keypoint' : matches[m,:2],
                                               'points' : np.empty((0,3)),
                                               'observations' : np.empty((0,2)),
                                               'db_indices' : []}
                matches_per_feat[m_key]['observations'] = np.append(matches_per_feat[m_key]['observations'], (matches[m, 2:] - np.array([cx, cy])).reshape(1,2), axis=0)
                matches_per_feat[m_key]['points'] = np.append(matches_per_feat[m_key]['points'], pt.reshape(1,3), axis=0)
                matches_per_feat[m_key]['db_indices'].append(j)

        matches = []

        for m_key in matches_per_feat.keys():
            matches.append(matches_per_feat[m_key])

        pose_options = {'triangulate' : args.triangulate,
                        'merge_3D_points' : args.merge_3D_points,
                        'cluster_keypoints' : args.cluster_keypoints,
                        'covisibility_filtering' : args.covisibility_filtering,
                        'use_all_matches' : args.all_matches_ransac,
                        'inlier_threshold' : reproj_error,
                        'num_LO_iters' : 10,
                        'min_ransac_iterations' : args.min_ransac_iterations,
                        'max_ransac_iterations' : args.max_ransac_iterations,
                        'ransac_type' : args.ransac_type,
                        'refinement_range' : args.refinement_range,
                        'refinement_step' : args.refinement_step}

        estimate = meshloc.pose_estimation(camera_dict, top_ranked_cameras,
                                           matches, pose_options)
        
        if estimate['success']:
            if best_inliers[q_name] < estimate['num_inliers']:
                poses[q_name] = (estimate['qvec'], estimate['tvec'])
                best_inliers[q_name] = estimate['num_inliers']
            
            print(estimate['qvec'])
            print(estimate['tvec'])

    # Writes out the poses. Code taken from
    # https://github.com/cvg/Hierarchical-Localization/blob/master/hloc/localize_sfm.py#L192
    pose_file = args.out_prefix + str(num_top_ranked) + "_" + method + "_" + str(args.reproj_error)
    if args.triangulate:
        pose_file = pose_file + "_triangulated"
    if args.merge_3D_points:
        pose_file = pose_file + "_merged_3D_points"
    if args.cluster_keypoints:
        pose_file = pose_file + "_keypoint_clusters"
    if args.covisibility_filtering:
        pose_file = pose_file + "_covis_filtering"
    if args.all_matches_ransac:
        pose_file = pose_file + "_all_matches_ransac"
    pose_file = pose_file + "_" + args.ransac_type
    pose_file = pose_file + "_min_" + str(args.min_ransac_iterations) + "_max_" + str(args.max_ransac_iterations)
    pose_file = pose_file + "_ref_" + str(args.refinement_range) + "_" + str(args.refinement_step)
    pose_file = pose_file + "_bias_" + str(args.bias_x) + "_" + str(args.bias_y)
    
    print(pose_file)
    with open(pose_file, 'w') as f:
        for q in poses:
            qvec, tvec = poses[q]
            qvec = ' '.join(map(str, qvec))
            tvec = ' '.join(map(str, tvec))
            name = q.split('/')[-1]
            f.write(f'{name} {qvec} {tvec}\n')


if __name__ == "__main__":
    main()
