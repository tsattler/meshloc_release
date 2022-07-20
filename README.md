# MeshLoc
[//]: # (TODO: Change the paper name to link to arXiv)
Code for the paper **MeshLoc: Mesh-Based Visual Localization**

[//]: # (TODO: Add bibtex reference)

## License
This repository is licensed under the 3-Clause BSD License. See the [LICENSE](https://github.com/tsattler/meshloc_release/blob/main/LICENSE) file for full text.

## Installation
- follow [Image Matching Toolbox](https://github.com/GrumpyZhou/image-matching-toolbox/blob/main/docs/install.md) installation guide
	- clone the repository (`git clone https://github.com/GrumpyZhou/image-matching-toolbox.git`)
	- prepare `immatch` conda environment (`conda env create -f environment.yml`)
	- activate the `immatch` environment (`conda activate immatch`)
	- install immatch toolbox as Python package (`python setup.py develop`)
		- possibly update of [PyTorch](https://pytorch.org/get-started/locally/) package will be necessary (e.g. `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`)
- install other necessary packages to `immatch` environment (`conda install pyyaml`)
- clone the MeshLoc repository
    - `git clone --recurse-submodules https://github.com/tsattler/meshloc_release.git`
- install mesh_loc Python package to `immatch` environment 
	- `cd <mesh_loc_dir>`
	- `python setup.py develop`

## Usage
Current implementation contains localization scripts for two datasets - [Aachen v1.1](https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/) and [12 Scenes](http://graphics.stanford.edu/projects/reloc/). The scripts should be launched from `image-matching-toolbox` directory so the scripts have direct access to the feature matching configuration files. Also activate the prepared `immatch` conda environment with all necessary packages.

Example on Aachen v1.1 dataset:
```
cd <image_matching_toolbox_dir>

conda activate immatch

python3 <mesh_loc_dir>/localize.py \
--db_image_dir <aachen_dataset_path>/images_upright \
--db_depth_image_dir <meshloc_dataset_path>/aachen_day_night_v11/db_renderings/AC14_depth_orig_undist \
--colmap_model_dir <meshloc_dataset_path>/aachen_day_night_v11/db_colmap_models/orig_undist \
--query_dir <aachen_dataset_path>/images_upright \
--query_list <meshloc_dataset_path>/aachen_day_night_v11/night_time_queries_with_intrinsics.txt \
--out_prefix <experiment_outputs_dir_path> \
--match_prefix <experiment_matches_dir_path> \
--method_name patch2pix \
--method_config aachen_v1_1 \
--method_string patch2pix_aachen_v1_1_ \
--retrieval_pairs <meshloc_dataset_path>/aachen_day_night_v11/retrieval_pairs/NetVLAD_top50.txt \
--top_k 50 \
--max_side_length -1 \
--ransac_type POSELIB+REF \
--min_ransac_iterations 10000 \
--max_ransac_iterations 100000 \
--reproj_error 20.0 \
--use_orig_db_images True \
--cluster_keypoints True
```

Example on 12 Scenes dataset:
```
cd <image_matching_toolbox_dir>

conda activate immatch

python3 <mesh_loc_dir>/localize_12scenes.py \
--db_image_dir <12_scenes_dataset_path>/office1/manolis/data \
--db_depth_image_dir <meshloc_dataset_path>/12_scenes/db_renderings/12_scenes_apt1_kitchen_depth \
--colmap_model_dir <meshloc_dataset_path>/12_scenes/db_colmap_models/apt1_kitchen \
--query_dir <12_scenes_dataset_path>/office1/manolis/data \
--query_list <meshloc_dataset_path>/12_scenes/queries_with_intrinsics/apt1_kitchen_queries_with_intrinsics.txt \
--out_prefix <experiment_outputs_dir_path> \
--match_prefix <experiment_matches_dir_path> \
--method_name loftr \
--method_config default \
--method_string loftr_default_ \
--retrieval_pairs <meshloc_dataset_path>/12_scenes/retrieval_pairs/apt1_kitchen_DVLAD_top20.txt \
--top_k 20 \
--max_side_length -1 \
--ransac_type POSELIB+REF \
--min_ransac_iterations 10000 \
--max_ransac_iterations 100000 \
--reproj_error 20.0 \
--use_orig_db_images True
```

List of localization script arguments:
- **db_image_dir** = directory with database images
- **db_depth_image_dir** = directory with depth database images (assume `_depth.npz`  postfix)
- **rendering_postfix** = postfix of rendered database images if used (our IBMR rendering pipeline produces either `_rendered_color.png` or `_rendered_no_color.png` depending on shader preset)
- **colmap_model_dir** = directory of a COLMAP model defining the database cameras - `images` file contains subpaths from `db_image_dir` to the images
- **query_dir** = directory with query images
- **query_list** = list of query images with intrinsics - contains subpaths from `query_dir` to the images
- **out_prefix** = prefix (including path) of output file (with estimated query poses) in format accepted by [The Visual Localization Benchmark](https://www.visuallocalization.net/)
- **match_prefix** = prefix (including path) for files containing local matches between query and database images
- **method_name** = matching method name - see file names in [image-matching-toolbox/configs](https://github.com/GrumpyZhou/image-matching-toolbox/tree/main/configs) - e.g. `patch2pix`
- **method_config** = one of the method configurations - see configurations inside individual YAML files (e.g. [image-matching-toolbox/configs/patch2pix.yml](https://github.com/GrumpyZhou/image-matching-toolbox/blob/main/configs/patch2pix.yml)) - e.g. `aachen_v1_1`
- **method_string** - string used for your identification of matching method in output files - e.g. `patch2pix_aachen_v1_1_`
- **retrieval_pairs** = file from a retrieval step containing similar database images for each query image (text file, each line contains `query_image_file database_image_file`)
- **top_k** = number of top retrieved database images, which will be used for localization
- **max_side_length** = longer side of the database images - for scaling of the query intrinsics (-1 for full resolution)
- **ransac_type** = RANSAC type (possible values: `POSELIB`, `POSELIB+REF`)
- **min_ransac_iterations** = minimum number of RANSAC iterations
- **max_ransac_iterations** = maximum number of RANSAC iterations
- **reproj_error** = reprojection error RANSAC threshold
- **use_orig_db_images** = set True if using original images (not renderings) for `db_image_dir` 
- **triangulate** = set True to use triangulation instead of 3D points from depth maps
- **merge_3D_points** = set True to select one of multiple 3D points available per query feature
- **cluster_keypoints** = set True to cluster keypoints (applicable only for patch2pix)
- **covisibility_filtering** = set True to use covisibility filtering
- **all_matches_ransac** = use all possible 2D-3D matches in RANSAC
- **refinement_range** = range for the +REF refinement
- **refinement_step** = step size for the +REF refinement
- **bias_x** = bias term for x-direction for feature detections
- **bias_y** = bias term for y-direction for feature detections

The pose estimates from the localization pipeline on Aachen v1.1 can be evaluated by benchmark at [https://www.visuallocalization.net/](https://www.visuallocalization.net/). The pose estimates on 12 Scenes can be evaluated by the [https://github.com/tsattler/visloc_pseudo_gt_limitations](https://github.com/tsattler/visloc_pseudo_gt_limitations) repository, which also contains links to 12 Scenes COLMAP models.

## Data
The data repository can be found at [https://data.ciirc.cvut.cz/public/projects/2022MeshLoc](https://data.ciirc.cvut.cz/public/projects/2022MeshLoc).

You can use [the download script](https://github.com/tsattler/meshloc_release/blob/main/download_meshloc_data.sh) to get all the data easily.

Script parameters:
- if no parameter are passed, the whole data repository will be downloaded to current directory
- **-n <all|aachen|12_scenes>** - specifies which dataset to download
- **-p <string>** - specifies the directory where the data will be downloaded
- **-z** - unzips everything and removes the zip files



## Acknowledgements
This repository is heavily using [PoseLib](https://github.com/vlarsson/PoseLib), [RansacLib](https://github.com/tsattler/RansacLib) and [Image Matching Toolbox](https://github.com/GrumpyZhou/image-matching-toolbox/). We would to thank all the contributors of these repositories.