#!/usr/bin/env bash

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



# Script for downloading the data from MeshLoc data repository
# - usage:
#   - if no parameter are passed, the whole data repository will be downloaded
#     to current directory
#   -n <all|aachen|12_scenes> - specifies which dataset to download
#   -p <string> - specifies directory where the data will be downloaded
#   -z - unzips everything and removes the zip files
#
# - individual files can be manually downloaded from: 
#   https://data.ciirc.cvut.cz/public/projects/2022MeshLoc

base_url="https://data.ciirc.cvut.cz/public/projects/2022MeshLoc"

dataset_name="all"
target_path=$(pwd)
unzip=false

help() { echo "Usage: $0 [-n <all|aachen|12_scenes>] [-p <string>] [-z]" 1>&2; exit 1; }

while getopts ":n:p:z" o; do
    case "${o}" in
        n)
            n=${OPTARG}
            [[ "${n}" == "all" || "${n}" == "aachen" || "${n}" == "12_scenes" ]] && dataset_name=${n} || help
            ;;
        p)
            p=${OPTARG}
            [ -d ${p} ] && target_path=${p} || { echo "The given path is not a directory: ${p}" >&2; exit 1; }
            ;;
        z)
            unzip=true
            ;;
        :)
            echo "Missing option argument for -$OPTARG" >&2; exit 1
            ;;
        *)
            help
            ;;
    esac
done
shift $((OPTIND-1))

case "${dataset_name}" in 
    all)
        download_url=${base_url}
        ;;
    aachen)
        download_url="${base_url}/aachen_day_night_v11"
        ;;
    12_scenes)
        download_url="${base_url}/12_scenes"
        ;;
esac

echo "Downloading the data..."
wget -r -nH --no-parent --cut-dirs=3 --reject="index*" --directory-prefix=${target_path} ${download_url}

if [[ "${unzip}" == true ]]
then
    echo "Unzipping the data..."
    case "${dataset_name}" in 
    all)
        unzip "${target_path}/12_scenes/db_renderings/*.zip" -d "${target_path}/12_scenes/db_renderings"
        unzip "${target_path}/aachen_day_night_v11/db_renderings/*.zip" -d "${target_path}/aachen_day_night_v11/db_renderings"
        unzip "${target_path}/aachen_day_night_v11/meshes/*.zip" -d "${target_path}/aachen_day_night_v11/meshes"

        rm ${target_path}/12_scenes/db_renderings/*.zip
        rm ${target_path}/aachen_day_night_v11/db_renderings/*.zip
        rm ${target_path}/aachen_day_night_v11/meshes/*.zip
        ;;
    aachen)
        unzip "${target_path}/aachen_day_night_v11/db_renderings/*.zip" -d "${target_path}/aachen_day_night_v11/db_renderings"
        unzip "${target_path}/aachen_day_night_v11/meshes/*.zip" -d "${target_path}/aachen_day_night_v11/meshes"

        rm ${target_path}/aachen_day_night_v11/db_renderings/*.zip
        rm ${target_path}/aachen_day_night_v11/meshes/*.zip
        ;;
    12_scenes)
        unzip "${target_path}/12_scenes/db_renderings/*.zip" -d "${target_path}/12_scenes/db_renderings"

        rm ${target_path}/12_scenes/db_renderings/*.zip
        ;;
esac
fi