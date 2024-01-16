"""
智算平台
"""
import json
import os
import subprocess
import sys
from typing import List, Union

import moxing as mox

from tools.modelarts_adapter.modelarts import run_with_single_rank, LOCAL_RANK, _logger


@run_with_single_rank(local_rank=LOCAL_RANK, signal="/tmp/DOWNLOAD_CKPT_SUCCESS")
def pretrain_to_env(pretrain_url, pretrain_dir) -> Union[None, str]:
    """copy pretrain to training image"""
    pretrain_url_json = json.loads(pretrain_url)
    if not os.path.exists(pretrain_dir):
        os.makedirs(pretrain_dir)

    if len(pretrain_url_json) == 0:
        return None

    pretrain_url_json = pretrain_url_json[0]
    url, name = pretrain_url_json["model_url"], pretrain_url_json["model_name"]
    modelfile_path = os.path.join(pretrain_dir, name)
    try:
        mox.file.copy(url, modelfile_path)
        _logger.info(f'Successfully download {url} to {modelfile_path}')
    except Exception as e:
        _logger.info(f'moxing download {url} to {modelfile_path} failed: {e}')
    return modelfile_path


@run_with_single_rank(local_rank=LOCAL_RANK, signal="/tmp/UPLOAD_SUCCESS")
def multidataset_to_env(multi_data_url, data_dir):
    """copy single or multi dataset to training image"""
    multi_data_json = json.loads(multi_data_url)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for data_json in multi_data_json:
        url, name = data_json["dataset_url"], data_json["dataset_name"]
        zipfile_path = os.path.join(data_dir, name)
        try:
            mox.file.copy(url, zipfile_path)
            _logger.info(f'Successfully download {url} to {zipfile_path}')

            # get filename and unzip the dataset
            if zipfile_path.endswith(".zip"):
                os.system(f"unzip -q {zipfile_path} -d {os.path.dirname(zipfile_path)}")
            elif zipfile_path.endswith(".tgz") or zipfile_path.endswith(".tar.gz"):
                os.system(f"tar -zxf {zipfile_path} -C {os.path.dirname(zipfile_path)}")
            else:
                raise ValueError(f"invalid zipfile: {zipfile_path}")
        except Exception as e:
            _logger.info(f'moxing download {url} to {zipfile_path} failed: {e}')


@run_with_single_rank(local_rank=LOCAL_RANK, signal="/tmp/INSTALL_SUCCESS")
def install_packages(req_path: str = "requirements.txt") -> None:
    # NOTE 智算平台用不了清华源
    # requirement_txt = os.path.join(project_dir, "requirements.txt")
    _logger.info(f"Packages to be installed: {req_path}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])

    # 手动添加libgeos库，有个镜像已经包含了
    # import shutil
    # third_part = os.path.join('/'.join(__file__.split('/')[:-3]), 'third_part/libgeos_aarch64/')
    # dst = os.path.join(sys.prefix, 'lib')
    # for file in os.listdir(third_part):
    #     if 'libgeos' in file and not os.path.exists(os.path.join(dst, file)):
    #         src = os.path.join(third_part, file)
    #         _logger.info(f'copy {src} to {dst}')
    #         shutil.copy(src, dst, follow_symlinks=False)


@run_with_single_rank(local_rank=LOCAL_RANK, signal="/tmp/INSTALL_SUCCESS")
def conda_install_packages(pkgs: List[str]):
    # FIXME 智算平台训练任务用不了conda，报错 sh: conda: command not found
    if len(pkgs) == 0:
        _logger.info(f"None packages need to be installed.")
        return
    _logger.info(f"Packages to be installed: {pkgs}.")

    # 找conda
    python = sys.executable
    _logger.info(f"python path: {python}")
    python_split = python.split('/')
    for i, dir in enumerate(python_split):
        if 'conda' in dir:
            break
    else:
        _logger.info(f"Cannot find conda.")
        return
    conda = os.path.join('/'.join(python_split[:i+1]), 'bin/conda')
    assert os.path.exists(conda), 'invalid path'
    _logger.info(f"conda path: {conda}")

    # 判断是否在base外的虚拟环境
    if 'envs' in python:
        subprocess.check_call([conda, "install", "-n", python_split[i+2], "-y"] + pkgs)
        _logger.info(f"Packages have been installed to env {python_split[i+2]} successfully.")
    else:
        subprocess.check_call([conda, "install", "-y"] + pkgs)
        _logger.info(f"Packages have been installed to base env successfully.")


def modelarts_setup(args):
    if args.enable_modelarts:
        cur_dir = os.path.dirname(os.path.abspath(__file__))

        # change relative path of configure file to absolute
        if not os.path.isabs(args.config):
            args.config = os.path.abspath(os.path.join(cur_dir, "../../", args.config))

        # req_path = os.path.abspath(os.path.join(cur_dir, "../../requirements/modelarts.txt"))
        req_path = os.path.abspath(os.path.join(cur_dir, "../../requirements/c2net.txt"))
        install_packages(req_path)
        # conda_install_packages(["geos"])  # FIXME sh: conda command not found
        return True
    return False
