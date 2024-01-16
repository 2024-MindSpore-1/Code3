# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Moxing adapter for ModelArts"""

import os

_global_sync_count = 0


def get_device_id():
    device_id = os.getenv('DEVICE_ID', '0')
    return int(device_id)


def get_device_num():
    device_num = os.getenv('RANK_SIZE', '1')
    return int(device_num)


def get_rank_id():
    global_rank_id = os.getenv('RANK_ID', '0')
    return int(global_rank_id)


def get_job_id():
    job_id = os.getenv('JOB_ID')
    job_id = job_id if job_id != "" else "default"
    return job_id


def sync_data(from_path, to_path, threads=16, unzip=False):
    """
    Download data from remote obs to local directory if the first url is remote url and the second one is local path
    Upload data from local directory to remote obs in contrast.
    """
    import moxing as mox
    import time
    global _global_sync_count
    sync_lock = "/tmp/copy_sync.lock" + str(_global_sync_count)
    _global_sync_count += 1

    # Each server contains 8 devices as most.
    if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
        print("from path: ", from_path)
        print("to path: ", to_path)
        mox.file.copy_parallel(from_path, to_path, threads=threads)
        print("===finish data synchronization===")
        try:
            os.mknod(sync_lock)
        except IOError:
            pass
        print("===save flag===")

    while True:
        if os.path.exists(sync_lock):
            break
        time.sleep(1)

    print("Finish sync data from {} to {}.".format(from_path, to_path))

    sync_lock = "/tmp/copy_sync.lock" + str(_global_sync_count)
    _global_sync_count += 1

    import zipfile
    import glob

    zip_file = zipfile.ZipFile(to_path, "r")
    zip_file_list = zip_file.filelist
    zip_file_list_len = len(zip_file_list)
    results = [zip_file_list_len // min(get_device_num(), 8), ] * min(get_device_num(), 8)
    addition = zip_file_list_len % min(get_device_num(), 8)
    for index in range(addition):
        results[index] += 1

    print(f"total: {zip_file_list_len}")

    if get_device_id() == 0:
        start, end = 0, results[0]
    else:
        start, end = sum(results[0:get_device_id()]), sum(results[0:get_device_id() + 1])

    print(f"device_id: {get_device_id()}, start: {start}, end: {end}")

    for index in range(start, end):
        zip_file.extract(zip_file_list[index], '/cache')

    imagenet_num = len(glob.glob("/cache/imagenet/*/*/*"))
    time.sleep(1)
    while imagenet_num != len(glob.glob("/cache/imagenet/*/*/*")):
        imagenet_num = len(glob.glob("/cache/imagenet/*/*/*"))
        time.sleep(1)

    if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
        try:
            os.mknod(sync_lock)
        except IOError:
            pass

    while True:
        if os.path.exists(sync_lock):
            break
        time.sleep(1)

    print("Device_id: {}. Finish sync data from {} to {}.".format(get_device_id(), from_path, to_path))
