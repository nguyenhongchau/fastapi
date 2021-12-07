import requests
import git
import subprocess
import time
import os
from pathlib import Path

repo = git.Repo.init(".")
repo.config_writer().set_value("user", "name", "athoca").release()
repo.config_writer().set_value("user", "email", "athoca.nguyen@gmail.com").release()

def download_dataset_file(url, folder):
    # download large file and write into folder
    local_filename = url.split('/')[-1]
    local_filename = f'nets/datasets/{folder}/{local_filename}'
    r = requests.get(url)
    if r.ok:  
        with open(local_filename, 'wb') as f:
            f.write(r.content)
        return local_filename

def version_datasets(dataset_id, path="nets/datasets",timeout=100):
    updated_files = [path + ".dvc", os.path.join(str(Path(path).parents[0]), ".gitignore")]
    process = subprocess.Popen(['dvc', 'add', path], 
                           stdout=subprocess.PIPE,
                           universal_newlines=True)
    start = time.time()
    while time.time() - start < timeout:
        # output = process.stdout.readline()
        # print(output.strip())
        return_code = process.poll()
        if return_code is not None:
            # print('RETURN CODE', return_code)
            # Process has finished, read rest of the output 
            # for output in process.stdout.readlines():
            #     print(output.strip())
            break
        time.sleep(0.5)
    # git add and commit new dvc file
    repo.git.add(*updated_files)
    if "nothing added to commit" not in repo.git.status():
        repo.git.commit('-m', f'add new dataset{dataset_id}')

def version_weights(path="nets/weights",timeout=100):
    updated_files = [path + ".dvc", os.path.join(str(Path(path).parents[0]), ".gitignore")]
    process = subprocess.Popen(['dvc', 'add', path], 
                           stdout=subprocess.PIPE,
                           universal_newlines=True)
    start = time.time()
    while time.time() - start < timeout:
        # output = process.stdout.readline()
        # print(output.strip())
        return_code = process.poll()
        if return_code is not None:
            # print('RETURN CODE', return_code)
            # Process has finished, read rest of the output 
            # for output in process.stdout.readlines():
            #     print(output.strip())
            break
        time.sleep(0.5)
    # git add and commit new dvc file
    repo.git.add(*updated_files)
    if "nothing added to commit" not in repo.git.status():
        repo.git.commit('-m', 'add new trained weights')



