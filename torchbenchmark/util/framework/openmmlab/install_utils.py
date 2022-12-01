import subprocess
import sys
from typing import List

mapping = {
    'mmengine': 'mmengine',
    'mmcv': 'mmcv>=1.0.0rc0',
    'mmcls': 'mmcls>=1.0.0rc0',
    'mmdet': 'mmdet>=3.0.0rc0'
}

def pip_install_openmim():
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', '-q', 'openmim'
    ])

def mim_install_repos(repos: List[str]):
    pip_install_openmim()
    for repo in repos:
        subprocess.check_call([
            sys.executable, '-m', 'mim', 'install', '-q',
            mapping[repo]
        ])