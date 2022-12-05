from torchbenchmark.util.framework.openmmlab.install_utils import mim_install_repos

repos = ['mmcv', 'mmengine', 'mmdet']

if __name__ == '__main__':
    mim_install_repos(repos)
