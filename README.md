[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"
# Reinforcement Learning Project

This project was created to make it easier to get started with Reinforcement Learning. It now features: 
- An implementation of the [DDPG Algorithm](https://arxiv.org/abs/1509.02971) in Python, which works for both single-agent environments and multi-agent environments.
- Single and parallel environments in [Unity ML agents](https://unity.com/products/machine-learning-agents) using the [Python API](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Python-API.md).
- Two Jupyter notebooks:
  - [3DBall.ipynb](notebooks/3DBall.ipynb): This is a simple example to get started with Unity ML Agents & the DDPG Algorithm.
  - [3DBall_parallel_environment.ipynb](notebooks/3DBall_parallel_environment.ipynb): The same, but now for an environment run in parallel.

# Getting Started

## Install Basic Dependencies

To set up your python environment to run the code in the notebooks, follow the instructions below. 

- If you're on Windows I recommend installing [Miniforge](https://github.com/conda-forge/miniforge). It's a minimal installer for Conda. I also recommend using the [Mamba](https://github.com/mamba-org/mamba) package manager instead of [Conda](https://docs.conda.io/). It works almost the same as Conda, but only faster. There's a [cheatsheet](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html) of Conda commands which also work in Mamba. To install Mamba, use this command:
```bash
conda install mamba -n base -c conda-forge 
```
- Create (and activate) a new environment with Python 3.6 or later. I recommend using Python 3.9:

    - __Linux__ or __Mac__:
    ```bash
    mamba create --name rl39 python=3.9 numpy
    source activate rl39
    ```
    - __Windows__:
    ```bash
    mamba create --name rl39 python=3.9 numpy
    activate rl39
    ```
- Install PyTorch by following instructions on [Pytorch.org](https://pytorch.org/). For example, to install PyTorch on
   Windows with GPU support, use this command:

```bash
mamba install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

- Install additional packages:
```bash
mamba install jupyter notebook matplotlib
```

- Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `rl39` environment in Jupyter.

```bash
python -m ipykernel install --user --name rl39 --display-name "rl39"
```

- Change the kernel to match the `rl39` environment by using the drop-down menu `Kernel` -> `Change kernel` inside Jupyter Notebook.

## Install Unity Machine Learning Agents

**Note**: 
In order to run the notebooks on **Windows**, it's not necessary to install the Unity Editor, because I have provided the [standalone executables](notebooks/README.md) of the environments for you.

[Unity ML Agents](https://unity.com/products/machine-learning-agents) is the software that we use for the environments. The agents that we create in Python can interact with these environments. Unity ML Agents consists of several parts:
- [The Unity Editor](https://unity.com/) is used for creating environments. To install:
  - Install [Unity Hub](https://unity.com/download).
  - Install the latest version of Unity by clicking on the green button `Unity Hub` on the [download page](https://unity3d.com/get-unity/download/archive). 
  
  To start the Unity editor you must first have a project:
     
   - Start the Unity Hub.
   - Click on "Projects"
   - Create a new dummy project.
   - Click on the project you've just added in the Unity Hub. The Unity Editor should start now.

- [The Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents#unity-ml-agents-toolkit). Download [the latest release](https://github.com/Unity-Technologies/ml-agents/releases) of the source code or use the [Git](https://git-scm.com/downloads/guis) command: `git clone --branch release_18 https://github.com/Unity-Technologies/ml-agents.git`.
- The Unity ML Agents package is used inside the Unity Editor. Please read [the instructions for installation](https://github.com/Unity-Technologies/ml-agents/blob/release_18_docs/docs/Installation.md#install-the-comunityml-agents-unity-package).
- The `mlagents` Python package is used as a bridge between Python and the Unity editor (or standalone executable). To install, use this command: `python -m pip install mlagents==0.27.0`.
Please note that there's no conda package available for this.

## Install an IDE for Python Development

For Windows, I would recommend using [PyCharm](https://www.jetbrains.com/pycharm/) (my choice), or [Visual Studio Code](https://code.visualstudio.com/).
Inside those IDEs you can use the Conda environment you have just created.

## Creating a custom Unity executable

### Load the examples project
[The Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents#unity-ml-agents-toolkit) contains several [example environments](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md). Here we will load them all inside the Unity editor:
- Start the Unity Hub.
- Click on "Projects"
- Add a project by navigating to the `Project` folder inside the toolkit.
- Click on the project you've just added in the Unity Hub. The Unity Editor should start now.

### Create a 3D Ball executable
The 3D Ball example contains 12 environments in one, but this doesn't work very well in the Python API. The main problem is that there's no way to reset each environment individually. Therefore, we will remove the other 11 environments in the editor:
- Load the 3D Ball scene, by going to the project window and navigating to `Examples` -> `3DBall` -> `Scenes`-> `3DBall`
- In the Hierarchy window select the other 11 3DBall objects and delete them, so that only the `3DBall` object remains.

Next, we will build the executable:
- Go to `File` -> `Build Settings`
- In the Build Settings window, click `Build`
- Navigate to `notebooks` folder and add `3DBall` to the folder name that is used for the build.


## Instructions for running the notebooks

1. [Download](notebooks/README.md) the Unity executables for Windows. In case you're not on Windows, you have to build the executables yourself by following the instructions above. 
2. Place the Unity executable folders in the same folder as the notebooks.
3. Load a notebook with Jupyter notebook. (The command to start Jupyter notebook is `jupyter notebook`)
4. Follow further instructions in the notebook.
