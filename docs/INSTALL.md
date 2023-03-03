# Installation

In our experiments, we used the following setup:
- Ubuntu 20.04.2 LTS
- Python 3.6.11
- CUDA 10.2
- Pytorch 1.5.1


In order to reimplement our development environment, pelase, follow the below-mentioned instructions.


<hr>


## I. Setup Code Environment

NOTE: In case you have multiple CUDA versions installed, please, make sure to initialise the appropriate system CUDA version before running any command.
```bash
# <10.2> - CUDA version number
module load cuda-10.2
```

1) Setup a conda environment:

    - With Conda [Recommended]:
    ```bash
    # Create a conda environment with dependencies from the environment.yml file
    conda env create --name med_lowdata -f environment.yml
    # Activate the environment
    conda activate med_lowdata
    ```
    
    - With PIP [Not Tested]:
    ```bash
    # Create a conda environment
    conda create -n med_lowdata python=3.6.11
    # Activate the environment
    conda activate med_lowdata
    # Install dependencies from the requirements.txt file
    pip install -r requirements.txt

    ```

2) Install the Apex library for mixed-precision training:

    - With conda [Our choice]:
    ```bash
    ## Both commands are necessary
    # May throw warnings, but it is okay
    conda install -c conda-forge nvidia-apex
    # Answer 'Yes' when numpy package upgrade is inquired
    conda install -c conda-forge nvidia-apex=0.1 
    ```

    - From source [Recommended]:
    ```bash
    git clone https://github.com/NVIDIA/apex
    cd apex
    # May throw unexpected system-specific errors
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    ```

3) [Additional] In case of runtime errors related to numpy or scikit-learn packages, force downgrade numpy to the '1.15.4' version:

    ```bash
    pip install numpy==1.15.4
    ```


<hr>


## II. Download Pre-trained Models
