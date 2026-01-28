conda create --prefix=./.conda python=3.11
conda activate ./.conda
conda env update -n .conda --file environment.yaml
pip install opencv-python
------------------------------------------
conda env create -f environment.yaml
conda install -c conda-forge albumentations=1.3.1