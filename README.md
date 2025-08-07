## Source code repository for the paper: 

# Debiasing Implicit Feedback Recommenders via Sliced Wasserstein Distance-based Regularization



## Install

Get Miniconda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
Execute the dependency installation
```bash
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
conda env create -n swd --file environment.yaml 
conda activate swd
```
## Data Preprocessing
Setup `datasources.yaml` by filling each of the maps with your local username and correspoinding url to datasets

Modify the variables `ROOT_DIR` and `OUTPUT_DIR` in `preprocess_dataset.py`

`OUTPUT_DIR` should be the same corresponding to the entry in `local_dataset_path_map`in `datasources.yaml`  

```bash
conda activate swd
python preprocess_dataset.py  
```
## Running

The current implementation is optimized for gpus only and for the following datasets

- [config_files/regmultvae_ml1m.yaml](config_files/regmultvae_ml1m.yaml) 
- [config_files/regmultvae_ekstrabladet.yaml](config_files/regmultvae_ekstrabladet.yaml) 
- [config_files/regmultvae_lfm-demobias.yaml](config_files/regmultvae_lfm-demobias.yaml)

The class that we use a recommender model is in [src/recsys_models/reg_mult_vae.py (RegMultVAE)](src/recsys_models/reg_mult_vae.py)

<!-- We also include the adversarial training based baselines [src/recsys_models/advx_mult_vae.py (AdvXMultVAE)](src/recsys_models/advx_mult_vae.py) for all datasets:
- [config_files/advxmultvae_ekstrabladet.yaml](config_files/advxmultvae_ekstrabladet.yaml)
- [config_files/advxmultvae_lfm-demobias.yaml](config_files/advxmultvae_lfm-demobias.yaml) 
- [config_files/advxmultvae_ml1m.yaml](config_files/advxmultvae_ml1m.yaml) -->


In the following lines we provide templates for excution of experiments:

To train a model:
```bash
conda activate swd
python run.py --config config_files/regmultvae_ml1m.yaml --n_parallel 1 --gpus 0
```
To perform inference attack of a pretrained model:
```bash
conda activate swd
python run_atk.py --experiment /results/path/ml-1m/RegMultVAE--DD-MM-YYYY/ \ 
    --atk_config config_files/regmultvae_ml1m_gender_atk.yaml --n_parallel 1 --gpus 0
```


<!-- ## Using W&B

First generate a __sweep\_id__ :
```bash
wandb sweep config_files/<model>_<dataset>_<sweep>.yaml
```

A __sweep\_id__  of the shape **\<entity\>/\<project\>/\<sweep\>** will be printed in the terminal which should used in the following command 

```bash
python run_sweep_agent.py --sweep_id <entity>/<project>/<sweep> -p 6 --gpus 0,1
```
 -->

