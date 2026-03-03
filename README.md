+ Create a new environment in anaconda
```
conda create -n pytorch-gpu python=3.10
```
+ Activate the environment pytorch-gpu
```
conda activate pytorch-gpu
```
+ Deactivate the current environment
```
conda deactivate
```
+ install pytorch, torchvision, torchaudio pytorch-cuda
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
+ For jupyter notebook to open (1)
```
conda install ipykernel
```
+ For jupyter notebook to open (2)
```
python -m ipykernel install --user --name pytorch-gpu --display-name "Python (pytorch-gpu)"
```
+ install cuDNN
```
conda install nvidia::cudnn=9.18.1.3
```
+ install diffusers
```
conda install anaconda::diffusers
```
