## [Empowerment driven Exploration using Mutual Information Estimation](https://arxiv.org/pdf/1810.05533.pdf) ##


Code inspired by Exploration using Random Distillation

### Installation and Usage
The following command should train an RND agent on Montezuma's Revenge
```bash
python run_atari.py --gamma_ext 0.999
```
To use more than one gpu/machine, use MPI (e.g. `mpiexec -n 8 python run_atari.py --num_env 128 --gamma_ext 0.999` should use 1024 parallel environments to collect experience on an 8 gpu machine). 

### [Blog post](https://navneet-nmk.github.io/2018-08-26-empowerment/)
