# DropConnect2023_Li
Improve model robustness against hardware faults on RRAM accelerators with the linear bottleneck. This project is a derivative work of [Xiang et.al 2024](https://arxiv.org/abs/2404.15498). For more introduction of the project, see its [project page](https://jhanliufu.github.io/projects/drop_connect.html) on Jhan's website.

## Files
- **[trainer.py](trainer.py)** is the main training script. The model to be trained and the training hyperparameters are specified by JSON configuration files stored in [config](config). To run the script, run the following command: ```python trainer.py --json config/[your_config_file].json```. Check ```python trainer.py --help``` for legal arguments.
- **[flops.py](flops.py)** is a script that counts the number of floating-point operations (FLOPs) a particular neural network pass involves when forward passing a sample from a particular dataset.  To run the script, run the following command: ```python flops.py --dataset <dataset> --arch <model>```
- **[models](models)** contains implementations of various deep learning models. 
