# A Hybrid Active Learning Regression Approach for Accelerating Annotation with Data Generation Constraints
The source code of the MQR-UD active learning method.
An AL method in a constrained framework.

## Dependencies
Please find the dependencies in the `requirements.txt` file.

## Code Structure
The code is structured as follows:
- `al_experiments/al_framework/al_methods`: The active learning methods. mqr_ud.py is the main implementation of our method.
- `al_experiments/al_framework/batch_split`: The defined groups for batch splitting.
- `data_processing`: The data processing code.
- `datasets`: The dataset files.
- `results`: The folder to save the experimental results.
- `al_experiments/DatasetName`: The experimental scripts for each dataset.

## License
The code implementation is licensed under the Apache 2.0 license.

## Comparison Methods Code
The baseline black-box LCMD source code can be retrieved from https://github.com/dholzmueller/bmdal_reg, and https://github.com/BlackHC/2302.08981.

The baseline GSx code is adapted from https://github.com/sronast/al_3dgraph/blob/main/al/selection_methods.py.

## Citations: the comparision methods and some utility functions in this repository are from:

```bibtex

@inproceedings{raychaudhuri1995minimisation,
  title={Minimisation of data collection by active learning},
  author={RayChaudhuri, Tirthankar and Hamey, Leonard GC},
  booktitle={Proceedings of ICNN'95-International Conference on Neural Networks},
  volume={3},
  pages={1338--1341},
  year={1995},
  organization={IEEE}
}

@article{wu2019active,
  title={Active learning for regression using greedy sampling},
  author={Wu, Dongrui and Lin, Chin-Teng and Huang, Jian},
  journal={Information Sciences},
  volume={474},
  pages={90--105},
  year={2019},
  publisher={Elsevier}
}

@article{holzmuller2023framework,
  title={A framework and benchmark for deep batch active learning for regression},
  author={Holzm{\"u}ller, David and Zaverkin, Viktor and K{\"a}stner, Johannes and Steinwart, Ingo},
  journal={Journal of Machine Learning Research},
  volume={24},
  number={164},
  pages={1--81},
  year={2023}
}

@article{kirschblack,
  title={Black-Box Batch Active Learning for Regression},
  author={Kirsch, Andreas},
  journal={Transactions on Machine Learning Research},
  year={2023}
}

@article{subedi2024empowering,
  title={Empowering active learning for 3D molecular graphs with geometric graph isomorphism},
  author={Subedi, Ronast and Wei, Lu and Gao, Wenhan and Chakraborty, Shayok and Liu, Yi},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={55507--55537},
  year={2024}
}

@article{sener2017active,
  title={Active learning for convolutional neural networks: A core-set approach},
  author={Sener, Ozan and Savarese, Silvio},
  journal={In International Conference on Learning Representations},
  year={2018}
}

@article{modAL2018,
  title={mod{AL}: {A} modular active learning framework for {P}ython},
  author={Tivadar Danka and Peter Horvath},
  url={https://github.com/modAL-python/modAL},
  note={available on arXiv at \url{https://arxiv.org/abs/1805.00979}}
}
```


