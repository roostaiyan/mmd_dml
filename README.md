# mmd_dml
Source Code for "Multi-Modal Deep Distance Metric Learning"
# 
- If you have used this source code, please also cite the following paper:

@article{roostaiyan2017multi,
  title={Multi-modal deep distance metric learning},
  author={Roostaiyan, Seyed Mahdi and Imani, Ehsan and Baghshah, Mahdieh Soleymani},
  journal={Intelligent Data Analysis},
  volume={21},
  number={6},
  pages={1351--1369},
  year={2017},
  publisher={IOS Press}
}


- This code is based on Wang Wei's [1] code for "Effective Multimodal Retrieval Based on Stacked Autoencoders."

- Installing the dependencies is like the original code: (The following lines are from Wang Wei's ReadMe file)

	1. Basic Python libraries such as numpy and scipy, are required. 
	It has been tested on CentOS 6.4 using CUDA 5.0 with NVIDIA GPU (GeForce GTX TITAN).
	2. install CUDA tookit (https://developer.nvidia.com/cuda-toolkit-50-archive)
	3. install CUDAMat (https://code.google.com/p/cudamat/)
	add libcudamat.so to your LD_LIBARAY_PATH
	4. install gnumpy (http://www.cs.toronto.edu/~tijmen/gnumpy.html)
	it is already included. Do not use the one from the website directly, 
	as it does not support function pow(matrix, matrix).
	5. install configparser 3.3.0r2 (https://pypi.python.org/pypi/configparser)
	this package is an upgrade of configparser built-in with python 2.7

- A tiny dataset (extracted from cal10) is provided in data/ .

- A sample config file for the dataset is in config/tiny/ . You can modify it or create new config files using create_config.py

- run.py contains a script for training and testing the network.

- After training, the code saves the targets of the network as numpy arrays (The last line of output when running run.py shows this). 
  You can use them for computing MAP or other measures of information retrieval. (Look at these functions in evaluate.py: evalSingleModal2, saveTarget)


[1] 	Wang, Wei, Beng Chin Ooi, Xiaoyan Yang, Dongxiang Zhang, and Yueting Zhuang. 
	"Effective multi-modal retrieval based on stacked auto-encoders." Proceedings of the VLDB Endowment 7, no. 8 (2014): 649-660.
