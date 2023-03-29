# Autoregressive Transformers for Data-Driven Spatio-Temporal Learning of Turbulent Flows 
[![PoF 2021](https://img.shields.io/badge/arXiv-2209.08052-red.svg)](https://arxiv.org/abs/2209.08052) 

###

### Abstract:
A convolutional encoder-decoder-based transformer model has been developed to autoregressively train on spatio-temporal data of turbulent flows. It works by predicting future fluid flow fields from the previously predicted fluid flow field to ensure long-term predictions without diverging. The model exhibits significant agreements for \textit{a priori} assessments, and the \textit{a posterior} predictions, after a considerable number of simulation steps, exhibit predicted variances. Autoregressive training and prediction of \textit{a posteriori} states is the primary step towards the development of more complex data-driven turbulence models and simulations.

### Article: https://arxiv.org/abs/2209.08052

### Usage:
#### Download
```bash
git clone https://github.com/aakash30jan/Spatio-Temporal-Learning-of-Turbulent-Flows.git
```
#### May the source be with you
If git is not installed, you can get the source zip with
```bash
wget -O Spatio-Temporal-Learning-of-Turbulent-Flows.zip https://github.com/aakash30jan/https://github.com/aakash30jan/Spatio-Temporal-Learning-of-Turbulent-Flows/archive/refs/heads/main.zip 
unzip Spatio-Temporal-Learning-of-Turbulent-Flows.zip
```

####  Train the model
Make sure you install TF2.0 with GPU support.  
Make sure the training data is stored at `case_dir`  
```bash
cd src
python3 ./train.py 1 both 2 case2 1 2
```

The file train.py is self-explanatory: We first load the system and user-defined libraries, set the training parameters, load the pre-processed dataset, load the model architectures, define training and validation steps to suit TF2.0, and then perform the training. Make sure cuda-capabale devices and drivers are visible to Tensorflow, you may need to `module load cudaxxx` depending on the machine configuration. 

### Issues:
Problems? Please raise an issue at [https://github.com/aakash30jan/Spatio-Temporal-Learning-of-Turbulent-Flows/issues](https://github.com/aakash30jan/Spatio-Temporal-Learning-of-Turbulent-Flows/issues).

[![Issues](https://img.shields.io/github/issues/Spatio-Temporal-Learning-of-Turbulent-Flows/issues)](#Spatio-Temporal-Learning-of-Turbulent-Flows)  [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](#Spatio-Temporal-Learning-of-Turbulent-Flows)

### Citation:
Please use https://arxiv.org/abs/2209.08052 for citing this code or article. You may also download [this](https://scholar.googleusercontent.com/scholar.bib?q=info:PTzhIVpW0RsJ:scholar.google.com/&output=citation&scisdr=CptbyCTkEM7u1rty8Y4:AJ9-iYsAAAAAZCR06Y4lXSAz_rP5bwEKb5A6H6c&scisig=AJ9-iYsAAAAAZCR06ZPjMvPni9SNmZ-KZV5-YX8&scisf=4&ct=citation&cd=-1&hl=en) .bib file or copy the following bibtex entry. 
```
@article{patil2022autoregressive,
  title={Autoregressive transformers for data-driven spatio-temporal learning of turbulent flows},
  author={Patil, Aakash and Viquerat, Jonathan and Hachem, Elie},
  journal={arXiv preprint arXiv:2209.08052},
  year={2022}
}
```

### Disclaimers:
No Warranty:  The subject software is provided "as is" without any warranty of any kind, either expressed, implied, or statutory, including, but not limited to, any warranty that the subject software will conform to specifications, any implied warranties of merchantability, fitness for a particular purpose, or freedom from infringement, any warranty that the subject software will be error free, or any warranty that documentation, if provided, will conform to the subject software. This agreement does not, in any manner, constitute an endorsement by any agency or any prior recipient of any results, resulting designs, hardware, software products or any other applications resulting from use of the subject software. Further, the subject software  disclaims all warranties and liabilities regarding third-party software, if present in the original software, and distributes it "as is.

<!--- ![TARDowns](https://gpvc.arturio.dev/aakash30jan) -->
[![Hits](https://hits.deltapapa.io/github/aakash30jan/Spatio-Temporal-Learning-of-Turbulent-Flows.svg)](#)
