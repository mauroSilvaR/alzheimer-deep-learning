# Classifying Alzheimer Disease with Convolutional Neural Networks from MRI
> The goal of this project is classifying alzheimer disease from magnetic ressonance images compressed in .nii files using convolutional neural networks architectures.

## Developers

| Full Name                          | E-mail                     |
| -----                              | ----------------           |
| `Matheus de Almeida Silva`         | ms.asilvas1@gmail.com      |
| `Gabriel Darin Verga`              | gabrieldarin@hotmail.com   |
| `Mauricio Alves Bedun Junior`      | mauriciobedun@hotmail.com  |
| `Mauro da Silva Ribeiro`           | msr_hck1@hotmail.com       |
| `Kaue Santos Bueno`                | kauesb@hotmail.com         |
| `Matheus Augusto Somera Fernandes` | matheus.somera@gmail.com   |

## Usage
See how simple that is to use this service.

**1.** Clone this repository:
```bash
$ git clone https://github.com/matheus-asilva/alzheimer-deep-learning.git
$ cd alzheimer-deep-learning
```

**2.** Create environment:
```bash
$ conda create --name alzheimer-deep-learning --file requirements.txt
$ conda activate alzheimer-deep-learning
```

**3.** Run nii to png converter script:
```bash
$ python src\nii2png.py --disease AD
```

Script arguments:
- `disease`: Type of disease. It could be `AD`, `MCI` or `CN`.

**4.** After converting the images, run training script
```bash
$ python src\train.py
```
