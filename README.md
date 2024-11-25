# Angular Deviation Diffuser



## Overview

**Angular Deviation Diffuser** is a transformer-based diffusion model designed for efficiently generating conformational ensembles of protein backbones by using angular deviations as data flow. It aims to overcome the limitations of traditional molecular dynamics (MD) simulations by providing a fast and computationally efficient approach for sampling protein conformational landscapes. This model leverages the concepts of SE(3) symmetry, angular deviations, and diffusion processes to produce dynamic ensembles that closely match those generated through MD simulations, thereby offering a new way to study protein structure and function.



![Angular Deviation Diffuser Workflow](https://github.com/AlanYangYi/angular_deviation_diffuser/blob/2db3939e7b49b2c8b821c8dd9ac6b1210c9f8f50/Pictures/overview.png)

## Computational results

###  Generated Conformations Example (Dark State and Light State)
![Generated Conformations Example](https://github.com/AlanYangYi/angular_deviation_diffuser/blob/2db3939e7b49b2c8b821c8dd9ac6b1210c9f8f50/Pictures/Dark_and_light_generated_by_our_model.gif)

### Using absolute angles vs. using angle deviations for the denoising process(Sampling Process)
![Using angles V.S. Using angle deviation](https://github.com/AlanYangYi/angular_deviation_diffuser/blob/2db3939e7b49b2c8b821c8dd9ac6b1210c9f8f50/Pictures/angleVSanglechange.gif)


## Background

Protein dynamics are essential for understanding biological functionality, as proteins exist not only in a single static structure but also in multiple dynamic conformational states. MD simulations are the gold standard for studying these dynamics, but they are resource-intensive and limited in their ability to fully explore all possible conformational states. The Angular Deviation Diffuser addresses these limitations by utilizing advanced deep learning techniques, specifically a diffusion model integrated with SE(3) invariance, to efficiently generate accurate protein conformations.

## Features

- **Angular Deviation-Based Diffusion**: Uses angular deviations rather than absolute angles for data representation, improving stability and efficiency.
- **Transformer Backbone**: Utilizes a transformer architecture for learning protein dynamics from training data, capturing the conformational space effectively.
- **SE(3) Symmetry Integration**: Ensures the generated conformations respect the inherent rotational and translational symmetry of molecular systems.
- **Efficient Ensemble Generation**: Capable of generating diverse conformational ensembles in significantly less time compared to traditional MD simulations.

## Installation

To install and use Angular Deviation Diffuser, follow these steps:

### Prerequisites

- **Conda**: Ensure that [Conda](https://docs.conda.io/en/latest/miniconda.html) is installed to manage the environment and dependencies.

### Steps

1. **Create and Activate Conda Environment**:
   
   ```bash
   conda create -n angular_deviation_diffusion python==3.8
   conda activate angular_deviation_diffusion
   ```

2. **Install Angular Deviation Diffuser**:
   
   ```bash
   pip install Angular_Deviation_Diffuser
   python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'
   ```

## Usage

The following sections provide detailed guidance on how to use the package for generating protein conformations:

### 1. Extract Six Types of Angles

Extract the backbone angles (ϕ, ψ, ω, θ₁, θ₂, θ₃) from a given PDB file.

```python
from Angular_Deviation_Diffuser import extract_six_angles

angles = extract_six_angles.get_angle_from_pdb("your_pdb_file.pdb")
```
Replace `"your_pdb_file.pdb"` with the appropriate PDB file name. This function returns an angle matrix containing all six types of backbone angles.

### 2. Reconstruct 3D Coordinates and Generate a PDB File

Reconstruct the 3D atomic coordinates using the six angle types and generate the corresponding PDB file.

```python
from Angular_Deviation_Diffuser import reconstruct_coordinate

# Given an L x 6 angle matrix, reconstruct the Cartesian coordinates of the atoms.
# Replace 'angles' with the actual angle data in a numpy array type.
coor = reconstruct_coordinate.angles2coord(angles)

# Save the reconstructed coordinates to a PDB file
reconstruct_coordinate.coor_to_pdb(coor, "reconstructed_structure.pdb")
```
Replace `"reconstructed_structure.pdb"` with the desired output PDB file name.

### 3. Training the Model

Train the transformer-based diffusion model using the angular deviation data obtained from the previous step.

```bash
python training.py --data_dir 'path/to/training_data'
```
Replace `'path/to/training_data'` with the path to your training dataset. The training set can be downloaded from [here](https://github.com/AlanYangYi/angular_deviation_diffuser/blob/2eaf4d98dacb188eeeb56005ff526e1130f02dc3/Training_Set.zip).

### 4. Generating Conformations

Generate a diverse ensemble of protein backbone conformations using the trained model.

```python
from Angular_Deviation_Diffuser import sampling

# Generate conformations with refinement
sampling.generate_conformations_with_refinement(batch_size=10, total_samples=10)
```
The pre-trained model weights can be downloaded from [here](https://drive.usercontent.google.com/download?id=1ld2lZgJFoZJZrwbKdzHcDAZU7t9jmBkH&export=download&authuser=0&confirm=t&uuid=17381f09-3b32-4d7f-976d-1d49de944a7f&at=AENtkXZcdxy-fTKXegNBrIdxcF-T:1731452378968).

### 5. Adding Side Chains with Refinement

Utilize PyRosetta to add side chains to the generated backbone structures and refine them.

```python
from Angular_Deviation_Diffuser import refine

# Refine the generated backbone structure
refine.refine_conformations('reconstructed_structure.pdb', "refined_conformation.pdb")
```
Replace `'reconstructed_structure.pdb'` and `"refined_conformation.pdb"` with the appropriate file names for input and output.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Acknowledgements

We are grateful to our research team for their invaluable contributions and support throughout the development of this model.


---

If you have any issues or questions, please feel free to open an issue in the repository or contact us directly.


