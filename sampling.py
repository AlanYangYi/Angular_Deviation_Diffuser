import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import math
from tqdm import tqdm
from Angular_Deviation_Diffuser.model import FoldingDiff
from Angular_Deviation_Diffuser.util import wrap
import numpy as np
import os
from Angular_Deviation_Diffuser import reconstruct_coordinate as rc
from Angular_Deviation_Diffuser import refine

DEFAULT_MU = [
    -0.0056, -0.0255, 0.1337, 0.0090, -0.0026, 0.0020
]






def generate_conformations(ckpt='model_para.pth',timepoints=1000,num_residues=147,batch_size=10,mu=DEFAULT_MU,output='sample_trajectory.npy',total_samples=10,reference='reference_confor.npy'):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = timepoints
    mu = torch.tensor(mu).float().to(device)
    if not os.path.exists(reference):
        raise FileNotFoundError(f"No reference file found at {reference}")
    reference_ = torch.from_numpy(np.load(reference)).float()
    reference_.nan_to_num_(0.0)

    # Load model
    model = FoldingDiff()

    state_dict = torch.load(ckpt, map_location=torch.device('cpu'))['state_dict']
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()

    s = 8e-3
    t = torch.arange(T + 1)
    f_t = torch.cos((t / T + s) / (1 + s) * math.pi / 2.0).square()
    alpha_bar = f_t / f_t[0]
    beta = torch.cat([torch.tensor([0.0]), torch.clip(1 - alpha_bar[1:] / alpha_bar[:-1], min=1e-5, max=1 - 1e-5)])

    alpha = 1 - beta

    total_batches = math.ceil(total_samples / batch_size)

    trajectory = []
    with torch.no_grad():
        for batch_idx in tqdm(range(total_batches), desc='sampling batches',leave=False):

            current_batch_size = batch_size if batch_idx < total_batches - 1 else total_samples % batch_size or batch_size

            if current_batch_size == 0:
                break

            random_init = torch.randn(current_batch_size, num_residues, 6).to(device)
            x = wrap(random_init)

            for t in tqdm(range(T, 0, -1), desc='sampling', leave=False):

                sigma_t = math.sqrt((1 - alpha_bar[t - 1]) / (1 - alpha_bar[t]) * beta[t])

                # Sample from N(0, sigma_t^2)
                if t > 1:
                    z = torch.randn(current_batch_size, num_residues, 6).to(device) * sigma_t * 0.0015


                else:
                    z = torch.zeros(current_batch_size, num_residues, 6).to(device)

                # Update x
                t_tensor = torch.tensor([t]).long().unsqueeze(0).to(device)
                out = model(x, t_tensor).to(device)
                out_ = 1 / math.sqrt(alpha[t]) * (x - beta[t] / math.sqrt(1 - alpha_bar[t]) * out) + z
                x = wrap(out_.to(device))
                x_ = out_ + reference_.to(device)

                if t == 1:
                    trajectory.append(x_.unsqueeze(1))
    trajectory = wrap(torch.cat(trajectory, dim=1) + mu)
    np.save(output, trajectory.cpu().numpy())

def process_and_refine_conformations(input_file='sample_trajectory.npy', output_dir='output_pdbs'):
   
    data = np.load(input_file) 
    N1, N2, _, _ = data.shape

    
    os.makedirs(output_dir, exist_ok=True)
    refined_pdb_files = []

   
    model_count = 0
    for i in range(N1):
        for j in range(N2):
           
            coor = rc.angles2coord(data[i, j])

            
            pdb_file = os.path.join(output_dir, f'reconstruct_pdb_{model_count}.pdb')
            rc.coor_to_pdb(coor, pdb_file)

          
            refined_pdb_file = os.path.join(output_dir, f'reconstruct_pdb_{model_count}_with_refinement.pdb')
            refine.refine_conformations(pdb_file, refined_pdb_file)
            refined_pdb_files.append(refined_pdb_file)

            model_count += 1

   
    combined_pdb_file = os.path.join(output_dir, 'combined_refined_models.pdb')
    with open(combined_pdb_file, 'w') as combined_file:
        for idx, refined_file in enumerate(refined_pdb_files):
          
            combined_file.write(f"MODEL {idx}\n")

            with open(refined_file, 'r') as pdb_file:
                combined_file.write(pdb_file.read())

           
            combined_file.write("ENDMDL\n")

    print(f"所有优化PDB文件已合并为: {combined_pdb_file}")




def generate_conformations_with_refinement(ckpt='model_para.pth',batch_size=10,total_samples=10):
    generate_conformations(ckpt=ckpt,batch_size=batch_size,total_samples=total_samples)
    process_and_refine_conformations()
    
