"""
Test Cross Dataset
    For help
    ```bash
    python test_cross_dataset.py --help
    ```
 Date: 2018/9/20
"""








from argparse import ArgumentParser
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from main import RandomCropPatches, VarianceThresholdPatchSelection, NRnet
import numpy as np
import h5py, os , random


if __name__ == "__main__":
    parser = ArgumentParser(description='PyTorch WaDIQaM-FR test on the whole cross dataset')
    parser.add_argument("--dist_dir", type=str, default=None,
                        help="distorted images dir.")
    parser.add_argument("--ref_dir", type=str, default=None,
                        help="reference images dir.")
    parser.add_argument("--names_info", type=str, default=None,
                        help=".mat file that includes image names in the dataset.")
    parser.add_argument("--model_file", type=str, default='checkpoints/WaDIQaM-FR-KADID-10K-EXP1000-5-lr=0.0001-bs=4',
                        help="model file (default: checkpoints/WaDIQaM-FR-KADID-10K-EXP1000-5-lr=0.0001-bs=4)")
    parser.add_argument("--save_path", type=str, default='scores',
                        help="save path (default: scores)")

    parser.add_argument("--patching_method", type=str, default='random',
                        help="random  or variance")

    parser.add_argument("--number_of_samples", type=int, default=500,
                        help="number of random images during cross dataset. (-1 : for all images in  dataset) (default: 500)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NRnet(weighted_average=True).to(device)

    model.load_state_dict(torch.load(args.model_file))

    NUMBER_OF_SAMPLES = args.number_of_samples

    Info = h5py.File(args.names_info, 'r')
    im_names = [Info[Info['im_names'][0, :][i]][()].tobytes()\
                        [::2].decode() for i in range(len(Info['im_names'][0, :]))]

    if NUMBER_OF_SAMPLES != -1 : 

        rands = random.sample(range(0, len(im_names)-1), NUMBER_OF_SAMPLES)
        tmp = []
        for i in rands:
            tmp.append(im_names[i])

        im_names = tmp

        with open("tmp_imgIndex.txt", "w") as f:
            for s in rands:
                f.write(str(s) +"\n")
    
    else:
        with open("tmp_imgIndex.txt", "w") as f:
            for s in range(len(im_names)):
                f.write(str(s) +"\n")



    print(len(im_names),"@@@@@@@@@@@@  LEN  @@@@@@@@@@@@@@",im_names)
    ref_names = [Info[Info['ref_names'][0, :][i]][()].tobytes()\
                        [::2].decode() for i in (Info['ref_ids'][0, :]-1).astype(int)]

    model.eval()
    scores = []   
    with torch.no_grad():
        for i in range(len(im_names)):
            im = Image.open(os.path.join(args.dist_dir, im_names[i])).convert('RGB')
            # ref = Image.open(os.path.join(args.ref_dir, ref_names[i])).convert('RGB')
            # data = RandomCropPatches(im, ref)

            if args.patching_method == "random":
                data = RandomCropPatches(im)
                print("random----------------------")

            if args.patching_method == "variance":
                data = VarianceThresholdPatchSelection(im)
                print("variance+++++++++++++++++++++")

            
            
            dist_patches = data.unsqueeze(0).to(device)
            # ref_patches = data[1].unsqueeze(0).to(device)
            # score = model((dist_patches, ref_patches))

            score = model((dist_patches))
            scores.append(score.item())
    np.save(args.save_path, scores)