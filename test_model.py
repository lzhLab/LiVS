import os
import datetime
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim, nn
from models.FPN_ours import FPN
from utils.dataset_loader_highpass import MyDataset
from torchvision import transforms
from einops import rearrange
import torch.nn.functional as F
from utils.metrics import hausdorff95,Assd,asd,hausdorff,voe,rvd,dice,msd,recall

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(41)


def mkdirs(path):
    if not os.path.exists(path): os.makedirs(path)


def load_checkpoint_model(model, ckpt_best, device):
    state_dict = torch.load(ckpt_best, map_location=device)
    #model.load_state_dict(state_dict['state_dict'])
    model.load_state_dict(state_dict)
    return model


@torch.no_grad()
def test_epoch(model, dl):
    model.eval()
    v_Assd, v_asd, v_hausdorff, v_voe, v_rvd, v_dice, v_msd, ii = 0, 0, 0,0, 0, 0, 0, 0
    for x2, gb, hp, mask in dl:
        outputs = model(x2.float().to(device), gb.float().to(device))
        mask = mask.float().to(device)    
        
        outputs = outputs.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()
        outputs = outputs>0
        v_dice += dice(outputs, mask)
        v_voe += voe(outputs, mask)
        v_rvd += rvd(outputs, mask)
        v_msd += msd(outputs, mask)
        v_hausdorff += hausdorff95(outputs, mask)
        v_asd += asd(outputs, mask)
        v_Assd += Assd(outputs, mask)

        ii += 1
    return v_dice / ii, v_voe / ii, v_rvd / ii, v_msd / ii, v_hausdorff / ii, v_asd / ii, v_Assd / ii


def test():
    model = FPN([3,4,23,3], 1, back_bone="resnet50")
    pth = 'ckpt/20220615091958_setr/ours_ynbce.pth'
    model = load_checkpoint_model(model, pth, device)
    model = model.to(device)
    model = nn.DataParallel(model)
    
    root_dir = '../dataset/LiVS'
    train_image_root = 'train'
    val_image_root = 'val'

    test_dataset = MyDataset(model_type=val_image_root, data_filename=root_dir,sub_name='',transform=transforms.ToTensor())
    test_dl = DataLoader(dataset=test_dataset, batch_size=10, num_workers=8, shuffle=False)

    # logs
    fout_log = open('ours_reslog.txt', 'w')

    print(len(test_dataset))
    v_dice, v_voe, v_rvd, v_msd, v_hausdorff, v_asd, v_Assd = test_epoch(model, test_dl)
    log = "v_dice:%0.5f\n v_voe:%0.5f\n v_rvd:%0.5f\n v_msd:%0.5f\n v_hausdorff:%0.5f\n v_asd:%0.5f\n v_Assd:%.5f" % (
            v_dice, v_voe, v_rvd, v_msd, v_hausdorff, v_asd, v_Assd)
    print(log)
    fout_log.write(log)
    fout_log.close()

    
if __name__ == '__main__':
    test()
