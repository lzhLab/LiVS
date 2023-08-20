import os
import datetime
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim, nn
#from dataset_gabor import MyDataset
from utils.dataset_loader_highpass import MyDataset
from models.FPN_LSF import FPN_LSF
from torchvision import transforms
from einops import rearrange
import torch.nn.functional as F
import pdb

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(41)


def mkdirs(path):
    if not os.path.exists(path): os.makedirs(path)


def dice_metric(output, target):
    output = output > 0
    dice = ((output * target).sum() * 2+0.1) / (output.sum() + target.sum() + 0.1)
    return dice

def load_checkpoint_model(model, ckpt_best, device):
    state_dict = torch.load(ckpt_best, map_location=device)
    #model.load_state_dict(state_dict['state_dict'])
    model.load_state_dict(state_dict)
    return model

def voe_metric(output, target):
    output = output > 0
    voe = ((output.sum() + target.sum()-(target*output).sum().float()*2)+0.1) / (output.sum() + target.sum()-(target*output).sum().float() + 0.1)
    return voe.item()

def rvd_metric(output, target):
    output = output > 0
    rvd = ((output.sum() / (target.sum() + 0.1) - 1) * 100)
    return rvd.item()

def acc_m(output,target):
    output = (output>0).float()
    target, output = target.view(-1), output.view(-1)
    acc = (target==output).sum().float() / target.shape[0]
    return acc

def sen_m(output,target):
    output = (output>0).float()
    target, output = target.view(-1), output.view(-1)
    p = (target*output).sum().float()
    sen = (p+0.1) / (output.sum()+0.1)
    return sen

def spe_m(output,target):
    output = (output>0).float()
    target, output = target.view(-1), output.view(-1)
    tn = target.shape[0] - (target.sum() + output.sum() - (target*output).sum().float())
    spe = (tn+0.1) / (target.shape[0] - output.sum()+0.1)
    return spe


def gabor_loss(input, target, gb_weight, epoch=0, reduction='mean', beta=0.75, eps=1e-5):
    n = input.size(0)
    iflat = torch.sigmoid(input).view(n, -1).clamp(eps, 1 - eps)
    gb = gb_weight.view(n, -1).clamp(eps, 1 - eps).float()
    
    gama = 0.1
    gb = gb.clamp(min=gama, max=1.0)
    
    tflat = target.view(n, -1)

    #focal = -(tflat * gb * iflat.log() +
    #          (1 - tflat) * (1-gb) * (1 - iflat).log()).mean(-1)

    focal = -(beta * tflat * gb * iflat.log() +
              (1 - beta) * (1 - tflat) * gb * (1 - iflat).log()).mean(-1)
    if torch.isnan(focal.mean()) or torch.isinf(focal.mean()):
        pdb.set_trace()
    if reduction == 'mean':
        return focal.mean()
    elif reduction == 'sum':
        return focal.sum()
    else:
        return focal

def gabor_bce_with_logits(input, target, weight=None, epoch=1):
    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    #print('==',loss)

    alph = epoch // 5
    bata = 0.5 - 0.1*alph
    if bata<0:
        bata=0.01

    weight = weight.clamp(min=bata, max=1.0)

    if weight is not None:
        loss = loss * weight
    #print('===', loss)
    return loss.mean()


def train_epoch(epoch, model, dl, optimizer, criterion):
    model.train()
    bar = tqdm(dl)
    bar.set_description_str("%02d" % epoch)
    loss_v, dice_v, loss_pre, loss_sur, ii = 0, 0, 0, 0, 0
    #for x2, mask in bar:
    for x2, gb, hp, mask in bar:
        #x2 = rearrange(x2.float(),'b h w c -> b c h w')
        x2 = x2.float().to(device)
        gb = gb.float().to(device)
        hp = hp.float().to(device)
        mask = mask.float().to(device)
        outputs = model(x2,hp+gb)
        loss1 = criterion(outputs, mask)
        loss2 = gabor_loss(outputs, mask, gb)
        loss = loss1 + loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        dice = dice_metric(outputs, mask)
        dice_v += dice
        loss_v += loss.item()
        loss_pre += 0#loss3.item()
        loss_sur += 0#loss2.item()
        ii += 1
        bar.set_postfix(loss=loss.item(), dice=dice.item())
    return loss_v / ii, dice_v / ii, loss_pre / ii, loss_sur / ii


@torch.no_grad()
def val_epoch(model, dl, criterion):
    model.eval()
    loss_v, dice_v, voe_v, rvd_v,acc_v, sen_v, spe_v, ii = 0, 0, 0,0, 0, 0, 0, 0
    for x2, gb, hp, mask in dl:
        x2 = x2.float().to(device)
        gb = gb.float().to(device)
        hp = hp.float().to(device)
        mask = mask.float().to(device)
        outputs = model(x2,hp+gb)

        loss_v += criterion(outputs, mask).item()
        dice_v += dice_metric(outputs, mask)
        voe_v += voe_metric(outputs, mask)
        rvd_v += rvd_metric(outputs, mask)
        acc_v += acc_m(outputs, mask)
        sen_v += sen_m(outputs, mask)
        spe_v += spe_m(outputs, mask)

        ii += 1
    return loss_v / ii, dice_v / ii, voe_v / ii, rvd_v / ii, acc_v / ii, sen_v / ii, spe_v / ii


def train(opt):
    model = FPN_LSF([3,4,23,3], 1, back_bone="resnet50")

    model = model.to(device)
    model = nn.DataParallel(model)
    
    root_dir = opt.dataset_path
    train_image_root = 'train'
    val_image_root = 'val'

    train_dataset = MyDataset(model_type=train_image_root, data_filename=root_dir,sub_name='',transform=transforms.ToTensor())
    train_dl = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, num_workers=8, shuffle=True)
    val_dataset = MyDataset(model_type=val_image_root, data_filename=root_dir,sub_name='',transform=transforms.ToTensor())
    val_dl = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, num_workers=8, shuffle=False)

    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # logs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6,patience=6)
    best_dice_epoch, best_dice, b_voe, b_rvd, train_loss, train_dice, b_acc, b_sen, b_spe,pre_loss, sur_loss =  0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0
    save_dir = os.path.join(opt.ckpt, datetime.datetime.now().strftime('%Y%m%d%H%M%S')) + "_" + opt.name
    mkdirs(save_dir)

    w_dice_best = os.path.join(save_dir, 'ours_ynbce.pth')

    fout_log = open(os.path.join(save_dir, 'ours_ynbce.txt'), 'w')
    print(len(train_dataset), len(val_dataset), save_dir)
    for epoch in range(opt.max_epoch):
        if not opt.eval:
            train_loss, train_dice, pre_loss,sur_loss = train_epoch(epoch, model, train_dl, optimizer, criterion)
        val_loss, val_dice, voe_v, rvd_v, acc_v, sen_v, spe_v = val_epoch(model, val_dl, criterion)
        if best_dice < val_dice:
            best_dice, best_dice_epoch, b_voe, b_rvd,b_acc, b_sen, b_spe = val_dice, epoch, voe_v, rvd_v, acc_v, sen_v, spe_v
            torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), w_dice_best)
        
        lr = optimizer.param_groups[0]['lr']
        log = "%02d train_loss:%0.3e, train_dice:%0.5f,pre_loss:%0.3e,sur_loss:%0.3e, val_loss:%0.3e, val_dice:%0.5f, lr:%.3e\n best_dice:%.5f, voe:%.5f, rvd:%.5f, acc:%.5f, sen:%.5f, spe:%.5f(%02d)\n" % (
            epoch, train_loss, train_dice, pre_loss, sur_loss, val_loss, val_dice, lr, best_dice, b_voe, b_rvd, b_acc, b_sen, b_spe, best_dice_epoch)
        print(log)
        fout_log.write(log)
        fout_log.flush()
        scheduler.step(val_loss)
        #cur = cur + 1
    fout_log.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='setr', help='study name')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--input_size', type=int, default=512, help='input size')
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--ckpt', type=str, default='ckpt', help='the dir path to save model weight')
    parser.add_argument('--w', type=str, help='the path of model wight to test or reload')
    parser.add_argument('--suf', type=str, choices=['.dcm', '.JL', '.png'], help='suffix', default='.png')
    parser.add_argument('--eval', action="store_true", help='eval only need weight')
    parser.add_argument('--test_root', type=str, help='root_dir')
    parser.add_argument('--dataset_path', type=str, help='dataset_dir')

    opt = parser.parse_args()
    train(opt)
