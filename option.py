import argparse


parser = argparse.ArgumentParser(description="VDSPT")


# Hardware specifications
parser.add_argument("--gpu_id", type=str, default='2')

# Data specifications
parser.add_argument('--data_root', type=str, default='../SRdata/arad/arad_dataset', help='dataset directory')
parser.add_argument('--data_name', type=str, default='ARAD', help='dataset directory')
parser.add_argument('--noise', type=str, nargs='+', default=None, help='noise types to apply (e.g., "easy", "medium", "hard")')


# Saving specifications
parser.add_argument('--outf', type=str, default='./exp_noise/', help='saving_path')
parser.add_argument('--model_name', type=str, default='VDSPT', help='model name')
parser.add_argument('--scale', type=int, default=4, help='super-resolution scale')

# Model specifications
parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model directory')
parser.add_argument('--dimention', type=int, default=64, help='feature dimention')
parser.add_argument('--stage', type=int, default=2, help='feature dimention')
parser.add_argument('--in_ch', type=int, default=31, help='input channel')
parser.add_argument('--out_ch', type=int, default=31, help='output channel')

# Training specifications
parser.add_argument('--batch_size', type=int, default=4, help='the number of HSIs per batch')
parser.add_argument('--crop_size', type=int, default=128, help='the size of cropped patch')
parser.add_argument("--max_epoch", type=int, default=2000, help='total epoch')
parser.add_argument("--scheduler", type=str, default='CosineAnnealingLR', help='MultiStepLR or CosineAnnealingLR')
parser.add_argument("--milestones", type=int, default=[50,100,150,200,250], help='milestones for MultiStepLR')
parser.add_argument("--gamma", type=float, default=0.5, help='learning rate decay for MultiStepLR')
parser.add_argument("--learning_rate", type=float, default=0.0004)

opt = parser.parse_args()

for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False
