import thop
from model import BAM_SCL_DCPPP_reloss3_nbs8lr0004
import torch
import argparse
from util import config
def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Few-Shot Semantic Segmentation')
    parser.add_argument('--arch', type=str, default='BAM_SCL_DCPPP_reloss3_nbs8lr0004') #
    parser.add_argument('--viz', action='store_true', default=False)
    parser.add_argument('--config', type=str, default='config/coco/coco_split0_resnet50.yaml', help='config file') # coco/coco_split0_resnet50.yamlpascal/pascal_split0_vgg.yaml
    parser.add_argument('--local_rank', type=int, default=-1, help='number of cpu threads to use during batch generation')
    parser.add_argument('--opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg = config.merge_cfg_from_args(cfg, args)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg
def main():
    global args, logger, writer
    args = get_parser()
    model = eval(args.arch).OneModel(args, cls_type='Base')
    s_input = torch.randn((1,1,3,473,473))
    s_mask = torch.randn((1,1,473,473))
    input = torch.randn((1,3,473,473))
    target = torch.randn((1,473,473))
    target_b= torch.randn((1,473,473))
    subcls=[1, 9, 6, 10, 9, 10, 10, 9]


    flops, params = thop.profile(model, inputs=(input,s_input, s_mask,  target,target_b, subcls,))
    flops, params = thop.clever_format([flops, params], '%.3f')
    print('flops:', flops)
    print('params:', params)

if __name__ == '__main__':
    main()
