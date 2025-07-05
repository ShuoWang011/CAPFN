'''
import torch
import torchvision
import os
x = torch.randn(1,3,473,473)
y = torch.randn(1,500,500)
z = torch.randn(1,2,500,500)
torchvision.utils.save_image(x, os.path.join(r'C:\Users\zyr\Desktop\Natural_image\Projects\BAM-main\BAM\exp\pascal\BAM\split0\resnet50\result\test_img', 'x.jpg'))
torchvision.utils.save_image(y, os.path.join(r'C:\Users\zyr\Desktop\Natural_image\Projects\BAM-main\BAM\exp\pascal\BAM\split0\resnet50\result\test_img', 'y.jpg'))
torchvision.utils.save_image(z, os.path.join(r'C:\Users\zyr\Desktop\Natural_image\Projects\BAM-main\BAM\exp\pascal\BAM\split0\resnet50\result\test_img', 'z.jpg'))


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Few-Shot Semantic Segmentation')
    parser.add_argument('--arch', type=str, default='BAM') #
    parser.add_argument('--viz', action='store_true', default=False)
    parser.add_argument('--config', type=str, default=r'C:\Users\zyr\Desktop\Natural_image\Projects\BAM-main\BAM\config/pascal/pascal_split0_vgg.yaml', help='config file') # coco/coco_split0_resnet50.yaml
    parser.add_argument('--local_rank', type=int, default=-1, help='number of cpu threads to use during batch generation')
    parser.add_argument('--opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg = config.merge_cfg_from_args(cfg, args)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg
args = get_parser()
x = torch.ones(4,3,40,40)
model = OneModel(args, cls_type='Base')
y = model(x)
print(y.shape)
'''

"""
将tensor保存为cv2格式
:param input_tensor: 要保存的tensor
:param filename: 保存的文件名

assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
# 复制一份
input_tensor = input_tensor.clone().detach()
# 到cpu
input_tensor = input_tensor.to(torch.device('cpu'))
# 反归一化
# input_tensor = unnormalize(input_tensor)
# 去掉批次维度
input_tensor = input_tensor.squeeze()
# 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
# RGB转BRG
input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
cv2.imwrite(os.path.join(r'C:\Users\zyr\Desktop\Natural_image\Projects\BAM-main\BAM\exp\pascal\BAM\split0\resnet50\result\test_img', 'i{}.jpg'.format(num)), input_tensor)
"""