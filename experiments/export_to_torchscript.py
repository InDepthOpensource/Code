import io
import numpy as np
import argparse
from torch import nn
import torch.onnx

from models import CBAMDilatedUNet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-model', type=str, default='/Users/user/PycharmProjects/DepthCompletion/tmp/best_l1_originalcbam6crop_finetune.pth')
    parser.add_argument('--export-path', type=str, default='/Users/user/PycharmProjects/DepthCompletion/tmp/depth_completion_torchscript.pt')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CBAMDilatedUNet(device, 2, use_cbam_encoder=False, use_cbam_fuse=False, use_cbam_decoder=False)
    model.to(device)
    model.load_state_dict(torch.load(args.load_model, map_location=device))
    model.eval()

    color_input = torch.randn(1, 3, 256, 320, requires_grad=True)
    depth_input = torch.randn(1, 1, 256, 320, requires_grad=True)
    # Export the model
    traced_script_module = torch.jit.trace(model, (color_input, depth_input))
    traced_script_module.save(args.export_path)
