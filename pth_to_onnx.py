import torch
import torchvision
#model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
model.eval()

with torch.no_grad():
    x = torch.randn(1,3,416,416)
    torch.onnx.export(model, x, "mask_rcnn.onnx", opset_version = 11, verbose = True)
#x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
#predictions = model(x)
#torch.onnx.export(model, x, "mask_rcnn.onnx", opset_version = 11)
