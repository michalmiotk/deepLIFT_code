import torch,  torchvision
from torchvision import transforms
from PIL import Image

class SomeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.vgg19(pretrained=True)
        self.model.eval()
        self.baseline = torch.zeros((1,3,224,224))
        self.baseline.requires_grad = True
        self.transform = transforms.Compose(
                        [transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        transforms.Resize((224,224))])

        self.im = Image.open('elephant.jpeg')
        self.im = self.transform(self.im)
        self.im = self.im.unsqueeze_(0)
        self.im.requires_grad = True
        self.name_activations={"ReLU", "AdaptiveAvgPool2d"}
        self.forward_handles = []

    def forward_ref_hook_base(self, module, inputs, outputs):
        print("ref", module)
        setattr(module, "ref_inputs", inputs)
        setattr(module, "ref_outputs", outputs)
    
    def forward_original_hook_base(self, module, inputs, outputs):
        print("ofig", module)
        setattr(module, "orig_inputs", inputs)
        setattr(module, "orig_outputs", outputs)
    
    def unregister_forward_hooks(self):
        for forward_handle in self.forward_handles:
            forward_handle.remove()

    def back_hook_base(self, module, grad_input, grad_output):
        print("bbback", module._get_name())
        if module._get_name() == "ReLU":
            y_dif = module.orig_outputs[0] - module.ref_outputs[0]
            x_dif = module.orig_inputs[0] - module.ref_inputs[0]
            
            grad_input = torch.where(abs(x_dif)<1e-10, grad_input[0], grad_output[0]*y_dif/x_dif)
            #print("grad_input",grad_input)
            return (grad_input,)

    def register_hook_base(self, ref:bool=False):
        for some_module in self.model.children():
            i=-1
            for i, el in enumerate(some_module.children()):
                if el._get_name() in self.name_activations:
                    if ref:
                        self.forward_handles.append(el.register_forward_hook(self.forward_ref_hook_base))
                    else:
                        self.forward_handles.append(el.register_forward_hook(self.forward_original_hook_base))
                        el.register_backward_hook(self.back_hook_base)
            if i == -1:
                if some_module._get_name() in self.name_activations:
                    if ref:
                        self.forward_handles.append(some_module.register_forward_hook(self.forward_ref_hook_base))
                    else:
                        self.forward_handles.append(some_module.register_forward_hook(self.forward_original_hook_base))
                        some_module.register_backward_hook(self.back_hook_base)

    def interpret(self):
        target = torch.argmax(self.model(self.im))
        print(self.model)
        self.register_hook_base(ref=True)
        self.model(self.baseline)
        self.unregister_forward_hooks()
        self.register_hook_base(ref=False)
        self.model(self.im)
        self.unregister_forward_hooks()
        grads = torch.autograd.grad(self.model(self.im)[:,target], self.im, torch.ones((1)))
        print(grads[0].shape,  len(grads))
        grad = grads[0]/grads[0].max()
        torchvision.utils.save_image(grad, "out_without.jpeg")
SomeModel().interpret()