import ttach as tta
import cv2
from cv2 import cv2
import numpy as np


class AbstractGradCAM:
  def __init__(self, model, target_layer, use_cuda=False):
    self.model = model.eval()
    self.target_layer = target_layer
    self.cuda = use_cuda

    if self.cuda:
      self.model = model.cuda()
    
    self.activations = []
    self.gradients = []
    # for more information about hooks https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks
    target_layer.register_forward_hook(self.save_activation)
    target_layer.register_full_backward_hook(self.save_gradient)

  def forward(self, input_tensor, target_category=None):
    if self.cuda:
        input_tensor = input_tensor.cuda()
    
    self.activations = []
    self.gradients = []
    output_layer = self.model(input_tensor)

    if target_category is None:
      target_category = np.argmax(output_layer.cpu().data.numpy(), axis=-1)
    elif type(target_category) is int:
      target_category = [target_category]
    
    self.model.zero_grad()
    loss = self.get_loss(output_layer, target_category)
    loss.backward(retain_graph=True)
    #creating a NumPy array from the tensor
    activations = self.activations[-1].cpu().data.numpy()
    grads = self.gradients[-1].cpu().data.numpy()
    # calculating weights and activations to perforom __cam eqn
    weights = self.get_weights(activations, grads)
    weighted_activations = weights[:, :, None, None] * activations
    cam = weighted_activations.sum(axis=1)
    # RelU
    cam = np.maximum(cam, 0)
    # if more than one image is passed, you need to loop over cam list
    res = []
    for img in cam:
      img = cv2.resize(img, input_tensor.shape[-2:][::-1])
      img = img - np.min(img)
      img = img / np.max(img)
      res.append(img)
    res = np.float32(res)
    return res
  
  def get_weights(self, activations, grads):
    pass
  ###
  def get_loss(self, output_layer, target_category):
      loss = 0
      for i in range(len(target_category)):
          loss += output_layer[i, target_category[i]]
      return loss

  def save_activation(self, module, input, output_layer):
      activation = output_layer
      self.activations.append(activation.cpu().detach())

  def save_gradient(self, module, grad_input, grad_output_layer):
      # Gradients are computed in reverse order
      grad = grad_output_layer[0]
      self.gradients = [grad.cpu().detach()] + self.gradients
  ###

  def __call__(self, input_tensor, target_category=None):
      return self.forward(input_tensor, target_category)

class GradCAM(AbstractGradCAM):
  def __init__(self, model, target_layer, use_cuda=False):
    super(GradCAM, self).__init__(model, target_layer, use_cuda)

  def get_weights(self, activations, grads):
    return np.mean(grads, axis=(2, 3))

class GradCAMPlusPlus(AbstractGradCAM):
  def __init__(self, model, target_layer, use_cuda=False):
    super(GradCAMPlusPlus, self).__init__(model, target_layer, use_cuda)

  def get_weights(self, activations, grads):
    alphas = grads**2 / (2*(grads**2) + np.sum(activations, axis=(2, 3))[:, :, None, None]*(grads**3))
    weights = np.maximum(grads, 0)*alphas
    weights = np.sum(weights, axis=(2, 3))
    return weights

'''
Inspired by https://github.com/jacobgil/pytorch-grad-cam
'''

def show_cam_on_image(img: np.ndarray, mask: np.ndarray, use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)