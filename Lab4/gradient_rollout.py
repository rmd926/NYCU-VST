import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from timm.models.vision_transformer import Attention
import cv2
import os
import argparse

class GradientRollout:
    def __init__(self):
        self.model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = True
        self.attentions = []
        self.attention_grads = []
        self.num_heads = 3
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),  # ImageNet mean
                std=(0.229, 0.224, 0.225)    # ImageNet std
            )
        ])
        # Replace the 'forward' method of all 'Attention' modules 
        for name, module in self.model.named_modules():
            if isinstance(module, Attention):
                # Bind the attention_forward method to the module instance
                module.forward = self.attention_forward.__get__(module, Attention)

                # Register the forward hook to extract attention weights
                module.register_forward_hook(self.get_attention)

    @staticmethod
    def attention_forward(self, x):
        """

        TODO:
            Implement the attention computation and store the attention maps.
            You need to save the attention map into variable "self.attn_weights"

            Note: Due to @staticmethod, "self" here refers to the "Attention" module instance, not the class itself.

        """
        B, N, C = x.shape  # B: batch size, N: number of tokens, C: channels
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # Step 2: Compute scaled dot-product attention scores
        attn_scores = (Q @ K.transpose(-2, -1)) * (1.0 / (K.shape[-1] ** 0.5))

        # Step 3: Apply softmax to obtain attention weights
        attn_weights = attn_scores.softmax(dim=-1)

        # Step 4: Store attention weights
        self.attn_weights = attn_weights

        # Step 5: Compute the output
        attn_output = (attn_weights @ V).transpose(1, 2).reshape(B, N, C)

        return self.proj(attn_output)
    
    # Define a hook function to extract attention weights
    def get_attention(self, module, input, output):
        # Append the attention weights
        self.attentions.append(module.attn_weights.detach().cpu())

        # Retain gradients on attention weights
        module.attn_weights.retain_grad()

        # Register a hook on attn_weights to save the gradients during backward pass
        def save_attn_grad(grad):
            self.attention_grads.append(grad.cpu())
        module.attn_weights.register_hook(save_attn_grad)

    def clear_attentions(self):
        # clear the stored attention weights
        self.attentions = []
        self.attention_grads = []
    
    
    def gradient_rollout(self, method='mean'):
        """
        Define the gradient rollout function that accumulates the final attention flows.
        You need to return parameter "mask", which should be the final attention flows.
        method: specify the method to fuse attention heads - 'mean', 'max', or 'min'
        """
        result = torch.eye(self.attentions[0].size(-1))

        with torch.no_grad():
            for idx in range(len(self.attentions)):
                attention = self.attentions[idx]
                grad = self.attention_grads[idx]

                # Step 1: Perform element-wise multiplication of attention map and its gradient
                grad_attention = attention * grad

                # Step 2: Fuse attention heads based on the specified method
                if method == 'mean':
                    fused_attention = torch.mean(grad_attention, dim=1)
                elif method == 'max':
                    fused_attention = torch.max(grad_attention, dim=1)[0]
                elif method == 'min':
                    fused_attention = torch.min(grad_attention, dim=1)[0]
                else:
                    raise ValueError("Invalid method. Choose from 'mean', 'max', or 'min'")

                # Step 3: Add identity matrix for skip connection
                skip_connection = torch.eye(fused_attention.size(-1))
                fused_attention = fused_attention + skip_connection

                # Step 4: Wipe out all negative values to 0 (ReLU operation)
                fused_attention = torch.relu(fused_attention)

                # Step 5 (optional): Normalize fused attention map
                fused_attention = fused_attention / torch.clamp(fused_attention.sum(dim=-1, keepdim=True), min=1e-6)

                # Step 6: Matrix multiplication with previous result to accumulate attention
                result = torch.matmul(fused_attention, result)

        # Step 7: Extract attention flow from [CLS] token to the rest tokens
        mask = result[0, 0, 1:]

        # Step 8: Normalize the mask values to range [0.0, 1.0]
        width = int(mask.size(-1) ** 0.5)
        mask = mask.reshape(width, width).numpy()
        mask = mask / np.max(mask)

        return mask

    def perform_backpropagation(self, input_tensor, target_category):
        """

        TODO:
            In order to get gradients, you need to do backpropagation mannually.
            You can follow the below procedures:

            1.Inference the model
            2.define a loss simple function that focus on the target category
            2-1. The loss function can be sum of the logits of target category or
            2-2. cross entropy of target category
            3.perform backpropagation with respect to the loss.

            This function will be invoked in "run" function.

        """

        # write your code here to perform backpropagation
        # Step 1: Forward pass
        
        outputs = self.model(input_tensor)

        # Step 2: Define a loss function that focuses on the target category score
        loss = outputs[0, target_category]

        # Step 3: Calculate loss and perform backward pass to calculate gradients
        self.model.zero_grad()
        loss.backward()
        
    def show_mask_on_image(self, img, mask):

        """Do not modify this part"""

        # Normalize the value of img to [0.0, 1.0]
        img = np.float32(img) / 255

        # Reshape the mask to 224x224 for later computation
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

        # Generate heatmap and normalize the value
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255

        # Add heatmap and original image together and normalize the value
        combination = heatmap + np.float32(img)
        combination = combination / np.max(combination)
        
        # Scale back the value to [0.0, 255.0]
        combination = np.uint8(255 * combination)

        return combination
    
    def run(self, image_path, target_category):

        """Do not modify this part"""

        # clean previous attention maps and result
        self.clear_attentions()
        
        # get the image name for saving output image2
        image_name = os.path.basename(image_path)  # e.g., 'image.jpg'
        image_name, _ = os.path.splitext(image_name)  # ('image', '.jpg')
        
        # convert image to a tensor
        img = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(img).unsqueeze(0)  # Add batch dimension
        input_tensor.requires_grad_(True)

        # do backpropagation manually during inference to produce gradients
        self.perform_backpropagation(input_tensor, target_category)

        np_img = np.array(img)[:, :, ::-1]
        mask = self.gradient_rollout()
        output_heatmap = self.show_mask_on_image(np_img, mask)
        output_filename = f"gradient_result_{image_name}.png"  
        cv2.imwrite(output_filename, output_heatmap)


if __name__ == '__main__':

    """Do not modify this part"""

    # arg parsing
    parser = argparse.ArgumentParser(description='Process an image for attention visualization with respect to target category.')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--category', type=int, required=True, help="target category of attention flows")
    args = parser.parse_args()

    # Execution
    model = GradientRollout()
    outputs = model.run(args.image, args.category)
