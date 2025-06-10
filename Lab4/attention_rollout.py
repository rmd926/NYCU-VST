import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from timm.models.vision_transformer import Attention
import cv2
import os
import argparse


class AttentionRollout:
    def __init__(self):
        self.model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
        self.attentions = [] # list that should contain attention maps of each layer
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
        self.hooks = []  # Store forward hook handles
        self.original_forwards = {}

        for name, module in self.model.named_modules():
            if isinstance(module, Attention):
                # Store the original forward method
                self.original_forwards[name] = module.forward

                # Bind the attention_forward method to the module instance
                module.forward = self.attention_forward.__get__(module, Attention)

                # Register the forward hook to extract attention weights
                hook_handle = module.register_forward_hook(self.get_attention)
                self.hooks.append(hook_handle)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        # Restore the original forward methods
        for name, module in self.model.named_modules():
            if name in self.original_forwards:
                module.forward = self.original_forwards[name]

    @staticmethod
    def attention_forward(self, x):
        """
        Implements the attention computation and stores the attention maps.
        You need to save the attention map into variable "self.attn_weights"
        
        Note: Due to @staticmethod, "self" here refers to the "Attention" module instance, not the class itself.
        """
        # Step 1: Split input into Query (Q), Key (K), Value (V)
        # Assuming self.qkv is the layer to generate Q, K, V
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

    def get_attention(self, module, input, output):
        self.attentions.append(module.attn_weights.detach().cpu())

    def clear_attentions(self):
        # clear the stored attention weights
        self.attentions = []


    def attention_rollout(self, method='mean', start_layer=1, end_layer=8, discard_ratio=0.65):
        """
        Define the attention rollout function that accumulates the final attention flows.
        You need to return parameter "mask", which should be the final attention flows.
        method: specify the method to fuse attention heads - 'mean', 'max', or 'min'
        start_layer: specify the start layer index to use
        end_layer: specify the end layer index to use
        discard_ratio: ratio of lowest attention values to discard (e.g., 0.9 means discarding the lowest 90%)
        """
        result = torch.eye(self.attentions[0].size(-1))
        with torch.no_grad():
            total_layers = len(self.attentions)
            
            # Ensure the specified layers are within valid bounds
            start_layer = max(0, start_layer)
            end_layer = min(total_layers, end_layer)

            for idx in range(start_layer, end_layer):
                attention = self.attentions[idx]

                # Step 1: Fuse attention heads based on the specified method
                if method == 'mean':
                    fused_attention = torch.mean(attention, dim=1)
                elif method == 'max':
                    fused_attention, _ = torch.max(attention, dim=1)
                elif method == 'min':
                    fused_attention, _ = torch.min(attention, dim=1)
                else:
                    raise ValueError("Invalid method. Choose from 'mean', 'max', or 'min'")

                # Step 2: Optionally discard the lowest attention values
                flat = fused_attention.view(fused_attention.size(0), -1)
                _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, largest=False)
                indices = indices[indices != 0]  # Ensure not to discard the [CLS] token
                flat[0, indices] = 0

                # Step 3: Add scaled skip connection (scaled identity matrix)
                skip_weight = 0.1 # Reduce the weight of the skip connection further
                identity_matrix = torch.eye(fused_attention.size(-1), device=fused_attention.device) * skip_weight
                fused_attention = fused_attention + identity_matrix

                # Step 4: Normalize fused attention map to improve concentration
                fused_attention = fused_attention / torch.clamp(fused_attention.sum(dim=-1, keepdim=True), min=1e-3)

                # Step 5: Matrix multiplication with previous result to accumulate attention
                result = torch.matmul(fused_attention, result)

            # Step 6: Extract and normalize the attention flow mask
            mask = result[0, 0, 1:]  # Extract attention flow from [CLS] token to the rest tokens
            width = int(mask.size(-1) ** 0.5)
            mask = mask.reshape(width, width).numpy()
            mask = mask / np.max(mask)

            # Optional: Filter out low attention areas to enhance focus on high attention regions
            #mask[mask < 0.3] = 0

        return mask



    
    # def attention_rollout(self, method='max', start_layer=3, end_layer=5):# 2.png start layer = 7較佳
    #     """
    #     Define the attention rollout function that accumulates the final attention flows.
    #     You need to return parameter "mask", which should be the final attention flows.
    #     method: specify the method to fuse attention heads - 'mean', 'max', or 'min'
    #     start_layer: specify the start layer index to use
    #     end_layer: specify the end layer index to use
    #     """
    #     result = torch.eye(self.attentions[0].size(-1))
    #     with torch.no_grad():
    #         total_layers = len(self.attentions)
            
    #         # Ensure the specified layers are within valid bounds
    #         start_layer = max(0, start_layer)
    #         end_layer = min(total_layers, end_layer)

    #         for idx in range(start_layer, end_layer):
    #             attention = self.attentions[idx]

    #             # Step 1: Fuse attention heads based on the specified method
    #             if method == 'mean':
    #                 fused_attention = torch.mean(attention, dim=1)
    #             elif method == 'max':
    #                 fused_attention, _ = torch.max(attention, dim=1)
    #             elif method == 'min':
    #                 fused_attention, _ = torch.min(attention, dim=1)
    #             else:
    #                 raise ValueError("Invalid method. Choose from 'mean', 'max', or 'min'")
    #             # Step 2: Add skip connection (scaled identity matrix)
    #             # Adding a scaled identity matrix instead of a full one to reduce impact
    #             skip_weight = 0.03  # Reduce the weight of the skip connection to lessen its impact
    #             identity_matrix = torch.eye(fused_attention.size(-1), device=fused_attention.device) * skip_weight
    #             fused_attention = fused_attention + identity_matrix

    #             # Step 3: Normalize fused attention map to improve concentration
    #             # Increase the minimum value to make the normalization less aggressive
    #             fused_attention = fused_attention / torch.clamp(fused_attention.sum(dim=-1, keepdim=True), min=1e-3)

    #             # Step 4: Matrix multiplication with previous result to accumulate attention
    #             result = torch.matmul(fused_attention, result)

    #             # # Step 2: Add skip connection (identity matrix)
    #             # fused_attention = fused_attention + torch.eye(fused_attention.size(-1))

    #             # # Step 3: Normalize fused attention map to improve concentration
    #             # fused_attention = fused_attention / torch.clamp(fused_attention.sum(dim=-1, keepdim=True), min=1e-2)

    #             # # Step 4: Matrix multiplication with previous result to accumulate attention
    #             # result = torch.matmul(fused_attention, result)

    #         # Step 5: Extract and normalize the attention flow mask
    #         mask = result[0, 0, 1:]  # Extract attention flow from [CLS] token to the rest tokens
    #         width = int(mask.size(-1) ** 0.5)
    #         mask = mask.reshape(width, width).numpy()
    #         mask = mask / np.max(mask)
    #         # mask[mask < 0.4] = 0  # Filter out low attention areas to enhance focus on high attention regions

    #     return mask
    
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

    def run(self, image_path):
        """Do not modify this part"""

        # clean previous attention maps and result
        self.clear_attentions()
        
        # get the image name for saving output image2
        image_name = os.path.basename(image_path)  # e.g., 'image.jpg'
        image_name, _ = os.path.splitext(image_name)  # ('image', '.jpg')
        
        # convert image to a tensor
        img = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(img).unsqueeze(0)  # Add batch dimension

        # run the process of gathering attention flows
        # and put the mask on the input image
        with torch.no_grad():
            outputs = self.model(input_tensor)
            np_img = np.array(img)[:, :, ::-1]
            mask = self.attention_rollout()
            output_heatmap = self.show_mask_on_image(np_img, mask)
            output_filename = f"result_{image_name}.png"  
            cv2.imwrite(output_filename, output_heatmap)
        
        # Remove hooks after running
        #self.remove_hooks()

    


if __name__ == '__main__':
    """Do not modify this part"""

    # arg parsing
    parser = argparse.ArgumentParser(description='Process an image for attention visualization.')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image.')
    args = parser.parse_args()

    # Execution
    model = AttentionRollout()
    with torch.no_grad():
        outputs = model.run(args.image)
