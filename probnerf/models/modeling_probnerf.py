# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
from accelerate.logging import get_logger

from .embedder import CameraEmbedder
from .rendering.synthesizer import TriplaneSynthesizer
from .gaussian_transform import GaussianTransform
from .realnvp import RealNVP
from .hypernetwork.hypernetwork import HyperNetwork
from .hypernetwork.hypernetwork_integration import apply_parameters
from .triplane_generate import TriplaneGenerator
from .nerf.nerf_networks import NeRFDensity, NeRFColor

logger = get_logger(__name__)


class ModelProbNeRF(nn.Module):
    """
    Full model of the basic single-view large reconstruction model.
    """
    def __init__(self, camera_embed_dim: int, rendering_samples_per_ray: int,
                 triplane_generate_dim: int,
                 gaussian_dim: int, # Dimension get out of Gaussian Transform Process
                 triplane_low_res: int, triplane_high_res: int, triplane_dim: int,
                 encoder_freeze: bool = True, encoder_type: str = 'dino',
                 encoder_model_name: str = 'facebook/dino-vitb16', encoder_feat_dim: int = 768,
                 alpha: float = 0.05,
                 enable_perturbations: bool = False):
        super().__init__()
        
        # attributes
        self.encoder_feat_dim = encoder_feat_dim
        self.camera_embed_dim = camera_embed_dim
        self.triplane_low_res = triplane_low_res
        self.triplane_high_res = triplane_high_res
        self.triplane_dim = triplane_dim
        self.alpha = alpha  # Perturbation strength
        self.enable_perturbations = enable_perturbations

        # modules
        self.encoder = self._encoder_fn(encoder_type)(
            model_name=encoder_model_name,
            freeze=encoder_freeze,
        )
        self.camera_embedder = CameraEmbedder(
            raw_dim=12+4, embed_dim=camera_embed_dim,
        )
        # initialize pos_embed with 1/sqrt(dim) * N(0, 1)
        #self.pos_embed = nn.Parameter(torch.randn(1, 3*triplane_low_res**2, transformer_dim) * (1. / transformer_dim) ** 0.5)

        self.upsampler = nn.ConvTranspose2d(triplane_generate_dim, triplane_dim, kernel_size=2, stride=2, padding=0)
        self.synthesizer = TriplaneSynthesizer(
            triplane_dim=triplane_dim, samples_per_ray=rendering_samples_per_ray,
        )
        self.gaussian_transform = GaussianTransform(output_dim=gaussian_dim)
        self.realnvp = RealNVP(input_dim=gaussian_dim, hidden_dim=512, num_blocks=4)
        self.hypernetwork = HyperNetwork()
        self.nerf_density = NeRFDensity(input_channels=63)
        self.nerf_color = NeRFColor(input_channels=127)
        self.triplane_generator = TriplaneGenerator(output_dim=triplane_generate_dim)
    
    @staticmethod
    def _encoder_fn(encoder_type: str):
        encoder_type = encoder_type.lower()
        assert encoder_type in ['dino', 'dinov2'], "Unsupported encoder type"
        if encoder_type == 'dino':
            from .encoders.dino_wrapper import DinoWrapper
            logger.info("Using DINO as the encoder")
            return DinoWrapper
        elif encoder_type == 'dinov2':
            from .encoders.dinov2_wrapper import Dinov2Wrapper
            logger.info("Using DINOv2 as the encoder")
            return Dinov2Wrapper

    #def forward_transformer(self, image_feats, camera_embeddings):
    #    assert image_feats.shape[0] == camera_embeddings.shape[0], \
    #        "Batch size mismatch for image_feats and camera_embeddings!"
    #    N = image_feats.shape[0]
    #    x = self.pos_embed.repeat(N, 1, 1)  # [N, L, D]
    #    x = self.transformer(
    #        x,
    #        cond=image_feats,
    #        mod=camera_embeddings,
    #    )
    #    return x

    def reshape_upsample(self, tokens):
        N = tokens.shape[0]
        H = W = self.triplane_low_res
        x = tokens.view(N, 3, H, W, -1)
        x = torch.einsum('nihwd->indhw', x)  # [3, N, D, H, W]
        x = x.contiguous().view(3*N, -1, H, W)  # [3*N, D, H, W]
        x = self.upsampler(x)  # [3*N, D', H', W']
        x = x.view(3, N, *x.shape[-3:])  # [3, N, D', H', W']
        x = torch.einsum('indhw->nidhw', x)  # [N, 3, D', H', W']
        x = x.contiguous()
        return x

    @torch.compile
    def forward_planes(self, image, camera):
        # image: [N, C_img, H_img, W_img]
        # camera: [N, D_cam_raw]
        N = image.shape[0]

        # encode image
        image_feats = self.encoder(image)
        assert image_feats.shape[-1] == self.encoder_feat_dim, \
            f"Feature dimension mismatch: {image_feats.shape[-1]} vs {self.encoder_feat_dim}"
            
        
        # embed camera
        camera_embeddings = self.camera_embedder(camera)
        assert camera_embeddings.shape[-1] == self.camera_embed_dim, \
            f"Feature dimension mismatch: {camera_embeddings.shape[-1]} vs {self.camera_embed_dim}"

        # transformer generating planes
        #tokens = self.forward_transformer(image_feats, camera_embeddings)
        #tokens = self.import_to_nerf(image_feats, camera_embeddings)
        #planes = self.reshape_upsample(tokens)
        #assert planes.shape[0] == N, "Batch size mismatch for planes"
        #assert planes.shape[1] == 3, "Planes should have 3 channels"
        
        # Assuming 'image_feats' and 'camera_feats' are your feature tensors
        combined_feats = torch.cat([image_feats, camera_embeddings], dim=1)  # Concatenate features along the feature dimension

        # Transformation to Gaussian Parameters
        means, variances = self.gaussian_transform(combined_feats)
        
        # Sample z from the Gaussian distribution defined by the means and variances
        std = torch.sqrt(variances)
        z = means + std * torch.randn_like(means)

        # RealNVP Transformation
        x = self.realnvp(z)
        
        # Generate dynamic NeRF parameters from the HyperNetwork
        nerf_params = generate_parameters(x,alpha=self.alpha, use_perturbation=self.enable_perturbations)
        
        # Use nerf_parameters to form the input for TriplaneGenerator
        tokens = self.triplane_generator(nerf_params)
        planes = self.reshape_upsample(tokens)
        #assert planes.shape[0] == N, "Batch size mismatch for planes"
        #assert planes.shape[1] == 3, "Planes should have 3 channels"
        
        return planes, means, variances

    def forward(self, image, source_camera, render_cameras, render_anchors, render_resolutions, render_bg_colors, render_region_size: int):
        # image: [N, C_img, H_img, W_img]
        # source_camera: [N, D_cam_raw]
        # render_cameras: [N, M, D_cam_render]
        # render_anchors: [N, M, 2]
        # render_resolutions: [N, M, 1]
        # render_bg_colors: [N, M, 1]
        # render_region_size: int
        assert image.shape[0] == source_camera.shape[0], "Batch size mismatch for image and source_camera"
        assert image.shape[0] == render_cameras.shape[0], "Batch size mismatch for image and render_cameras"
        assert image.shape[0] == render_anchors.shape[0], "Batch size mismatch for image and render_anchors"
        assert image.shape[0] == render_bg_colors.shape[0], "Batch size mismatch for image and render_bg_colors"
        N, M = render_cameras.shape[:2]

        planes, means, variances = self.forward_planes(image, source_camera)
        log_var = torch.log(variances.clamp(min=1e-9))
        
        # render target views
        render_results = self.synthesizer(planes, render_cameras, render_anchors, render_resolutions, render_bg_colors, render_region_size)
        assert render_results['images_rgb'].shape[0] == N, "Batch size mismatch for render_results"
        assert render_results['images_rgb'].shape[1] == M, "Number of rendered views should be consistent with render_cameras"

        return {
            'planes': planes,
            'mu': means,
            'log_var': log_var,
            **render_results,
        }
