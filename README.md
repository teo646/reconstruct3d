# reconstruct3d
3D Reconstruction from Multiple Images

---

## 개요

2D 디지털 이미지는 3D 장면을 카메라를 통해 담아낸 **이산 신호**이다.  
따라서 2D 이미지에는 장면에 대한 **의미론적(Semantic) 정보**뿐만 아니라 **3D 기하·광도 정보**가 함축되어 있다.

다만, **한 장의 2D 이미지는 한 시점의 3D 장면 정보만**을 담고 있어 완전한 3D 복원에는 한계가 있다.  
이처럼 **불완전한 장면 정보**를 **여러 시점의 이미지**를 이용해 보완하는 과정을 **Inverse Rendering**이라 한다.

---

## 목표

본 과제에서는 **하나의 장면**에 대해 **서로 다른 시점에서 촬영된 다중 2D 이미지**를 이용하여 **온전한 3D 복원**을 목표로 한다.

이를 위해 다음을 다룬다.

- **카메라 모델** 및 **이미지 프로세싱 파이프라인** 학습  
- 2D 이미지 **픽셀**과 **3D 공간 정보**의 **매핑 관계** 파악  
- 다양한 **Inverse Rendering** 방법 구현

---

## 참고 문헌

| Category | Model | Paper Title |
|----------|-------|-------------|
| **Classification** | | |
| | AlexNet | ImageNet Classification with Deep Convolutional Neural Networks |
| | VGG | Very Deep Convolutional Networks for Large-Scale Image Recognition |
| | GoogleNet | Going Deeper with Convolutions |
| | ResNet | Deep Residual Learning for Image Recognition |
| | SENet | Squeeze-and-Excitation Networks |
| | ViT | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale |
| | SwinTransformer | Swin Transformer: Hierarchical Vision Transformer using Shifted Windows |
| **Detection** | | |
| | R-CNN | R-CNN: Region-based Convolutional Neural Networks |
| | Fast R-CNN | Fast R-CNN |
| | Faster R-CNN | Faster R-CNN |
| | SSD | SSD: Single Shot MultiBox Detector |
| | YOLO | YOLO: Real-Time Object Detection |
| | DETR | End-to-End Object Detection with Transformers |
| | Mask R-CNN | Mask R-CNN for Object Detection and Segmentation |
| | Focal Loss | Focal Loss for Dense Object Detection |
| **Segmentation** | | |
| | FCN | FCN: Fully Convolutional Networks for Semantic Segmentation |
| | DeepLab | DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs |
| | U-Net | U-Net: Convolutional Networks for Biomedical Image Segmentation |
| **Generative** | | |
| | VAE | Auto-Encoding Variational Bayes |
| | GAN | Generative Adversarial Nets |
| | DCGAN | Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks |
| | Pix2pix | Image-to-Image Translation with Conditional Adversarial Networks |
| | CycleGAN | Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks |
| | DDPM | Denoising Diffusion Probabilistic Model |
| | DDIM | Denoising Diffusion Implicit Models |
| | Latent Diffusion | High-Resolution Image Synthesis with Latent Diffusion Models |
| **Super-resolution** | | |
| | SRCNN | Image Super-Resolution Using Deep Convolutional Networks |
| | VDSR | Accurate Image Super-Resolution Using Very Deep Convolutional Networks |
| | SwinIR | SwinIR: Image Restoration Using Swin Transformer |
| **ETC** | | |
| | Stitching | Automatic Panoramic Image Stitching using Invariant Features |
| | Attention | Attention Is All You Need |
| **NeRF** | | |
| | NeRF | Representing Scenes as Neural Radiance Fields for View Synthesis |
| | Instant-NGP | Instant Neural Graphics Primitives (Instant-NGP) |
| | PlenOctrees | PlenOctrees for Real-time Neural Rendering |
| | PixelNeRF | PixelNeRF |
| | D-NeRF | D-NeRF (Dynamic NeRF) |
| **3DGS** | | |
| | 3D Gaussian Splatting | 3D Gaussian Splatting for Real-Time Radiance Field Rendering |
| | Mip-Splatting | Alias-free 3D Gaussian Splatting |
| | 2D Gaussian Splatting | 2D Gaussian Splatting for Geometrically Accurate Radiance Fields |
| | 4D Gaussian Splatting | 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering |
| | 1000+ FPS 4DGS | 1000+ FPS 4D Gaussian Splatting for Dynamic Scene Rendering |

