### ViS-GAN
_Generating Higher Quality Anti-Forensics DeepFakes with Adversarial Sharpening Mask_
#### Authors
#### Abstract
DeepFake, an artificial intelligent technology which can automatically synthesize facial forgeries has recently attracted worldwide attention. Albeit DeepFakes can be applied to entertain people, it can also be employed to spread falsified information or even be weaponized amid information war. So far, forensics researchers have been dedicated to designing new algorithms to resist such disinformation. On the other hand, there are techniques developed to make DeepFake products more aggressive. For instance, by launching anti-forensics attacks, DeepFakes can be disguised as pristine media that can disrupt forensics detectors. However, it usually sacrifices image qualities for achieving satisfactory undetectability in anti-forensics attacks. To address this issue, we propose a method to generate the novel adversarial sharpening mask. Unlike many existing arts, with such masks worn, DeepFakes could achieve high anti-forensics performance while exhibiting pleasant sharpening visual effects. After experimental evaluations, we prove that the proposed method could successfully disrupt the state-of-the-art DeepFake detectors. Besides, compared with the images processed by existing DeepFake anti-forensics methods, the qualities of anti-forensics images produced by the proposed method are significantly improved.



<p align="center">
  <img src="https://github.com/BingFanSpace/ViS-GAN/blob/main/readme_images/framwork.jpg" width="480">
</p>

<p align="center">
  <img src="https://github.com/BingFanSpace/ViS-GAN/blob/main/readme_images/G2.png" width="480">
</p>


### datasets
the tree of `./DFData`:
```
 .
 ├── Deeperforensics
 │   ...
 ├── FFPP
 │   ...
 └── Celeb_DF_v2
     ├── test
     │   ├── fake
     │   ├── fakeUSM_311
     │   └── real
     ├── train
     │   ├── fake
     │   ├── fakeUSM_311
     │   └── real
     └── valid
         ├── fake
         ├── fakeUSM_311
         └── real
```

## Usage
#### Dependencies
- Python 3.7.11
- Pytorch 1.10.0+cu113
- torchvision 0.11.1+cu113
- CUDA 11.4

#### Train Forensics Disruption Network(FDN)
Run
```shell
python ./FDN/main.py 
--dataset_name "Celeb_DF_v2" 
```
to save the `G1` parameters in `./FDN/saved_models/[dataset_name]/generator_[xx].pth`

#### Train Visual Enhancement Network(VEN)
Run
```shell
python ./VEN/main.py 
--dataset_name "Celeb_DF_v2" 
--generator_path "../FDN/saved_models/[dataset_name]/generator_[xx].pth"
```
to gain the final trained model in `./VEN/saved_models/[dataset_name]/visual_optimizer_[xx].pth`

### Evaluate FDN
Run
```shell
python ./FDN/generate_images.py 
--dataset_name "Celeb_DF_v2" 
```


### Evaluate VEN
Run
```shell
python ./VEN/generate_images.py 
--dataset_name "Celeb_DF_v2" 
```

first row: original DeepFake frames; third row: FDN result with less visual quality; fifth row: VEN result:
<p align="center">
  <img src="https://github.com/BingFanSpace/ViS-GAN/blob/main/readme_images/compare_FDN_VEN.png" width="480">
</p>
