# Outline
Liver vessels generated from computed tomography are usually pretty small, which poses major challenges for satisfactory vessel segmentation, including 1) the scarcity of high-quality and large-volume vessel masks, 2) the difficulty in capturing vessel-specific features, and 3) the heavily imbalanced distribution of vessels and liver tissues. To advance, a sophisticated model and an elaborated dataset have been built. The model has a newly conceived Laplacian salience filter that highlights vessel-like regions and suppresses other liver regions to shape the vessel-specific feature learning and to balance vessels against others. It is further coupled with a pyramid deep learning architecture to capture different levels of features, thus improving the feature formulation. Experiments show that this model markedly outperforms the state-of-the-art approaches, achieving a relative improvement of Dice score by at least 1.63% compared to the existing best model on available datasets. More promisingly, the averaged Dice score produced by the existing models on the newly constructed dataset is as high as 0.734 ± 0.070, which is at least 18.3% higher than that obtained from the existing best dataset under the same settings. These observations suggest that the proposed Laplacian salience, together with the elaborated dataset, can be helpful for liver vessel segmentation.

# Dataset

Here we present LiVS, a fine-grained hepatic vascular dataset constructed by Dr. Zhao’s lab. It has 532 volumes and 15,984 CT slices with vessel masks. The vessels of each slice are delineated by three senior medical imaging experts and the final mask is their majority voting. Due to the pretty small size of vessels, the delineation of each vessel can be oscillating a lot easily. To handle this problem, the coincidence of each vessel among the three masks is calculated. In case the majority voting over any mask is smaller than 0.5, the vessel is highlighted and sent back to the three experts for further refinement. This procedure repeats until no inconsistence exists.

The description of each volume is available [here](https://201610006.github.io/LiVS_site/), and the whole data has been uploaded to IEEE DataPort with DOI: [10.21227/rwys-mk84](https://ieee-dataport.org/documents/liver-vessel).

# Usage
how to start it?
```
1. Clone the repository:
     $git clone https://github.com/lzhLab/LiVS.git
     
2. Run the main program:     
     $python train_j2_fpn.py <--parameters>
```   

### Parameters

* `num_workers`: int
   <br>Number of workers, used to set the number of threads to load data.
* `ckpt`: str
  <br>Weight path, used to set the path to save the model. 
* `w`: str
  <br>The path of model to be tested or reloaded.
* `name`: int
  <br>Weights file name .
* `input_size`: int.
  <br>The size of input images.
* `channels`: int, default 3.
  <br>Number of image's channels.
* `dropout`: float between `[0, 1]`, default 0.
  <br>Dropout rate.
* `suf`: choices=['.dcm', '.JL', '.png'].
  <br>Image types.
* `eval`:action
  <br>Evaluation.
* `dataset_path`: str
  <br>The relative path of training and validation data.
* `batch_size`: int
  <br>Batch size.
* `max_epoch`: int 
  <br>The maximum number of epoch.
* `lr`: float
  <br>learning rate. 
```  
3. Example:  
	$python train_j2_fpn.py --dataset_path='dataset' batch_size='10' --max_epoch=100 --lr=1e-3
```
# Citation
If the model or LiVS is useful for your research, please cite:
```
Gao Z., Zong Q., Wang Y., Yan Y., Wang Y., Zhu N., Zhang J., Wang Y., and Zhao L. Laplacian salience-gated feature pyramid network for accurate liver vessel segmentation, IEEE Transactions on Medical Imaging, 2023, DOI:10.1109/TMI.2023.3273528.
```

