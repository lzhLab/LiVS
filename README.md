# LiVS

This study present an elaborated large-volume and high-quality dataset containing liver raw images, liver masks and liver vessel masks; in addition, Laplacian salience-gated feature pyramid
network for accurate liver vessel segmentation.

# Dataset

Here we present LiVS, a fine-grained hepatic vascular dataset constructed by Dr. Zhao’s team. This has 532 volumes and 15,984 CT slices having vessel masks. The vessels of each slice are delineated by three senior medical imaging experts and the final mask is their majority voting. Due to the pretty small size of vessels, the delineation of each vessel can be oscillating a lot easily. To handle this problem, the coincidence of each vessel among the three masks is calculated. In case the majority voting over any mask is smaller than 0.5, the vessel is highlighted and sent back to the three experts for further refinement. This procedure repeats until no inconsistence exists.

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
   <br>Number of workers. Used to set the number of threads to load data.
* `ckpt`: str
  <br>Weight path. Used to set the dir path to save model weight. 
* `w`: str
  <br>The path of model wight to test or reload.
* `name`: int
  <br>Weights file name .
* `input_size`: int.
  <br>The size of input images.
* `channels`: int, default 3.
  <br>Number of image's channels.
* `ckpt`: int.
  <br>The dir path to save model weight.
* `dropout`: float between `[0, 1]`, default 0.
  <br>Dropout rate.
* `suf`: choices=['.dcm', '.JL', '.png'].
  <br>Image types.
* `eval`:action
  <br>Eval only need weight.
* `dataset_path`: str
  <br>Used to set the relative path of training and validation set.
* `batch_size`: int
  <br>Batch size.
* `max_epoch`: int 
  <br>The maximum number of epoch for the current training.
* `lr`: float
  <br>learning rate. Used to set the initial learning rate of the model.
```  
3. Example:  
	$python train_j2_fpn.py --dataset_path='dataset' batch_size='10' --max_epoch=100 --lr=1e-3
```
# Citation
If the model or LiVS is useful for your research, please consider citing:
```

```
# reference

