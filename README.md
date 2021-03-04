# Sequence_Digit_Recognization

## Status
### Overall Process
1. - [x] Auto-rotate 
2. - [x] Detection cropping
3. - [x] Recognition
4. - [ ] Connect all the section
5. - [ ] Auxiliary methods (see Recognition Part in TODO

* 2021/02/25:
  * Rotation Head : 98~100 %
  * YOLO map: 96.8 -> 100
  * Upper part: 92.77 -> 96.39 %
  * Lower part: 92.77 -> 98.80 %
  * Combination: 84.34 -> 90.36~92 %

## Information
* `data.csv` is the correct version, some xmls are wrong
* MultiplicativeNoise, RandomBrightness, GaussNoise hurt accuracy
* rewrite the decoding of lprnet helps a lotttttttt!
* Inference and drawing on validation set is in `test_lprnet.py`
* video output of the LPRnet result is in `utils.py`

## TODO
### Recognition Part
* - [ ] Use cutmix augmentation (postpone)
* - [ ] Shuffle the number sequence
* - [ ] Test-time augmentation
* - [ ] Use the pattern remove image

## Inference Workflow
### LPRNet Workflow
1. Take the data from ground truth
   * LPRDataLoader in `dataset.py`
2. Train on `train_lprnet.py`
3. Inference and image output is in `test_lprnet.py`


## Finished
* - [x] Change image resolution 
* - [x] Enlarge a bit of annotation in x direction
* - [x] Change to cosine scheduler
* - [x] Use different decoder
* - [x] Move to the upper region
* - [x] Split the data into two sets
   1. One with 9 digits
   2. One with 3 digits + whatever
* - [x] Rotation
  * the rotation model is in 'model.py'
  * the weight is in 'weights/rotation/acc_100.00_loss_0.747.pth'
  * Result: 
    * val : 98.21% (56 images)
    * test: 100  % (55 images)
* YOLO
  * - [x] modify the xml files
    * check the code in EDA.ipynb (in "new annotation for yolo")
  * - [x] Put in the YOLO

## MTCNN (Decrypted)
* - [x] combine two the regions
* Add the src file to increase the features (-)
* - [x] write the metric
* Refine the trainings (-)
* Use the DFT (V)
  * Apply to the data, check the gain (-)
* kmean the bbox (-)

### MTCNN Workflow
0. Might need to rotate the images first.
In `preprocessing.py`
1. `data_check`
2. annotation data add in, also col note added
3. `dataframe_creation`
4. Below will be save in `yourDefineName_EXT_clear_2data_mode_resize.csv`
   ID                file_name    GT_1   GT_2 xmin ymin  xmax  ymax  ymax_2  mode
0   0  20201229080315_0EXT.bmp  389708  102MV  237  756  1267  1336    1046     0
1   1  20201229080446_0EXT.bmp  389708   207V  257  553  1318  1143     848     1
...
5. `pnet_traindata` for pnet data
6. `assemble_split` assemble image list
7. train Pnet in `train_pnet.py`
8. `onet_traindata`
9. `assemble_split` again
10. `train_onet.py`
11. `MTCNN_evaluation.py` can give qualitative result
12. `mtcnn_metric.py` gives the quantitative resulrt

### Rewrite the genrative data
* 'MTCNN/data_preprocessing/gen_Pnet_train_data.py' (V)
* 'MTCNN/data_preprocessing/gen_Onet_train_data.py' (V)
* 'MTCNN/data_preprocessing/assemble_Pnet_imglist.py' (V)
* 'MTCNN/data_preprocessing/assemble_Onet_imglist.py' (V)

### Training
* 'MTCNN/train/Train_Pnet.py' (V)
* 'MTCNN/train/Train_Onet.py (V)

### Evalution
* mtcnn_visual.py (V)
* mtcnn_metric.py (V)