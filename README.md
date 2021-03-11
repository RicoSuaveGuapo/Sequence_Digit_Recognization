# Sequence_Digit_Recognization

![Flowchart](https://i.imgur.com/yolklp3.png)

## Status
### Overall Process
1. - [x] Auto-rotate 
2. - [x] Detection cropping
3. - [x] Recognition
4. - [ ] Connect all the section
5. - [ ] Auxiliary methods (see Recognition Part in TODO)

* 2021/02/25 Results:
  * Rotation Head : 98~100 %
  * YOLO map: 96.8 -> 100
  * Upper part: 92.77 -> 96.39 %
  * Lower part: 92.77 -> 98.80 %
  * Combination: 84.34 -> 90.36~92 %

## Information
### Note
* `data.csv`: contains overall annotations
  * col: |ID|file_name|GT_1|GT_2|xmin|ymin|xmax|ymax|ymax_2|mode|
  * GT_1: ground truth of upper region
  * GT_2: ground truth of lower region
  * mode: 0/1/2: training/validation/testing
  ```
  (xmin,ymin)  --------------------
                |   upper region   |
                -------------------- (xmax,ymax_2)
                |   lower region   |
                --------------------
                                    (xmax,ymax)
  ```
* MultiplicativeNoise, RandomBrightness, GaussNoise data augmentation hurt LPRnet accuracy
* Rewrite the decoding of LPRnet helps a lotttttttt!

### How to Train
* rotation model: `train_rotation.py`
  * data in 'data/20201229/EXT/resize/new_image'
* YOLO          : `yolov4/train.py`
  * data in 'yolov4/VOCdevkit/VOC2007'
* lprnet        : `train_lprnet.py`
  * data in 'data/20201229/EXT/resize/new_image'
  * will use the `data.csv` to crop the images


### How to Inference

#### Rotation Model
* Single image: check rotation part in `output_pipeline.py`
* Evaluate the performance: check `test_rotation.py`

#### YOLO
* Qualitative Result (bbox on images):Check `yolov4/predict.py`
* Quantitative Result (Map, F1 and diagrams... etc.):
  1. Modifing (change to your training result) the model path in `yolov4/yolo.py`
  2. Run `yolov4/get_dr_txt.py`
  3. Run `yolov4/get_gt_txt.py`
  4. Run `yolov4/get_map.py`
  5. Results will be saved in `yolov4/results`

#### LPRnet
* Inference and image output is in `test_lprnet.py`
* Drawing diagrams on validation set is in `test_lprnet.py`
* Video output of the LPRnet result is in `utils.py`



## TODO
### Improvements
* - [ ] Use cutmix augmentation (postpone)
* - [ ] Shuffle the number sequence
* - [ ] Test-time augmentation
* - [ ] Use the pattern remove image

### Other methods
* Using [SHVM dataset](http://ufldl.stanford.edu/housenumbers/?fbclid=IwAR3C2sFr6IIH4LxXr_EVbuGVWky7JCCA46veUt-no8o2CcwkUdwbBIs7Zo8) train yolo.


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

================================== Decrypted ==================================
## MTCNN 
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