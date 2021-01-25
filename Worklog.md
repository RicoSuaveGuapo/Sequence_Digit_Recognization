# Digits Recognization
# Idea
* use YOLO to detect the sole number (?)

## Note
* MultiplicativeNoise, RandomBrightness, GaussNoise hurt acc...
* rewrite the decoding help at lotttttttt!
* Inference and drawing on validation set is in `test_lprnet.py`
* video output of the LPRnet result is in `utils.py`

## Finished
* Change image resolution (V)
* Change to cosine scheduler (V)
* Use different decoder (V)
* Move to the upper region (V)

## TODO
### Recognition Part
* Use cutmix augmentation (postpone)
* shuffle the number sequence
* test-time augmentation
* use the pattern remove image
* Replace the language model

### Detection Part
Limit the shape

* Upper Region
  1. Merge to one (Max height, Max width)
  2. Enlarge to certain size
* Lower Region
  1. Highest Score
  2. lower the upper region with certain step


## Dataset
Split the data into two sets (V)
   1. One with 9 digits
   2. One with 3 digits + whatever

## YOLO
* modify the xml files (V)
  * check the code in EDA.ipynb (in "new annotation for yolo")
* Put in the YOLO (V)
### Output Modification
* limit the min area
* get the largest of height and width (if two boxes present)
* increase the Upper class to 4/5 of the width

## MTCNN
TODO
* Add the src file to increase the features (-)
* combine two the regions (V)
* write the metric (V)
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


## LPRNet Workflow
1. Take the data from ground truth
   1. LPRDataLoader in `dataset.py`
2. Train on `train_lprnet.py`


### LPRnet
1. Two-model
   1. One with 9 digits
   2. One with 3 digits + whatever
2. One-model
   1. Combine them together