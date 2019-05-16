# Predict-Lung-Disease-through-Chest-X-Ray
We obtain this repository by refactoring the [code](https://github.com/Azure/AzureChestXRay) for the blog post [Using Microsoft AI to Build a Lung-Disease Prediction Model using Chest X-Ray Images](https://blogs.technet.microsoft.com/machinelearning/2018/03/07/using-microsoft-ai-to-build-a-lung-disease-prediction-model-using-chest-x-ray-images/). This instruction aims to help newcomers build the system in a very short time.   
# Installation
1. Clone this repository
   ```Shell
   git clone https://github.com/svishwa/crowdcount-mcnn.git
   ```
   We'll call the directory that you cloned PredictLungDisease `ROOT`  
  
2. All essential dependencies should be installed  
# Data set up
1. Download the NIH Chest X-ray Dataset from here:  
   https://nihcc.app.box.com/v/ChestXray-NIHCC.  
   You need to get all the image files (all the files under `images` folder in NIH Dataset), `Data_Entry_2017.csv` file, as well as the      Bounding Box data `BBox_List_2017.csv`.  

2. Create Directory 
   ```Shell
   mkdir ROOT/azure-share/chestxray/data/ChestX-ray8/ChestXray-NIHCC
   mkdir ROOT/azure-share/chestxray/data/ChestX-ray8/ChestXray-NIHCC_other
   ```  
3. Save all images under `ROOT/azure-share/chestxray/data/ChestX-ray8/ChestXray-NIHCC`  

4. Save `Data_Entry_2017.csv` and `BBox_List_2017.csv` under `ROOT/azure-share/chestxray/data/ChestX-ray8/ChestXray-NIHCC_other`  

5. Process the Data
   ```Shell
   mkdir ROOT/azure-share/chestxray/output/data_partitions
   ```  
   Run `000_preprocess.py` to create `*.pickle` files under this directory 
# Test  
1. We have provided the pretrained-model `azure_chest_xray_14_weights_712split_epoch_054_val_loss_191.2588.hdf5` under `ROOT/azure-share/chestxray/output/fully_trained_models`. You can also download it separately from [here](https://chestxray.blob.core.windows.net/chestxraytutorial/tutorial_xray/chexray_14_weights_712split_epoch_054_val_loss_191.2588.hdf5).  

2. Run `020_evaluate.py` and it will create `weights_only_azure_chest_xray_14_weights_712split_epoch_054_val_loss_191.2588.hdf5` saving weights of the pretrained-model under the same directory.

3. Below is the result showing the AUC score of all the 14 diseases:  

   | Disease            | Our AUC Score    | Stanford AUC Score | Delta     
   |--------------------|------------------|--------------------|-----------:
   | Atelectasis        | 0.822334         | 0.8094             | -0.012934 
   | Cardiomegaly       | 0.933610         | 0.9248             | -0.008810 
   | Effusion           | 0.882471         | 0.8638             | -0.018671 
   | Infiltration       | 0.744504         | 0.7345             | -0.010004 
   | Mass               | 0.858467         | 0.8676             |  0.009133 
   | Nodule             | 0.784230         | 0.7802             | -0.004030 
   | Pneumonia          | 0.800054         | 0.7680             | -0.032054 
   | Pneumothorax       | 0.829764         | 0.8887             |  0.058936 
   | Consolidation      | 0.811969         | 0.7901             | -0.021869 
   | Edema              | 0.894102         | 0.8878             | -0.006302 
   | Emphysema          | 0.847477         | 0.9371             |  0.089623
   | Fibrosis           | 0.882602         | 0.8047             | -0.077902 
   | Pleural Thickening | 1.000000         | 0.8062             | -0.193800 
   | Hernia             | 0.916610         | 0.9164             | -0.000210   
   
# Visualization    
1. Create Folder Test
   ```Shell
   mkdir ROOT/azure-share/chestxray/data/ChestX-ray8/test_images
   ```  
   Copy any number of images under `ChestXray-NIHCC` to `test_images` and resize them to 224x224 pixels.  

2. Run `004_cam_simple.py` and it will output a Class Activation Map(CAM). The CAM lets us see which regions in the image were relevant to this class.  

# Referenced Paper


