import sys, os
import cv2
import matplotlib
import keras_contrib
import azure_chestxray_cam, azure_chestxray_utils, azure_chestxray_keras_utils
import keras_contrib
from keras.models import Model


path = os.getcwd()+r'\azure-share'
amlWBSharedDir = path

prj_consts = azure_chestxray_utils.chestxray_consts()

data_base_output_dir=os.path.join(amlWBSharedDir,
                                  os.path.join(*(prj_consts.BASE_OUTPUT_DIR_list)))
data_base_input_dir=os.path.join(amlWBSharedDir,
                                 os.path.join(*(prj_consts.BASE_INPUT_DIR_list)))

# "quality" models, fully trained on all training data
fully_trained_weights_dir=os.path.join(data_base_output_dir,
    os.path.join(*(prj_consts.FULLY_PRETRAINED_MODEL_DIR_list)))

test_images_dir=os.path.join(data_base_input_dir,
    os.path.join(*(['test_images'])))

test_images=azure_chestxray_utils.get_files_in_dir(test_images_dir)

nih_chest_xray_data_dir=os.path.join(data_base_input_dir,
                                     os.path.join(*(prj_consts.ChestXray_IMAGES_DIR_list)))

chestXray_images=azure_chestxray_utils.get_files_in_dir(nih_chest_xray_data_dir)


model = azure_chestxray_keras_utils.build_model(keras_contrib.applications.densenet.DenseNetImageNet121)

model_file_name = prj_consts.PRETRAINED_DENSENET201_IMAGENET_CHESTXRAY_MODEL_FILE_NAME
model_file_name = 'weights_only_azure_chest_xray_14_weights_712split_epoch_054_val_loss_191.2588.hdf5'
model.load_weights(os.path.join(fully_trained_weights_dir, model_file_name))


cv2_image = cv2.imread(os.path.join(test_images_dir,test_images[3]))



predictions = model.predict(cv2_image[None,:,:,:])
print(predictions)
conv_map_model = Model(inputs=model.input, outputs=model.get_layer(index=-3).output)
conv_features = conv_map_model.predict(cv2_image[None,:,:,:])
conv_features = conv_features[0, :, :, :] #np.squeeze(conv_features)
class_weights = model.layers[-1].get_weights()

cv2_image = cv2.imread(os.path.join(test_images_dir,test_images[3]))

azure_chestxray_utils.print_image_stats_by_channel(cv2_image)
cv2_image = azure_chestxray_utils.normalize_nd_array(cv2_image)
cv2_image = 255*cv2_image
cv2_image=cv2_image.astype('uint8')
azure_chestxray_utils.print_image_stats_by_channel(cv2_image)

predictions, cam_image, predicted_disease_index = \
azure_chestxray_cam.get_score_and_cam_picture(cv2_image, model)
print(predictions)

prj_consts.DISEASE_list[predicted_disease_index]
print('likely disease: ', prj_consts.DISEASE_list[predicted_disease_index])
print('likely disease prob ratio: ', \
          predictions[predicted_disease_index]/sum(predictions))



NIH_annotated_nodules = ['00000706_000.png', '00000702_000.png']
azure_chestxray_cam.process_nih_data(NIH_annotated_nodules,
                                   nih_chest_xray_data_dir, model)