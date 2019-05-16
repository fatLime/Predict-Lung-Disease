import sys, os
import azure_chestxray_utils
import pickle
import random
import re
import tqdm
import cv2
import numpy as np
import pandas as pd
import sklearn.model_selection
from collections import Counter

paths_to_append = [os.path.join(os.getcwd(), os.path.join(*(['Code',  'src'])))]
def add_path_to_sys_path(path_to_append):
    if not (any(path_to_append in paths for paths in sys.path)):
        sys.path.append(path_to_append)

[add_path_to_sys_path(crt_path) for crt_path in paths_to_append]

path= os.getcwd()+r'\azure-share'
isExists=os.path.exists(path)
if not isExists:
    amlWBSharedDir = os.mkdir(path)
else:
    amlWBSharedDir = path




prj_consts = azure_chestxray_utils.chestxray_consts()
print(prj_consts)

data_base_input_dir=os.path.join(amlWBSharedDir, os.path.join(*(prj_consts.BASE_INPUT_DIR_list)))
data_base_output_dir=os.path.join(amlWBSharedDir, os.path.join(*(prj_consts.BASE_OUTPUT_DIR_list)))

isExists1 = os.path.exists(data_base_input_dir)
isExists2 = os.path.exists(data_base_output_dir)

if not isExists1:
    data_base_input_dir = os.mkdir(data_base_input_dir)
print(data_base_input_dir)

if not isExists2:
    data_base_output_dir = os.mkdir(data_base_output_dir)
print(data_base_output_dir)

nih_chest_xray_data_dir=os.path.join(data_base_input_dir,
                                     os.path.join(*(prj_consts.ChestXray_IMAGES_DIR_list)))
isExists3 = os.path.exists(nih_chest_xray_data_dir)
if not isExists3:
    nih_chest_xray_data_dir = os.mkdir(nih_chest_xray_data_dir)

print(nih_chest_xray_data_dir)

other_data_dir=os.path.join(data_base_input_dir, os.path.join(*(prj_consts.ChestXray_OTHER_DATA_DIR_list)))
data_partitions_dir=os.path.join(data_base_output_dir, os.path.join(*(prj_consts.DATA_PARTITIONS_DIR_list)))

ignored_images_set = set()

total_patient_number = 30805
NIH_annotated_file = 'BBox_List_2017.csv' # exclude from train pathology annotated by radiologists
manually_selected_bad_images_file = 'blacklist.csv'# exclude what viusally looks like bad images

patient_id_original = [i for i in range(1,total_patient_number + 1)]

bbox_df = pd.read_csv(os.path.join(other_data_dir, NIH_annotated_file))
bbox_patient_index_df = bbox_df['Image Index'].str.slice(3, 8)

bbox_patient_index_list = []
for index, item in bbox_patient_index_df.iteritems():
    bbox_patient_index_list.append(int(item))

patient_id = list(set(patient_id_original) - set(bbox_patient_index_list))
print("len of original patient id is", len(patient_id_original))
print("len of cleaned patient id is", len(patient_id))
print("len of unique patient id with annotated data",
      len(list(set(bbox_patient_index_list))))
print("len of patient id with annotated data",bbox_df.shape[0])

random.seed(0)
random.shuffle(patient_id)

print("first ten patient ids are", patient_id[:10])

# training:valid:test=7:1:2
patient_id_train = patient_id[:int(total_patient_number * 0.7)]
patient_id_valid = patient_id[int(total_patient_number * 0.7):int(total_patient_number * 0.8)]
# get the rest of the patient_id as the test set
patient_id_test = patient_id[int(total_patient_number * 0.8):]
patient_id_test.extend(bbox_patient_index_list)
patient_id_test = list(set(patient_id_test))

print("train:{} valid:{} test:{}".format(len(patient_id_train), len(patient_id_valid), len(patient_id_test)))

pathologies_name_list = prj_consts.DISEASE_list
NIH_patients_and_labels_file = 'Data_Entry_2017.csv'

labels_df = pd.read_csv(os.path.join(other_data_dir, NIH_patients_and_labels_file))


#show the label distribution

# Unique IDs frequencies can be computed using list comprehension or collections lib
# [[x,(list(crtData['fullID2'])).count(x)] for x in set(crtData['fullID2'])]
# for tallying, collections lib is faster than list comprehension
pathology_distribution = Counter(list(labels_df['Finding Labels']))

# Sort it by ID frequency (dict value)
sorted_by_freq = sorted(pathology_distribution.items(), key=lambda x: x[1], reverse=True)
print(len(sorted_by_freq))
print(sorted_by_freq[:20])
print(sorted_by_freq[-10:])

print(labels_df['Finding Labels'].str.split( '|', expand=False).str.join(sep='*').str.get_dummies(sep='*').sum())

def process_data(current_df, patient_ids):
    image_name_index = []
    image_labels = {}
    for individual_patient in tqdm.tqdm(patient_ids):
        for _, row in current_df[current_df['Patient ID'] == individual_patient].iterrows():
            processed_image_name = row['Image Index']
            if processed_image_name in ignored_images_set:
                pass
            else:
                image_name_index.append(processed_image_name)
                image_labels[processed_image_name] = np.zeros(14, dtype=np.uint8)
                for disease_index, ele in enumerate(pathologies_name_list):
                    if re.search(ele, row['Finding Labels'], re.IGNORECASE):
                        image_labels[processed_image_name][disease_index] = 1
                    else:
                        # redundant code but just to make it more readable
                        image_labels[processed_image_name][disease_index] = 0
                # print("processed", row['Image Index'])
    return image_name_index, image_labels


train_data_index, train_labels = process_data(labels_df, patient_id_train)
valid_data_index, valid_labels = process_data(labels_df, patient_id_valid)
test_data_index, test_labels = process_data(labels_df, patient_id_test)

print("train, valid, test image number is:", len(train_data_index), len(valid_data_index), len(test_data_index))

# save the data
labels_all = {}
labels_all.update(train_labels)
labels_all.update(valid_labels)
labels_all.update(test_labels)

partition_dict = {'train': train_data_index, 'test': test_data_index, 'valid': valid_data_index}

with open(os.path.join(data_partitions_dir, 'labels14_unormalized_cleaned.pickle'), 'wb') as f:
    pickle.dump(labels_all, f)

with open(os.path.join(data_partitions_dir, 'partition14_unormalized_cleaned.pickle'), 'wb') as f:
    pickle.dump(partition_dict, f)

# also save the patient id partitions for pytorch training
with open(os.path.join(data_partitions_dir, 'train_test_valid_data_partitions.pickle'), 'wb') as f:
    pickle.dump([patient_id_train, patient_id_valid,
                 patient_id_test,
                 list(set(bbox_patient_index_list))], f)

print(type(train_labels))
print({k: train_labels[k] for k in list(train_labels)[:5]})