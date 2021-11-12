from tensorflow.keras.models import load_model
import sys
import numpy as np
import glob
import os
import h5py
from Generator import DriveDataGenerator
MODEL_PATH = None

if (MODEL_PATH == None):
    models = glob.glob('../model/models/*.h5') 
    best_model = max(models, key=os.path.getctime)
    MODEL_PATH = best_model
    
print('Using model {0} for testing.'.format(MODEL_PATH))

model = load_model(MODEL_PATH)

COOKED_DATA_DIR = '../data_cooked/'
test_dataset = h5py.File(os.path.join(COOKED_DATA_DIR, 'test.h5'), 'r')

data_generator = DriveDataGenerator(
    rescale=1./255., horizontal_flip=True, brightness_range=[1-0.4,1+0.4]
)
test_generator = data_generator.flow\
    (test_dataset['image'], test_dataset['previous_state'], test_dataset['label'], zero_drop_percentage=0.95, roi=[76,135,0,255])    

print("Evaluate on test data")
results = model.evaluate(test_generator, batch_size=128, return_dict=True)
print(results)