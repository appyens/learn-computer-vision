
# coding: utf-8

# In[1]:

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

import os
# In[16]:


def prediction():
    
    """
    Function to predict if the retina image has diabetic retinopathy or not.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    y_pred: bool
            Whether or not the retina has diabetic retinopathy.
    percent_chance: float
            Percentage of chance the retina image has diabetic retinopathy.
    """

    PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))
    print(PROJECT_PATH)
    mod=load_model(PROJECT_PATH + '/model.hd5')
    
    test_gen = ImageDataGenerator(rescale = 1./255)


    PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))
    CAPTHA_ROOT = os.path.join(PROJECT_PATH,'test_images')
    
    test_data = test_gen.flow_from_directory(CAPTHA_ROOT,
                                              target_size = (64, 64),
                                              batch_size = 32,
                                              class_mode = 'binary', shuffle=False)
    
    predicted = mod.predict_generator(test_data)
    
    y_pred = predicted[0][0] > 0.4
    percent_chance = round(predicted[0][0]*100, 2)

    import matplotlib.pyplot as plt
    # % matplotlib inline
    img_path = CAPTHA_ROOT + '/dataset/test/'
    for i in os.listdir(img_path):
        j = img_path + i
        k = plt.imread(j)
        plt.imshow(k)
        plt.show()

    return y_pred, percent_chance

# In[17]:

if __name__ == '__main__':
    result, percentage = prediction()
    if result:
        print("Found symptoms of diabetic retinopathy")
        print("Diabetic retinopathy chances {}%".format(percentage))
    else:
        print("Found no symptoms of diabetic retinopathy")




