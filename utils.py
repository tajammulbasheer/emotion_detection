import os
import random
import matplotlib.pyplot as plt
from keras.models import load_model
PATH = os.getcwd()

# function to save figures
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    os.chdir(PATH)
    IMAGES_PATH = PATH +'/images'

    if os.path.isdir(IMAGES_PATH) is False:
        os.mkdir('images')
        IMAGES_PATH = PATH + '/images'
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)

    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    return


# image distribution

def count_plot(path, name):
  class_counts = {}

  for subdir in os.listdir(path):
      subdir_path = os.path.join(path, subdir)
      class_counts[subdir] = len(os.listdir(subdir_path))

  fig = plt.figure(figsize=(5, 5))

  sns.barplot(x=list(class_counts.keys()), y =list(class_counts.values()),data=None)
  plt.xlabel('Classes')
  plt.ylabel('Number of Images')
  save_fig(name)
  plt.show()

  
def load_modell(model_path):
    os.chdir(model_path)
    model = load_model('basic.h5')
    model.summary()
    print('Model loaded')
    return model