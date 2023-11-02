import os
import zipfile
import argparse
from utils import save_fig
from utils import count_plot

# ploting images from train dir
def plot_images(path):
  images = []
  for folder in os.listdir(path):
    for image in os.listdir(path + '/' + folder):
      images.append((os.path.join(path, folder, image), folder))

  plt.figure(1, figsize=(8, 8))

  for i in range(16):
    i += 1

    random_img = random.choice(images)
    imgs = plt.imread(random_img[0])
    plt.subplot(4, 4,i)
    plt.axis('off')
    plt.title(random_img[1])
    plt.imshow(imgs)
  save_fig('train_imgs')
  plt.show()


def download_dataset(username, apikey, download_path):
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = apikey

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    dataset = 'msambare/fer2013'

    api.dataset_download_files(dataset, download_path)

    zip_file_path = os.path.join(download_path, 'fer2013.zip') 

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall('dataset') 

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download and extract Kaggle dataset")
    parser.add_argument("--username", required=True, help="Kaggle username")
    parser.add_argument("--apikey", required=True, help="Kaggle API key")
    parser.add_argument("--download_path", required=True, help="Path to download and extract the dataset")
    args = parser.parse_args()
    download_dataset(args.username, args.apikey, args.download_path)
    plot_images(os.path.join(args.download_path, 'train'))
    count_plot((os.path.join(args.dataset_path, 'train')),'train_count')
    count_plot((os.path.join(args.dataset_path, 'test')),'test_count')
