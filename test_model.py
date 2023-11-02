from keras.models import load_model
from utils import save_fig, load_modell
from keras.preprocessing.image import ImageDataGenerator
import argparse
import os

BATCH_SIZE = 64
TARGET_SIZE = (48,48)
NUM_CLASSES = 7
IMG_SIZE = 48

# ploting the outputs using confusion matrix

def plot_confusion(y_test,y_predict,name,class_names):
	cm = confusion_matrix(y_test, y_predict)
	fig, ax = plt.subplots(figsize=(6,6))
	tick_marks = np.arange(len(class_names))
	plt.xticks(tick_marks, class_names)
	plt.yticks(tick_marks, class_names)
	sns.heatmap(pd.DataFrame(cm), annot=True, cmap="Purples" ,fmt='g')
	ax.xaxis.set_label_position("top")
	plt.tight_layout()
	plt.title('Confusion Matrix', y=1.1)
	plt.ylabel('Actual label')
	plt.xlabel('Predicted label')
	save_fig(name)
	plt.show()
	return

def create_batches(data_path):
    test_path = data_path + '/test'

    #  Create a data augmentor
    data_augmentor = ImageDataGenerator(
                                samplewise_center=True,
                                rescale=1./255,
                                shear_range=0.2,
                                zoom_range = 0.2,
                                samplewise_std_normalization=True,
                                validation_split=0.2)

    test_batches = data_augmentor.flow_from_directory(
                                                directory=test_path,
                                                target_size=TARGET_SIZE,
                                                shuffle=False,
                                                batch_size=BATCH_SIZE)


    return test_batches



def test_model(model_path,data_path):
  labels  = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
  test_batches = create_batches(data_path)
  test_labels = test_batches.classes
  model = load_modell(model_path)
  predictions = model.predict(test_batches)
  y_pred=predictions.argmax(axis=1)

  plot_confusion(test_labels,y_pred, 'basic_conf', labels)
  test_loss, test_acc = model.evaluate(test_batches, batch_size=BATCH_SIZE)
  return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test Model")
    parser.add_argument("--model_path", required=True, help="Model Path")
    parser.add_argument("--data_path",required=True, help="Dataset Path")
    args = parser.parse_args()
    test_model(args.model_path,args.data_path)