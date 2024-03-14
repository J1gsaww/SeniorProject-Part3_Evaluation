import tensorflow as tf
import numpy as np
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input, decode_predictions
import os
import random
from PIL import Image
from IPython.display import display
from keras.preprocessing.image import load_img, img_to_array

# Load Model
model_path = 'best_model3cls.h5'
model = tf.keras.models.load_model(model_path)
model.summary()

target_img_shape =(100,100)

train = r'C:\Users\User\Desktop\Senior Project\Implementation\dataset\data\train' # Path for train set
val = r'C:\Users\User\Desktop\Senior Project\Implementation\dataset\data\valid' # Path for validate set
test = r'C:\Users\User\Desktop\Senior Project\Implementation\dataset\Test\test' # Path for test set

train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
train_set = train_datagen.flow_from_directory(train,
                                              target_size = target_img_shape,
                                              batch_size = 32,
                                              class_mode='sparse')
val_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
val_set = val_datagen.flow_from_directory(val,
                                          target_size = target_img_shape,
                                          batch_size = 32,
                                          class_mode='sparse')
test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
test_set = test_datagen.flow_from_directory(test,
                                          target_size = target_img_shape,
                                          batch_size = 32,
                                          class_mode='sparse')


from keras.preprocessing.image import load_img, img_to_array
labels = (train_set.class_indices)
labels = dict((v,k) for k,v in labels.items())
labels

def predict(img_fname):
  img = load_img(img_fname,target_size = target_img_shape)
  plt.imshow(img)
  img = img_to_array(img)
  img = np.expand_dims(img, axis= 0)
  img = preprocess_input(img)

  pred = model.predict(img) ; print(pred[0])
  pred_cls = labels[np.argmax(pred, -1)[0]]
  print('Prediction:', pred_cls)

folder_path = r'C:\Users\User\Desktop\Senior Project\Implementation\dataset\Test\test'
truePositive = False
truePositive_count = 0
falseCase = 0

def predict_keepclass(img_fname):
  img = load_img(img_fname,target_size = target_img_shape)
  img = img_to_array(img)
  img = np.expand_dims(img, axis= 0)
  img = preprocess_input(img)

  pred = model.predict(img)
  pred_cls = labels[np.argmax(pred, -1)[0]]
  return pred_cls


while True:
  png_files = [i for i in os.listdir(folder_path) if i.endswith(".png")]

  if not png_files:
    print("Folder is empty")
    break
  random_png = random.choice(png_files) #Contain file name

  image_path = os.path.join(folder_path, random_png)
  image = Image.open(image_path)

  #display(image)
  print("File name: ", random_png)
  #predict(image_path)

  image_name_characters = list(random_png)
  image_predict_class = predict_keepclass(image_path)
  print("Actual Class: ", image_name_characters[0])
  print("Predict Class: ", image_predict_class)

  #Check
  if image_predict_class == image_name_characters[0]:
    truePositive = True
  else: truePositive = False

  if truePositive == True:
    truePositive_count = truePositive_count + 1
  else: falseCase = falseCase + 1

  #remove file
  os.remove(image_path)
  print(f"File '{random_png}' has been deleted.")

sample_count = truePositive_count + falseCase
print("True Positive Count: ", truePositive_count)
print("False Case Count: ", falseCase)
print("Sample Count: ", sample_count)


conf_matrix = np.zeros((57, 57), dtype=int)


for i in range(len(test_set)):
    batch_images, batch_labels = test_set[i]
    preds = model.predict(batch_images)
    pred_classes = np.argmax(preds, axis=1)
    
    
    for true_label, pred_label in zip(batch_labels, pred_classes):
        true_label = int(true_label)
        pred_label = int(pred_label)
        if true_label == pred_label:
            conf_matrix[true_label][pred_label] += 1
        elif true_label != pred_label:
           conf_matrix[true_label][pred_label] +=1
#Test Zone
#Display TP output in label forms
'''
for j, (true_label, pred_label) in enumerate(zip(batch_labels, pred_classes)):
    true_label = int(true_label)
    pred_label = int(pred_label)
    if true_label == pred_label:
            # Print the label of True Positives
            print(f"Sample {i * len(test_set) + j + 1}: True Label - {labels[true_label]}, Predicted Label - {labels[pred_label]}")
'''

rows = 57
cols = 57
total_samples = np.sum(conf_matrix)
print("\nTotal Samples:",total_samples)

TP = np.sum(np.diag(conf_matrix)) # True Positive
print("\nTrue Positive:", TP)

FP = total_samples-TP # False Positive
print("\nFalse Positive: ", FP)

FN = FP # False Negative
print("\nFalse Negative: ", FN)

TN = (rows*cols*np.sum(np.diag(conf_matrix)))-total_samples # True Negative
print("\nTrue Negative: ",TN)

acc = (TP+TN)/(TP+FP+FN+TN)
print("\nAccuracy: ",acc)

precision = TP/(TP+FP)
print("\nPrecision: ",precision)

recall = TP/(TP+FN)
print("\nRecall: ",recall)

f1_score = 2*precision*recall/(precision+recall)
print("\nF1-score: ",f1_score)

#Check labels
#print(labels)

# Labels with Thai language
plt.rcParams['font.family'] = 'TH Sarabun New'

# Plot Confusion Matrix 2*2
conf_matrix_2x2 = np.array([[TP, FP], [FN, TN]]) 

plt.figure(figsize=(20, 15))
sns.heatmap(conf_matrix_2x2, cmap='Blues', annot=True, fmt='d',annot_kws={"size": 16})
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('2x2 Confusion Matrix')
plt.xticks([0.5, 1.5], ['Positive', 'Negative'])
plt.yticks([0.5, 1.5], ['Positive', 'Negative'])
plt.show()


# Plot Confusion Matrix 57*57

plt.figure(figsize=(20, 15))
sns.heatmap(conf_matrix, cmap='Blues', annot=True, xticklabels=labels.values(), yticklabels=labels.values(), fmt='d')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


'''
plt.matshow(conf_matrix)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
'''
# display true labels / predicted labels
'''
for i in range(len(test_set)):
    batch_images, batch_labels = test_set[i]
    preds = model.predict(batch_images)
    pred_classes = np.argmax(preds, axis=1)

    for true_label, pred_label in zip(batch_labels, pred_classes):
        print("True Label:", true_label, "| Predicted Label:", pred_label)
'''

