"""Load and prepare the binary APTOS dataset."""

from pathlib import Path    #FOR FILE/FOLDERS PATHS

import numpy as np
import tensorflow as tf     #LOAD , PROCESS & PREPARE IMAGES
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet50 import preprocess_input     #PREPARE IMAGES IN FORMAT FOR RESNET50

# Image settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32     #MODEL PROCESS 32 IMAGES AT THE TIME
SEED = 42

CLASS_NAMES = ["No DR", "DR"]

# Dataset path
ROOT_DIR = Path(__file__).resolve().parents[1]      #STORE PYTHON FILE LOCATION
DATASET_DIR = ROOT_DIR / "datasets" / "APTOS 2019 (Original) (Binary)"      #COMPLETE PATH TO DATASET


def get_image_paths():
  image_paths = []      #STORE ALL IMAGES PATHS
  labels = []           #STORE THE CLASS NO. OF IMAGES

  extensions = {".jpg", ".jpeg", ".png"}

  for label, class_name in enumerate(CLASS_NAMES):      #ENUMERATE()    GIVE EACH CLASS ITS NO.
    class_dir = DATASET_DIR / class_name            #CREATE PATH FOR CURRENT CLASS FOLDER

    if not class_dir.exists():
      raise FileNotFoundError(f"Folder not found: {class_dir}")

    files = [
        p for p in class_dir.rglob("*") if p.suffix.lower() in extensions       #RGLOB("*") EARCH THROUGH FOLDER AND SUBFOLDER
    ]

    for file in files:      #ADD PATH TO IMAGES_PATHS AND CLASS NO TO LABELS
      image_paths.append(str(file))     #CONVERT PATH OBJ TO STRING
      labels.append(label)

  if len(image_paths) == 0:
    raise ValueError("No images found in dataset folders.")

  return np.array(image_paths), np.array(labels)

def load_image(path, label):
  image = tf.io.read_file(path)

  image = tf.image.decode_image(            #CONVERT FILE CONTENT TO ACTUAL IMAGE FOR TENSORFLOW
      image, channels=3, expand_animations=False
  )

  image.set_shape([None, None, 3])
  image = tf.image.resize(image, IMAGE_SIZE)        #224X224

  # ResNet50 preprocessing
  image = preprocess_input(image)

  # Fixed: Cast label to float32 without adding an extra (1,) dimension
  label = tf.cast(label, tf.float32)

  return image, label


def create_dataset(paths, labels, training=False):
  dataset = tf.data.Dataset.from_tensor_slices((paths, labels))     #COMBINE IMAGE PATHS AND LABEL INTO DATA SET

  if training:
    dataset = dataset.shuffle(buffer_size=len(paths), seed=SEED)

  dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)        #[load_image]= EVERY IMAGE: (READ, DECODE, RESIZEW, PREPROCESSED, PAIRED WITH LABEL)
                                                                                #AUTOTUNE= PARALLEL PROCESSING TO IMPROVE SPEED (TENSORFLOE)
  dataset = dataset.batch(BATCH_SIZE)       #BATCH_SIZE = 32 TRAINING IMAGES / 32 = [VALUE] BATCHES
  dataset = dataset.prefetch(tf.data.AUTOTUNE)      #LOAD & PREPARE NEXT BATCH

  return dataset


def load_data():
  paths, labels = get_image_paths()

  # 80% training, 20% remaining
  train_paths, temp_paths, train_labels, temp_labels = train_test_split(
      paths, labels, test_size=0.20, random_state=SEED, stratify=labels     #STRATIFY = MAINTAIN BOTH CLASSES PROPORTIION IN THE SPLIT
  )

  # 10% validation, 10% testing
  val_paths, test_paths, val_labels, test_labels = train_test_split(
      temp_paths, temp_labels, test_size=0.50, random_state=SEED, stratify=temp_labels
  )

  train_ds = create_dataset(train_paths, train_labels, training=True)
  val_ds = create_dataset(val_paths, val_labels)
  test_ds = create_dataset(test_paths, test_labels)

  print("Total images:", len(paths))
  print("Training images:", len(train_paths))
  print("Validation images:", len(val_paths))
  print("Testing images:", len(test_paths))

  print("Training class distribution:", np.bincount(train_labels))
  print("Validation class distribution:", np.bincount(val_labels))
  print("Testing class distribution:", np.bincount(test_labels))

  return train_ds, val_ds, test_ds