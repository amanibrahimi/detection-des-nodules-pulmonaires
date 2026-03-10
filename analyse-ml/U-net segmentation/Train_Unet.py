 # -- coding: utf-8 --
import csv
import glob
import random
import cv2
import numpy
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageOps
from typing import List, Tuple
from tensorflow import Tensor
from keras.optimizers import SGD
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, merge, BatchNormalization, SpatialDropout2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

# ================================================================
# SECTION 1 : CONFIGURATION ET PARAMÈTRES GLOBAUX
# ================================================================
MEAN_FRAME_COUNT = 1
CHANNEL_COUNT = 1
SEGMENTER_IMG_SIZE = 320
MODEL_DIR = './model/again/'   # Dossier de sauvegarde des modèles
BATCH_SIZE = 8
ELASTIC_INDICES = None

TRAIN_LIST = ''
VAL_LIST = ''
TRAIN_TEMP_DIR = './temp_dir/chapter4/'

# ================================================================
# SECTION 2 : OUTILS D'AUGMENTATION DE DONNÉES (DATA AUGMENTATION)
# Ces fonctions permettent de créer artificiellement de nouvelles images 
# pour améliorer la robustesse du modèle.
# ================================================================

class XYRange:
    """Classe utilitaire pour définir des plages de transformation X/Y."""
    def __init__(self, x_min, x_max, y_min, y_max, chance=1.0):
        self.chance = chance
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.last_x = 0
        self.last_y = 0

    def get_last_xy_txt(self):
        return f"x_{str(int(self.last_x * 100)).replace('-', 'm')}-y_{str(int(self.last_y * 100)).replace('-', 'm')}"

def random_scale_img(img, xy_range, lock_xy=False):
    """Redimensionnement aléatoire (Zoom) en conservant la taille de sortie."""
    if random.random() > xy_range.chance:
        return img
    if not isinstance(img, list): img = [img]

    scale_x = random.uniform(xy_range.x_min, xy_range.x_max)
    scale_y = scale_x if lock_xy else random.uniform(xy_range.y_min, xy_range.y_max)
    
    org_height, org_width = img[0].shape[:2]
    res = []
    for img_inst in img:
        scaled_width, scaled_height = int(org_width * scale_x), int(org_height * scale_y)
        scaled_img = cv2.resize(img_inst, (scaled_width, scaled_height), interpolation=cv2.INTER_CUBIC)
        # Gestion des bordures si l'image devient plus petite ou plus grande
        # ... (Logique de recadrage/padding pour maintenir SEGMENTER_IMG_SIZE)
        res.append(scaled_img) # Note: Simplifié pour la lisibilité
    return res

def random_translate_img(img, xy_range, padding="constant"):
    """Translation aléatoire (déplacement horizontal/vertical)."""
    if random.random() > xy_range.chance: return img
    if not isinstance(img, list): img = [img]
    
    org_height, org_width = img[0].shape[:2]
    translate_x = random.randint(xy_range.x_min, xy_range.x_max)
    translate_y = random.randint(xy_range.y_min, xy_range.y_max)
    trans_matrix = numpy.float32([[1, 0, translate_x], [0, 1, translate_y]])
    
    border_const = cv2.BORDER_REFLECT if padding == "reflect" else cv2.BORDER_CONSTANT
    res = [cv2.warpAffine(i, trans_matrix, (org_width, org_height), borderMode=border_const) for i in img]
    return res[0] if len(res) == 1 else res

def random_rotate_img(img, chance, min_angle, max_angle):
    """Rotation aléatoire de l'image autour de son centre."""
    if random.random() > chance: return img
    if not isinstance(img, list): img = [img]
    
    angle = random.randint(min_angle, max_angle)
    center = (img[0].shape[1] / 2, img[0].shape[0] / 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    return [cv2.warpAffine(i, rot_matrix, (i.shape[1], i.shape[0]), borderMode=cv2.BORDER_CONSTANT) for i in img]

def random_flip_img(img, horizontal_chance=0.0, vertical_chance=0.0):
    """Retournement horizontal ou vertical (Mirroring)."""
    flip_h = random.random() < horizontal_chance
    flip_v = random.random() < vertical_chance
    if not flip_h and not flip_v: return img
    
    flip_val = 1 if flip_h and not flip_v else (0 if flip_v and not flip_h else -1)
    if not isinstance(img, list): return cv2.flip(img, flip_val)
    return [cv2.flip(i, flip_val) for i in img]

def elastic_transform(image, alpha, sigma, random_state=None):
    """Déformation élastique pour simuler des variations organiques (très utile en médical)."""
    global ELASTIC_INDICES
    shape = image.shape
    if ELASTIC_INDICES is None:
        if random_state is None: random_state = numpy.random.RandomState(1301)
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        x, y = numpy.meshgrid(numpy.arange(shape[0]), numpy.arange(shape[1]))
        ELASTIC_INDICES = numpy.reshape(y + dy, (-1, 1)), numpy.reshape(x + dx, (-1, 1))
    return map_coordinates(image, ELASTIC_INDICES, order=1).reshape(shape)

def random_distort(img):
    """Distorsion de la luminosité, du contraste et des couleurs via PIL."""
    def r_brightness(i): return ImageEnhance.Brightness(i).enhance(numpy.random.uniform(0.5, 1.5))
    def r_contrast(i): return ImageEnhance.Contrast(i).enhance(numpy.random.uniform(0.5, 1.5))
    def r_color(i): return ImageEnhance.Color(i).enhance(numpy.random.uniform(0.5, 1.5))
    
    ops = [r_brightness, r_contrast, r_color]
    numpy.random.shuffle(ops)
    
    if not isinstance(img, list):
        img_pil = Image.fromarray(img)
        for op in ops: img_pil = op(img_pil)
        return numpy.asarray(img_pil)
    
    res = []
    for item in img:
        img_pil = Image.fromarray(item)
        for op in ops: img_pil = op(img_pil)
        res.append(numpy.asarray(img_pil))
    return res

# ================================================================
# SECTION 3 : PRÉPARATION DES FICHIERS ET NORMALISATION
# ================================================================

def prepare_image_for_net(img):
    """Normalise les pixels entre 0 et 1 et ajuste les dimensions pour Keras."""
    img = img.astype(numpy.float32) / 255.
    if len(img.shape) == 3:
        return img.reshape(img.shape[-3], img.shape[-2], img.shape[-1])
    return img.reshape(1, img.shape[-2], img.shape[-1], 1)

def get_train_holdout_files():
    """Charge les listes d'images et de masques depuis les fichiers CSV/TXT."""
    def load_list(path):
        with open(path, 'r') as f:
            return [ (row[0], row[0].replace("_img.png", "_mask.png")) for row in csv.reader(f) if row ]
    
    train = load_list(TRAIN_LIST)
    val = load_list(VAL_LIST)
    random.shuffle(train)
    return train, val

# ================================================================
# SECTION 4 : FONCTIONS DE PERTE ET CALLBACKS
# Le Dice Coefficient mesure la superposition entre le masque prédit et le réel.
# ================================================================

def dice_coef(y_true, y_pred):
    """Calcul du coefficient de Dice (métrique de similarité)."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 100) / (K.sum(y_true_f) + K.sum(y_pred_f) + 100)

def dice_coef_loss(y_true, y_pred):
    """La perte est l'opposé du Dice : on veut minimiser cette valeur."""
    return -dice_coef(y_true, y_pred)

class DumpPredictions(Callback):
    """Sauvegarde des images de prédiction à chaque fin d'époque pour le monitoring visuel."""
    def __init__(self, dump_filelist, model_type):
        super().__init__()
        self.dump_filelist = dump_filelist
        if not os.path.exists(TRAIN_TEMP_DIR): os.mkdir(TRAIN_TEMP_DIR)

    def on_epoch_end(self, epoch, logs=None):
        generator = image_generator(self.dump_filelist, 1, train_set=False)
        for i in range(10):
            x, y = next(generator)
            y_pred = self.model.predict(x, batch_size=1)
            # Conversion et sauvegarde des images i (input), o (output/mask), p (predicted)
            cv2.imwrite(f"{TRAIN_TEMP_DIR}img_{epoch:03d}_{i:02d}_p.png", (y_pred[0]*255).astype(numpy.uint8))

# ================================================================
# SECTION 5 : ARCHITECTURE DU RÉSEAU (U-NET)
# Structure en 'U' avec une partie descendante (extraction) et ascendante (reconstruction).
# ================================================================



def get_unet(learn_rate=0.0001) -> Model:
    """Définit l'architecture U-Net avec BatchNormalization."""
    inputs = Input((SEGMENTER_IMG_SIZE, SEGMENTER_IMG_SIZE, CHANNEL_COUNT))
    filter_size = 32
    growth_step = 32
    
    # --- Encodeur (Descente) ---
    x = BatchNormalization()(inputs)
    conv1 = Conv2D(filter_size, (3, 3), activation='relu', padding='same')(x)
    conv1 = Conv2D(filter_size, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # ... (Répétition des blocs de convolution et pooling avec augmentation des filtres)
    # Note : Le code original contient plusieurs blocs similaires (conv2 à conv6)
    
    # --- Décodeur (Montée avec Skip Connections) ---
    # On utilise merge.concatenate pour lier les couches de l'encodeur aux couches montantes
    # Cela permet de conserver les détails spatiaux fins.
    
    up10 = UpSampling2D(size=(2, 2))(conv9)
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(up10) # Sortie finale : masque binaire

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=SGD(lr=learn_rate, momentum=0.9, nesterov=True),
                  loss=dice_coef_loss, metrics=[dice_coef])
    model.summary()
    return model

# ================================================================
# SECTION 6 : BOUCLE D'ENTRAÎNEMENT PRINCIPALE
# ================================================================

def train_model(model_type, continue_from=None):
    """Lance l'entraînement, gère les générateurs et affiche les courbes de perte."""
    train_files, holdout_files = get_train_holdout_files()
    train_gen = image_generator(train_files, BATCH_SIZE, True)
    holdout_gen = image_generator(holdout_files, BATCH_SIZE, False)

    model = get_unet(0.001 if continue_from is None else 0.0005)
    if continue_from: model.load_weights(continue_from)

    # Configuration des sauvegardes et logs
    checkpoint = ModelCheckpoint(MODEL_DIR + model_type + "_{epoch:02d}.hd5", monitor='val_loss', save_best_only=True)
    dumper = DumpPredictions(holdout_files[::10], model_type)
    
    # Lancement du fit
    hist = model.fit_generator(train_gen, steps_per_epoch=200, epochs=80, 
                               validation_data=holdout_gen, callbacks=[checkpoint, dumper], validation_steps=10)

    # Affichage et sauvegarde des courbes de résultats (Loss & Accuracy)
    plt.figure(1)
    plt.plot(hist.history['loss'], label='train')
    plt.plot(hist.history['val_loss'], label='val')
    plt.title('Perte du Modèle (Loss)')
    plt.legend()
    plt.savefig("./temp_dir/chapter5/Unet_loss_curve.jpg")

if __name__ == "__main__":
    TRAIN_LIST = './data/chapter4/train_img.txt'
    VAL_LIST = './data/chapter4/val_img.txt'
    train_model(model_type='u-net', continue_from='./model/unet.hd5')