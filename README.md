# 11124237 朱瓊月 ， 11123142 林韋莘
# COVID-19：使用深度學習的醫學診斷

# 介紹
正在進行的名為COVID-19 的全球大流行是由SARS-COV-2引起的，該病毒傳播迅速並發生變異，引發了幾波疫情，主要影響第三世界和發展中國家。隨著世界各國政府試圖控制傳播，受影響的人數正穩定上升。
![image](https://github.com/user-attachments/assets/904519ac-9d46-44a2-b344-2ae70cf6e84b)
本文將使用CoronaHack-Chest X 光資料集。它包含胸部X 光影像，我們必須找到受冠狀病毒影響的影像。
我們之前談到的SARS-COV-2 是主要影響呼吸系統的病毒類型，因此胸部X 光是我們可以用來識別受影響肺部的重要影像方法之一。這是一個並排比較：
![image](https://github.com/user-attachments/assets/6fbb0acd-e7fb-4cef-b886-e53bfecc05cb)
如你所見，COVID-19 肺炎如何吞噬整個肺部，並且比細菌和病毒類型的肺炎更危險。
本文，將使用深度學習和遷移學習對受Covid-19影響的肺部的X 光影像進行分類和識別。
# 導入庫和載入數據
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import numpy as np
import pandas as pd
sns.set()
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  *
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.applications import DenseNet121, VGG19, ResNet50
import PIL.Image
import matplotlib.pyplot as mpimg
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from sklearn.utils import shuffle
train_df = pd.read_csv('/content/drive/MyDrive/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/Chest_xray_Corona_Metadata.csv')
train_df.shape
train_df.head(5)
![image](https://github.com/user-attachments/assets/2228b217-d760-4c34-a26e-151fef6f5256)
//
train_df.info()
![image](https://github.com/user-attachments/assets/23f23205-8ccf-4a8f-a0b6-21a337438727)

# 處理缺失值
//
missing_vals = train_df.isnull().sum()
missing_vals.plot(kind = 'bar')
![image](https://github.com/user-attachments/assets/2bd04c32-bfbf-484c-a887-8c6b780aa7b6)
//
train_df.dropna(how = 'all')
train_df.isnull().sum()
![image](https://github.com/user-attachments/assets/f0b796ef-a6a6-4ecc-b38a-fd3b8a757f2c)
//
train_df.fillna('unknown', inplace=True)
train_df.isnull().sum()
![image](https://github.com/user-attachments/assets/d1b46102-ed74-42e9-8e88-283e84064c88)
//
train_data = train_df[train_df['Dataset_type'] == 'TRAIN']
test_data = train_df[train_df['Dataset_type'] == 'TEST']
assert train_data.shape[0] + test_data.shape[0] == train_df.shape[0]
print(f"Shape of train data : {train_data.shape}")
print(f"Shape of test data : {test_data.shape}")
test_data.sample(10)
![image](https://github.com/user-attachments/assets/5120d645-7861-42f6-b703-6d1287200d19)
//
print((train_df['Label_1_Virus_category']).value_counts())
print('--------------------------')
print((train_df['Label_2_Virus_category']).value_counts())
![image](https://github.com/user-attachments/assets/d5a1f33f-cdde-41dc-a69e-f8f2c63b3b8a)
因此標籤2 類別包含COVID-19案例
# 顯示影像
//
test_img_dir = '/content/drive/MyDrive/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test'
train_img_dir = '/content/drive/MyDrive/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train'


sample_train_images = list(os.walk(train_img_dir)) [0][2][:8]
sample_train_images = list(map(lambda x: os.path.join(train_img_dir, x), sample_train_images))


sample_test_images = list(os.walk(test_img_dir)) [0][2][:8]
sample_test_images = list(map(lambda x: os.path.join(test_img_dir, x), sample_test_images))


plt.figure(figsize = (10,10))
for iterator, filename in enumerate(sample_train_images):
    image = PIL.Image.open(filename)
    plt.subplot(4,2,iterator+1)
    plt.imshow(image, cmap=plt.cm.bone)


plt.tight_layout()
![image](https://github.com/user-attachments/assets/06ca2dc0-4a69-4d14-994a-f69ab41263d3)
# 視覺化
//
plt.figure(figsize=(15,10))
sns.countplot(train_data['Label_2_Virus_category']);
![image](https://github.com/user-attachments/assets/a32eaac4-5008-4f25-b34a-0dafdae1bdd3)

//
fig, ax = plt.subplots(4, 2, figsize=(15, 10))




covid_path = train_data[train_data['Label_2_Virus_category']=='COVID-19']['X_ray_image_name'].values


sample_covid_path = covid_path[:4]
sample_covid_path = list(map(lambda x: os.path.join(train_img_dir, x), sample_covid_path))


for row, file in enumerate(sample_covid_path):
    image = plt.imread(filename)
    ax[row, 0].imshow(image, cmap=plt.cm.bone)
    ax[row, 1].hist(image.ravel(), 256, [0,256])
    ax[row, 0].axis('off')
    if row == 0:
        ax[row, 0].set_title('Images')
        ax[row, 1].set_title('Histograms')
fig.suptitle('Label 2 Virus Category = COVID-19', size=16)
plt.show()
![image](https://github.com/user-attachments/assets/a0e5bb17-39d2-41bc-a140-5f68f5b92402)
# 對於正常情況
//
fig, ax = plt.subplots(4, 2, figsize=(15, 10))




normal_path = train_data[train_data['Label']=='Normal']['X_ray_image_name'].values


sample_normal_path = normal_path[:4]
sample_normal_path = list(map(lambda x: os.path.join(train_img_dir, x), sample_normal_path))


for row, file in enumerate(sample_normal_path):
    image = plt.imread(file)
    ax[row, 0].imshow(image, cmap=plt.cm.bone)
    ax[row, 1].hist(image.ravel(), 256, [0,256])
    ax[row, 0].axis('off')
    if row == 0:
        ax[row, 0].set_title('Images')
        ax[row, 1].set_title('Histograms')
fig.suptitle('Label = NORMAL', size=16)
plt.show()
![image](https://github.com/user-attachments/assets/43cba9f7-eb90-47a4-9e74-97e397465de2)
//
final_train_data = train_data[(train_data['Label'] == 'Normal') |
                              ((train_data['Label'] == 'Pnemonia') &
                               (train_data['Label_2_Virus_category'] == 'COVID-19'))]


final_train_data['class'] = final_train_data.Label.apply(lambda x: 'negative' if x=='Normal' else 'positive')
test_data['class'] = test_data.Label.apply(lambda x: 'negative' if x=='Normal' else 'positive')


final_train_data['target'] = final_train_data.Label.apply(lambda x: 0 if x=='Normal' else 1)
test_data['target'] = test_data.Label.apply(lambda x: 0 if x=='Normal' else 1)


final_train_data = final_train_data[['X_ray_image_name', 'class', 'target', 'Label_2_Virus_category']]
final_test_data = test_data[['X_ray_image_name', 'class', 'target']]


test_data['Label'].value_counts()
# 數據增強
//
from tensorflow.keras.preprocessing import image # Import the image module

datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
)

def read_img(filename, size, path):
    img = image.load_img(os.path.join(path, filename), target_size=size) # Use image.load_img
    #convert image to array
    img = image.img_to_array(img) / 255 # Use image.img_to_array
    return img

samp_img = read_img(final_train_data['X_ray_image_name'][0],
                                 (255,255),
                                 train_img_dir) # Changed train_img_path to train_img_dir

plt.figure(figsize=(10,10))
plt.suptitle('Data Augmentation', fontsize=28)

i = 0

for batch in datagen.flow(tf.expand_dims(samp_img,0), batch_size=6):
    plt.subplot(3, 3, i+1)
    plt.grid(False)
    plt.imshow(batch.reshape(255, 255, 3));
    
    if i == 8:
        break
    i += 1
    
plt.show();
![image](https://github.com/user-attachments/assets/c6b11f68-f842-4b82-b8e0-ed216aeb2858)

//

corona_df = final_train_data[final_train_data['Label_2_Virus_category'] == 'COVID-19']
with_corona_augmented = []


def augment(name):
    img = read_img(name, (255,255), train_img_dir)
    i = 0
    for batch in tqdm(datagen.flow(tf.expand_dims(img, 0), batch_size=32)):
        with_corona_augmented.append(tf.squeeze(batch).numpy())
        if i == 20:
            break
        i =i+1


corona_df['X_ray_image_name'].apply(augment)
注意：輸出太長，無法包含在文章中。這是其中的一小部分。
![image](https://github.com/user-attachments/assets/3295566c-46a0-477b-9394-7744b9442824)
