from keras.preprocessing.image import ImageDataGenerator
import numpy as np

batches = ImageDataGenerator().flow_from_directory( 'B',
                                                shuffle=False, 
                                                target_size=(128, 128),
                                                batch_size=64,
                                                class_mode='categorical',
                                                )

sum = np.array([0.0, 0.0, 0.0]);
count = 0
mean = [129.41234668, 100.69618055, 101.47650676]
for imgs, labels in batches:
        imgs = np.delete(imgs, np.s_[3:], axis=1)
        imgs=np.subtract(imgs, mean)
        imgs=np.square(imgs)
        sum += np.sum(imgs, axis=(0, 1, 2))
        print ('%d/%d - %0.2f%%' % (count, batches.samples, 100.0*count/batches.samples), "\r",)
        count += imgs.shape[0]
        if count >= batches.samples:
                break


print('sum:',sum)
sum=np.sum(sum)
avg = sum/(count*128*128*3)
print(avg)
print('std:',np.sqrt(avg))


# A mean [107.10923645  78.02965603  76.23650783]  std 15.07452842463122
# [0.42023899 0.30655081 0.29950314] std: 0.059090827181001726
# B mean [129.41234668 100.69618055 101.47650676] 15.248858915461037
# [0.50732507 0.39511367 0.39813832] std: 0.05979150389245361








# from keras.preprocessing.image import img_to_array, load_img
# img = load_img('B/a/a3/a/color_0_0002.png')  
# x = img_to_array(img) 
# x = x.reshape((1,) + x.shape)  
# train_datagen.fit(x)
# test_datagen.fit(x)

# sample_set = ImageDataGenerator().flow_from_directory('B/a/a3')
# train_datagen.fit(sample_set)
# test_datagen.fit(sample_set)


# cnn model training & testing
# Image Preprocessing
# class FixedImageDataGenerator(ImageDataGenerator):
#     def standardize(self, x):
#         if self.featurewise_center:
#             x = ((x/255.) - 0.5) * 2.
#         return x














# if self.featurewise_center:
#   x -= self.mean
# if self.featurewise_std_normalization:
#   x /= (self.std + 1e-7)