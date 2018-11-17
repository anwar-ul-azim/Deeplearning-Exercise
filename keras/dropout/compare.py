classifier.add(Conv2D(30, kernel_size=3, activation='relu', input_shape=(128, 128, 3)))
classifier.add(Conv2D(30, kernel_size=3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(70, kernel_size=3, activation='relu'))
classifier.add(Conv2D(70, kernel_size=3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())
classifier.add(Dense(300, activation='relu'))
classifier.add(Dense(300, activation='relu'))
classifier.add(Dense(300, activation='relu'))
classifier.add(Dense(units=24, activation='softmax'))

Epoch 1/10
414/414 [==============================] - 141s 341ms/step - loss: 1.1503 - acc: 0.6364 - val_loss: 2.4199 - val_acc: 0.4423
Epoch 2/10
414/414 [==============================] - 132s 319ms/step - loss: 0.3270 - acc: 0.8904 - val_loss: 2.3497 - val_acc: 0.4960
Epoch 3/10
414/414 [==============================] - 144s 347ms/step - loss: 0.1926 - acc: 0.9364 - val_loss: 2.3484 - val_acc: 0.5100
Epoch 4/10
414/414 [==============================] - 198s 479ms/step - loss: 0.1461 - acc: 0.9518 - val_loss: 2.6016 - val_acc: 0.4788
Epoch 5/10
414/414 [==============================] - 143s 344ms/step - loss: 0.1158 - acc: 0.9632 - val_loss: 2.6995 - val_acc: 0.5050
Epoch 6/10
414/414 [==============================] - 136s 329ms/step - loss: 0.1035 - acc: 0.9671 - val_loss: 2.2332 - val_acc: 0.5555
Epoch 7/10
414/414 [==============================] - 142s 343ms/step - loss: 0.0852 - acc: 0.9722 - val_loss: 2.5768 - val_acc: 0.5316
Epoch 8/10
414/414 [==============================] - 134s 323ms/step - loss: 0.0821 - acc: 0.9737 - val_loss: 2.3877 - val_acc: 0.5208


classifier.add(Conv2D(30, kernel_size=3, activation='relu', input_shape=(128, 128, 3)))
classifier.add(Conv2D(30, kernel_size=3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.5))

classifier.add(Conv2D(70, kernel_size=3, activation='relu'))
classifier.add(Conv2D(70, kernel_size=3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.5))


classifier.add(Flatten())
classifier.add(Dense(300, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(300, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(300, activation='relu'))
classifier.add(Dense(units=24, activation='softmax'))

Epoch 1/10
414/414 [==============================] - 143s 345ms/step - loss: 3.2171 - acc: 0.0535 - val_loss: 3.1804 - val_acc: 0.0443
Epoch 2/10
414/414 [==============================] - 136s 329ms/step - loss: 3.1758 - acc: 0.0526 - val_loss: 3.1809 - val_acc: 0.0442
Epoch 3/10
414/414 [==============================] - 136s 328ms/step - loss: 2.1287 - acc: 0.3098 - val_loss: 1.8726 - val_acc: 0.4484
Epoch 4/10
414/414 [==============================] - 136s 327ms/step - loss: 1.1088 - acc: 0.6217 - val_loss: 1.6350 - val_acc: 0.5026
Epoch 5/10
414/414 [==============================] - 135s 326ms/step - loss: 0.8100 - acc: 0.7211 - val_loss: 1.6444 - val_acc: 0.5141
Epoch 6/10
414/414 [==============================] - 135s 326ms/step - loss: 0.6441 - acc: 0.7815 - val_loss: 1.6237 - val_acc: 0.5308
Epoch 7/10
414/414 [==============================] - 134s 324ms/step - loss: 0.5571 - acc: 0.8130 - val_loss: 1.5839 - val_acc: 0.5419
Epoch 8/10
414/414 [==============================] - 134s 323ms/step - loss: 0.4762 - acc: 0.8420 - val_loss: 1.7245 - val_acc: 0.5288
Epoch 9/10
414/414 [==============================] - 134s 324ms/step - loss: 0.4249 - acc: 0.8591 - val_loss: 1.6778 - val_acc: 0.5548
Epoch 10/10
414/414 [==============================] - 136s 327ms/step - loss: 0.3886 - acc: 0.8720 - val_loss: 1.7427 - val_acc: 0.5089


classifier.add(Conv2D(30, kernel_size=3, activation='relu', input_shape=(128, 128, 3)))
classifier.add(Conv2D(30, kernel_size=3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.5))

classifier.add(Conv2D(70, kernel_size=3, activation='relu'))
classifier.add(Conv2D(70, kernel_size=3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.5))

classifier.add(Flatten())
classifier.add(Dense(300, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(300, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(300, activation='relu'))
classifier.add(Dense(units=24, activation='softmax'))

Epoch 1/10
414/414 [==============================] - 144s 347ms/step - loss: 2.2362 - acc: 0.3266 - val_loss: 2.2146 - val_acc: 0.3744
Epoch 2/10
414/414 [==============================] - 141s 341ms/step - loss: 1.0904 - acc: 0.6359 - val_loss: 2.5693 - val_acc: 0.3798
Epoch 3/10
414/414 [==============================] - 140s 339ms/step - loss: 0.7713 - acc: 0.7456 - val_loss: 2.1903 - val_acc: 0.4645
Epoch 4/10
414/414 [==============================] - 141s 340ms/step - loss: 0.5972 - acc: 0.8072 - val_loss: 1.8800 - val_acc: 0.4905
Epoch 5/10
414/414 [==============================] - 139s 337ms/step - loss: 0.5087 - acc: 0.8391 - val_loss: 2.0832 - val_acc: 0.5067
Epoch 6/10
414/414 [==============================] - 140s 337ms/step - loss: 0.4272 - acc: 0.8655 - val_loss: 2.6926 - val_acc: 0.4152
Epoch 7/10
414/414 [==============================] - 140s 338ms/step - loss: 0.3823 - acc: 0.8792 - val_loss: 2.1295 - val_acc: 0.4985
Epoch 8/10
414/414 [==============================] - 140s 339ms/step - loss: 0.3408 - acc: 0.8930 - val_loss: 2.3352 - val_acc: 0.5109
Epoch 9/10
414/414 [==============================] - 145s 349ms/step - loss: 0.3248 - acc: 0.9009 - val_loss: 2.2890 - val_acc: 0.5171
Epoch 10/10
414/414 [==============================] - 142s 342ms/step - loss: 0.2966 - acc: 0.9098 - val_loss: 2.0028 - val_acc: 0.5135

classifier.add(Conv2D(30, kernel_size=3, activation='relu',kernel_regularizer=regularizers.l2(0.001), input_shape=(128, 128, 3)))
classifier.add(Conv2D(30, kernel_size=3, activation='relu',kernel_regularizer=regularizers.l2(0.001),))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.5))

classifier.add(Conv2D(70, kernel_size=3, activation='relu',kernel_regularizer=regularizers.l2(0.001),))
classifier.add(Conv2D(70, kernel_size=3, activation='relu',kernel_regularizer=regularizers.l2(0.001),))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.5))

classifier.add(Flatten())
classifier.add(Dense(300, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(300, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(300, activation='relu'))
classifier.add(Dense(units=24, activation='softmax'))

Epoch 1/10
414/414 [==============================] - 146s 353ms/step - loss: 2.3623 - acc: 0.3307 - val_loss: 2.2583 - val_acc: 0.3801
Epoch 2/10
414/414 [==============================] - 143s 346ms/step - loss: 1.2547 - acc: 0.6337 - val_loss: 2.1087 - val_acc: 0.4747
Epoch 3/10
414/414 [==============================] - 142s 343ms/step - loss: 0.9097 - acc: 0.7495 - val_loss: 2.0602 - val_acc: 0.4715
Epoch 4/10
414/414 [==============================] - 142s 343ms/step - loss: 0.7306 - acc: 0.8091 - val_loss: 2.5100 - val_acc: 0.4266
Epoch 5/10
414/414 [==============================] - 142s 343ms/step - loss: 0.6273 - acc: 0.8405 - val_loss: 2.1778 - val_acc: 0.5041
Epoch 6/10
414/414 [==============================] - 142s 342ms/step - loss: 0.5389 - acc: 0.8655 - val_loss: 3.0425 - val_acc: 0.4523
Epoch 7/10
414/414 [==============================] - 142s 342ms/step - loss: 0.4902 - acc: 0.8804 - val_loss: 2.2100 - val_acc: 0.5324
Epoch 8/10
414/414 [==============================] - 142s 342ms/step - loss: 0.4455 - acc: 0.8935 - val_loss: 2.5455 - val_acc: 0.4660
Epoch 9/10
414/414 [==============================] - 143s 346ms/step - loss: 0.4415 - acc: 0.8949 - val_loss: 2.6912 - val_acc: 0.4703
Epoch 10/10
414/414 [==============================] - 142s 344ms/step - loss: 0.4084 - acc: 0.9042 - val_loss: 2.5517 - val_acc: 0.4798

classifier.add(Dropout(0.7))

Epoch 1/10
414/414 [==============================] - 146s 352ms/step - loss: 3.4728 - acc: 0.0809 - val_loss: 3.3116 - val_acc: 0.0600
Epoch 2/10
414/414 [==============================] - 142s 343ms/step - loss: 2.9794 - acc: 0.1491 - val_loss: 2.8673 - val_acc: 0.1669
Epoch 3/10
414/414 [==============================] - 141s 341ms/step - loss: 2.5272 - acc: 0.2410 - val_loss: 2.6622 - val_acc: 0.2196
Epoch 4/10
414/414 [==============================] - 141s 341ms/step - loss: 2.1902 - acc: 0.3129 - val_loss: 2.4034 - val_acc: 0.3021
Epoch 5/10
414/414 [==============================] - 142s 343ms/step - loss: 1.9428 - acc: 0.3797 - val_loss: 2.4193 - val_acc: 0.3260
Epoch 6/10
414/414 [==============================] - 141s 341ms/step - loss: 1.7374 - acc: 0.4396 - val_loss: 2.4596 - val_acc: 0.3400
Epoch 7/10
414/414 [==============================] - 143s 344ms/step - loss: 1.6091 - acc: 0.4843 - val_loss: 2.2896 - val_acc: 0.3627
Epoch 8/10
414/414 [==============================] - 141s 341ms/step - loss: 1.4848 - acc: 0.5172 - val_loss: 2.8227 - val_acc: 0.3669


classifier.add(Conv2D(30, kernel_size=3, activation='relu',kernel_regularizer=regularizers.l2(0.001), input_shape=(128, 128, 3)))
classifier.add(Conv2D(30, kernel_size=3, activation='relu',kernel_regularizer=regularizers.l2(0.001),))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.7))

classifier.add(Conv2D(70, kernel_size=3, activation='relu',kernel_regularizer=regularizers.l2(0.001),))
classifier.add(Conv2D(70, kernel_size=3, activation='relu',kernel_regularizer=regularizers.l2(0.001),))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.7))

classifier.add(Conv2D(140, kernel_size=3, activation='relu',kernel_regularizer=regularizers.l2(0.001),))
classifier.add(Conv2D(140, kernel_size=3, activation='relu',kernel_regularizer=regularizers.l2(0.001),))
classifier.add(Conv2D(140, kernel_size=3, activation='relu',kernel_regularizer=regularizers.l2(0.001),))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.7))

classifier.add(Flatten())
classifier.add(Dense(300, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
classifier.add(Dropout(0.6))
classifier.add(Dense(300, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
classifier.add(Dropout(0.6))
classifier.add(Dense(300, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
classifier.add(Dense(units=24, activation='softmax'))
414/414 [==============================] - 211s 509ms/step - loss: 4.6138 - acc: 0.1919 - val_loss: 5.9296 - val_acc: 0.1158
Epoch 2/300
414/414 [==============================] - 141s 341ms/step - loss: 3.5288 - acc: 0.4361 - val_loss: 4.5184 - val_acc: 0.2581
Epoch 3/300
414/414 [==============================] - 140s 338ms/step - loss: 2.9915 - acc: 0.5832 - val_loss: 4.4998 - val_acc: 0.3611
Epoch 4/300
414/414 [==============================] - 140s 338ms/step - loss: 2.6366 - acc: 0.6692 - val_loss: 3.7298 - val_acc: 0.4553
Epoch 5/300
414/414 [==============================] - 139s 337ms/step - loss: 2.3991 - acc: 0.7218 - val_loss: 3.9776 - val_acc: 0.4090
Epoch 6/300
414/414 [==============================] - 140s 338ms/step - loss: 2.2613 - acc: 0.7601 - val_loss: 3.4874 - val_acc: 0.4652
Epoch 7/300
414/414 [==============================] - 139s 336ms/step - loss: 2.1179 - acc: 0.7874 - val_loss: 3.3978 - val_acc: 0.4637
Epoch 8/300
414/414 [==============================] - 139s 337ms/step - loss: 2.1108 - acc: 0.8016 - val_loss: 3.3909 - val_acc: 0.5256
Epoch 9/300
414/414 [==============================] - 139s 336ms/step - loss: 2.0312 - acc: 0.8200 - val_loss: 3.1431 - val_acc: 0.5507
Epoch 10/300
414/414 [==============================] - 140s 338ms/step - loss: 2.1441 - acc: 0.8286 - val_loss: 3.4172 - val_acc: 0.5039
Epoch 11/300
414/414 [==============================] - 139s 335ms/step - loss: 2.0582 - acc: 0.8377 - val_loss: 3.3124 - val_acc: 0.5421
Epoch 12/300
414/414 [==============================] - 139s 335ms/step - loss: 2.0364 - acc: 0.8498 - val_loss: 3.4294 - val_acc: 0.5169
Epoch 13/300
414/414 [==============================] - 140s 338ms/step - loss: 1.9949 - acc: 0.8556 - val_loss: 3.1932 - val_acc: 0.5551
Epoch 14/300
414/414 [==============================] - 140s 339ms/step - loss: 1.9632 - acc: 0.8566 - val_loss: 3.1392 - val_acc: 0.5506
Epoch 15/300
414/414 [==============================] - 140s 338ms/step - loss: 1.9524 - acc: 0.8628 - val_loss: 3.3830 - val_acc: 0.4726
Epoch 16/300
414/414 [==============================] - 140s 339ms/step - loss: 1.9190 - acc: 0.8685 - val_loss: 3.2722 - val_acc: 0.5405
Epoch 17/300
414/414 [==============================] - 140s 338ms/step - loss: 1.8964 - acc: 0.8725 - val_loss: 3.1230 - val_acc: 0.5821
Epoch 18/300
414/414 [==============================] - 141s 340ms/step - loss: 1.8811 - acc: 0.8754 - val_loss: 3.2367 - val_acc: 0.5418
Epoch 19/300
414/414 [==============================] - 141s 341ms/step - loss: 1.8397 - acc: 0.8788 - val_loss: 3.5053 - val_acc: 0.5286
Epoch 20/300
414/414 [==============================] - 142s 343ms/step - loss: 1.8175 - acc: 0.8814 - val_loss: 2.9108 - val_acc: 0.6013
Epoch 21/300
414/414 [==============================] - 140s 339ms/step - loss: 1.7802 - acc: 0.8887 - val_loss: 3.1924 - val_acc: 0.5403
Epoch 22/300
414/414 [==============================] - 139s 336ms/step - loss: 1.8099 - acc: 0.8852 - val_loss: 3.0455 - val_acc: 0.5719
Epoch 23/300
414/414 [==============================] - 139s 336ms/step - loss: 1.7561 - acc: 0.8927 - val_loss: 3.0964 - val_acc: 0.6153
Epoch 24/300
414/414 [==============================] - 139s 336ms/step - loss: 1.7912 - acc: 0.8874 - val_loss: 3.0712 - val_acc: 0.5849
Epoch 25/300
414/414 [==============================] - 139s 337ms/step - loss: 1.7587 - acc: 0.8965 - val_loss: 3.0586 - val_acc: 0.5891
Epoch 26/300
414/414 [==============================] - 140s 338ms/step - loss: 1.6950 - acc: 0.9024 - val_loss: 3.3262 - val_acc: 0.5574
Epoch 27/300
414/414 [==============================] - 1162s 3s/step - loss: 1.7615 - acc: 0.8917 - val_loss: 3.2835 - val_acc: 0.5363
Epoch 28/300
414/414 [==============================] - 142s 342ms/step - loss: 1.7082 - acc: 0.9018 - val_loss: 3.1133 - val_acc: 0.5786
Epoch 29/300
414/414 [==============================] - 141s 341ms/step - loss: 1.6870 - acc: 0.9022 - val_loss: 3.0975 - val_acc: 0.5379
Epoch 30/300
414/414 [==============================] - 142s 342ms/step - loss: 1.7021 - acc: 0.9009 - val_loss: 3.2845 - val_acc: 0.5752
Epoch 31/300
414/414 [==============================] - 141s 340ms/step - loss: 1.6636 - acc: 0.9053 - val_loss: 2.7417 - val_acc: 0.6209
Epoch 32/300
414/414 [==============================] - 141s 341ms/step - loss: 1.6702 - acc: 0.9036 - val_loss: 3.0988 - val_acc: 0.5820
Epoch 33/300
414/414 [==============================] - 142s 343ms/step - loss: 1.6550 - acc: 0.9078 - val_loss: 3.2543 - val_acc: 0.5541
Epoch 34/300
414/414 [==============================] - 139s 337ms/step - loss: 1.6741 - acc: 0.9071 - val_loss: 3.0751 - val_acc: 0.5866
Epoch 35/300
414/414 [==============================] - 140s 338ms/step - loss: 1.6751 - acc: 0.9109 - val_loss: 2.9417 - val_acc: 0.5954
Epoch 36/300
414/414 [==============================] - 139s 336ms/step - loss: 1.6440 - acc: 0.9113 - val_loss: 2.9299 - val_acc: 0.5980
Epoch 37/300
414/414 [==============================] - 141s 341ms/step - loss: 1.6057 - acc: 0.9139 - val_loss: 3.1132 - val_acc: 0.5515
Epoch 38/300
414/414 [==============================] - 142s 342ms/step - loss: 1.6649 - acc: 0.9083 - val_loss: 3.8694 - val_acc: 0.4999
Epoch 39/300
414/414 [==============================] - 140s 338ms/step - loss: 1.6334 - acc: 0.9110 - val_loss: 3.3064 - val_acc: 0.5310
Epoch 40/300
414/414 [==============================] - 140s 337ms/step - loss: 1.6168 - acc: 0.9136 - val_loss: 3.2238 - val_acc: 0.5545
Epoch 41/300
414/414 [==============================] - 140s 337ms/step - loss: 1.6016 - acc: 0.9144 - val_loss: 3.0154 - val_acc: 0.5924
Epoch 42/300
414/414 [==============================] - 140s 337ms/step - loss: 1.6086 - acc: 0.9135 - val_loss: 2.9382 - val_acc: 0.6176
Epoch 43/300
414/414 [==============================] - 139s 336ms/step - loss: 1.5733 - acc: 0.9208 - val_loss: 2.9948 - val_acc: 0.5885
Epoch 44/300
414/414 [==============================] - 140s 337ms/step - loss: 1.6143 - acc: 0.9157 - val_loss: 3.0566 - val_acc: 0.5733
Epoch 45/300
414/414 [==============================] - 563s 1s/step - loss: 1.5886 - acc: 0.9182 - val_loss: 2.8841 - val_acc: 0.5916
Epoch 46/300
414/414 [==============================] - 142s 344ms/step - loss: 1.5749 - acc: 0.9198 - val_loss: 3.4661 - val_acc: 0.5242
Epoch 47/300
414/414 [==============================] - 142s 343ms/step - loss: 1.5804 - acc: 0.9199 - val_loss: 3.6096 - val_acc: 0.5311
Epoch 48/300
414/414 [==============================] - 140s 338ms/step - loss: 1.6013 - acc: 0.9171 - val_loss: 3.1383 - val_acc: 0.5743
Epoch 49/300
414/414 [==============================] - 140s 338ms/step - loss: 1.5871 - acc: 0.9189 - val_loss: 3.0839 - val_acc: 0.6231
Epoch 50/300
414/414 [==============================] - 140s 337ms/step - loss: 1.5521 - acc: 0.9199 - val_loss: 2.7977 - val_acc: 0.6289
Epoch 51/300
414/414 [==============================] - 139s 337ms/step - loss: 1.5874 - acc: 0.9171 - val_loss: 3.0187 - val_acc: 0.5787
Epoch 52/300
414/414 [==============================] - 139s 336ms/step - loss: 1.5793 - acc: 0.9229 - val_loss: 2.9451 - val_acc: 0.6257
Epoch 53/300
414/414 [==============================] - 140s 337ms/step - loss: 1.5798 - acc: 0.9208 - val_loss: 2.9610 - val_acc: 0.5975
Epoch 54/300
414/414 [==============================] - 139s 336ms/step - loss: 1.5763 - acc: 0.9216 - val_loss: 3.1735 - val_acc: 0.5819
Epoch 55/300
414/414 [==============================] - 139s 335ms/step - loss: 1.5630 - acc: 0.9227 - val_loss: 3.0385 - val_acc: 0.5623
Epoch 56/300
414/414 [==============================] - 140s 338ms/step - loss: 1.5496 - acc: 0.9217 - val_loss: 3.1112 - val_acc: 0.5844
Epoch 57/300
414/414 [==============================] - 139s 336ms/step - loss: 1.5538 - acc: 0.9226 - val_loss: 3.2905 - val_acc: 0.5606
Epoch 58/300
414/414 [==============================] - 139s 336ms/step - loss: 1.5311 - acc: 0.9243 - val_loss: 2.9521 - val_acc: 0.6099
Epoch 59/300
414/414 [==============================] - 140s 338ms/step - loss: 1.5721 - acc: 0.9210 - val_loss: 3.0782 - val_acc: 0.5614
Epoch 60/300
414/414 [==============================] - 140s 338ms/step - loss: 1.5580 - acc: 0.9254 - val_loss: 3.1046 - val_acc: 0.5810
Epoch 61/300
414/414 [==============================] - 140s 338ms/step - loss: 1.5756 - acc: 0.9249 - val_loss: 2.6868 - val_acc: 0.6631
Epoch 62/300
414/414 [==============================] - 140s 338ms/step - loss: 1.5553 - acc: 0.9244 - val_loss: 2.8637 - val_acc: 0.6211
Epoch 63/300
414/414 [==============================] - 140s 338ms/step - loss: 1.5346 - acc: 0.9264 - val_loss: 3.0605 - val_acc: 0.5857
Epoch 64/300
414/414 [==============================] - 139s 336ms/step - loss: 1.5490 - acc: 0.9254 - val_loss: 2.8766 - val_acc: 0.6113
Epoch 65/300
414/414 [==============================] - 139s 336ms/step - loss: 1.5222 - acc: 0.9281 - val_loss: 3.0944 - val_acc: 0.5878
Epoch 66/300
414/414 [==============================] - 140s 337ms/step - loss: 1.5081 - acc: 0.9266 - val_loss: 2.9557 - val_acc: 0.6142
Epoch 67/300
414/414 [==============================] - 141s 340ms/step - loss: 1.5400 - acc: 0.9255 - val_loss: 2.9217 - val_acc: 0.6165
Epoch 68/300
414/414 [==============================] - 140s 339ms/step - loss: 1.4884 - acc: 0.9270 - val_loss: 3.3907 - val_acc: 0.5133
Epoch 69/300
414/414 [==============================] - 139s 337ms/step - loss: 1.5189 - acc: 0.9280 - val_loss: 3.1833 - val_acc: 0.5767
Epoch 70/300
414/414 [==============================] - 140s 337ms/step - loss: 1.5419 - acc: 0.9271 - val_loss: 3.0966 - val_acc: 0.5897
Epoch 71/300
414/414 [==============================] - 140s 338ms/step - loss: 1.5212 - acc: 0.9273 - val_loss: 2.9346 - val_acc: 0.6061
Epoch 72/300
414/414 [==============================] - 139s 336ms/step - loss: 1.5245 - acc: 0.9264 - val_loss: 3.0467 - val_acc: 0.6106
Epoch 73/300
414/414 [==============================] - 139s 336ms/step - loss: 1.5082 - acc: 0.9284 - val_loss: 2.9095 - val_acc: 0.6214
Epoch 74/300
414/414 [==============================] - 139s 337ms/step - loss: 1.5044 - acc: 0.9282 - val_loss: 2.8697 - val_acc: 0.6198
Epoch 75/300
414/414 [==============================] - 139s 336ms/step - loss: 1.5057 - acc: 0.9270 - val_loss: 3.1475 - val_acc: 0.6054
Epoch 76/300
414/414 [==============================] - 140s 338ms/step - loss: 1.5199 - acc: 0.9291 - val_loss: 2.9045 - val_acc: 0.5910
Epoch 77/300
414/414 [==============================] - 139s 336ms/step - loss: 1.4741 - acc: 0.9298 - val_loss: 3.2027 - val_acc: 0.5947
Epoch 78/300
414/414 [==============================] - 139s 336ms/step - loss: 1.4961 - acc: 0.9281 - val_loss: 3.5258 - val_acc: 0.5521
Epoch 79/300
414/414 [==============================] - 139s 336ms/step - loss: 1.5377 - acc: 0.9260 - val_loss: 3.2931 - val_acc: 0.5476
Epoch 80/300
414/414 [==============================] - 139s 336ms/step - loss: 1.4957 - acc: 0.9309 - val_loss: 2.8463 - val_acc: 0.6322
Epoch 81/300
414/414 [==============================] - 139s 336ms/step - loss: 1.4661 - acc: 0.9313 - val_loss: 3.1109 - val_acc: 0.5820
Epoch 82/300
414/414 [==============================] - 140s 337ms/step - loss: 1.5013 - acc: 0.9313 - val_loss: 2.9062 - val_acc: 0.6089
Epoch 83/300
414/414 [==============================] - 139s 336ms/step - loss: 1.4605 - acc: 0.9314 - val_loss: 2.9469 - val_acc: 0.5877
Epoch 84/300
414/414 [==============================] - 140s 339ms/step - loss: 1.5038 - acc: 0.9299 - val_loss: 2.8946 - val_acc: 0.6171
Epoch 85/300
414/414 [==============================] - 140s 338ms/step - loss: 1.4875 - acc: 0.9336 - val_loss: 3.2873 - val_acc: 0.5718
Epoch 86/300
414/414 [==============================] - 140s 339ms/step - loss: 1.5484 - acc: 0.9294 - val_loss: 3.0291 - val_acc: 0.5996
Epoch 87/300
414/414 [==============================] - 140s 339ms/step - loss: 1.4988 - acc: 0.9310 - val_loss: 3.0219 - val_acc: 0.6211
Epoch 88/300
414/414 [==============================] - 140s 337ms/step - loss: 1.4976 - acc: 0.9306 - val_loss: 3.1387 - val_acc: 0.5746
Epoch 89/300
414/414 [==============================] - 139s 336ms/step - loss: 1.5138 - acc: 0.9327 - val_loss: 3.0323 - val_acc: 0.5812
Epoch 90/300
414/414 [==============================] - 139s 335ms/step - loss: 1.4896 - acc: 0.9324 - val_loss: 2.7716 - val_acc: 0.6693
Epoch 91/300
414/414 [==============================] - 139s 335ms/step - loss: 1.4997 - acc: 0.9311 - val_loss: 2.9183 - val_acc: 0.6294
Epoch 92/300
414/414 [==============================] - 138s 334ms/step - loss: 1.4852 - acc: 0.9339 - val_loss: 3.0000 - val_acc: 0.5974
Epoch 93/300
414/414 [==============================] - 139s 335ms/step - loss: 1.5101 - acc: 0.9316 - val_loss: 2.9578 - val_acc: 0.6106
Epoch 94/300
414/414 [==============================] - 139s 337ms/step - loss: 1.4764 - acc: 0.9319 - val_loss: 2.9781 - val_acc: 0.6372
Epoch 95/300
414/414 [==============================] - 139s 335ms/step - loss: 1.4825 - acc: 0.9328 - val_loss: 3.2178 - val_acc: 0.5521
Epoch 96/300
414/414 [==============================] - 139s 336ms/step - loss: 1.4925 - acc: 0.9311 - val_loss: 3.1940 - val_acc: 0.5700
Epoch 97/300
414/414 [==============================] - 139s 336ms/step - loss: 1.5307 - acc: 0.9287 - val_loss: 2.9466 - val_acc: 0.6179
Epoch 98/300
414/414 [==============================] - 139s 335ms/step - loss: 1.4806 - acc: 0.9358 - val_loss: 3.0145 - val_acc: 0.6146
Epoch 99/300
414/414 [==============================] - 140s 338ms/step - loss: 1.4993 - acc: 0.9311 - val_loss: 2.8332 - val_acc: 0.6530
Epoch 100/300
414/414 [==============================] - 140s 338ms/step - loss: 1.5251 - acc: 0.9323 - val_loss: 3.2363 - val_acc: 0.6051
Epoch 101/300
414/414 [==============================] - 140s 338ms/step - loss: 1.4796 - acc: 0.9318 - val_loss: 3.0842 - val_acc: 0.6207
Epoch 102/300
414/414 [==============================] - 140s 338ms/step - loss: 1.4534 - acc: 0.9339 - val_loss: 3.1417 - val_acc: 0.6054
Epoch 103/300
414/414 [==============================] - 140s 338ms/step - loss: 1.4725 - acc: 0.9337 - val_loss: 3.1573 - val_acc: 0.6006
Epoch 104/300
414/414 [==============================] - 139s 336ms/step - loss: 1.4831 - acc: 0.9307 - val_loss: 2.7052 - val_acc: 0.6582
Epoch 105/300
414/414 [==============================] - 139s 336ms/step - loss: 1.4906 - acc: 0.9321 - val_loss: 2.9510 - val_acc: 0.6338
Epoch 106/300
414/414 [==============================] - 139s 335ms/step - loss: 1.5256 - acc: 0.9287 - val_loss: 2.9529 - val_acc: 0.6096
Epoch 107/300
414/414 [==============================] - 139s 335ms/step - loss: 1.4432 - acc: 0.9363 - val_loss: 2.9173 - val_acc: 0.6048
Epoch 108/300
414/414 [==============================] - 139s 335ms/step - loss: 1.4577 - acc: 0.9344 - val_loss: 3.1198 - val_acc: 0.5842
Epoch 109/300
414/414 [==============================] - 139s 336ms/step - loss: 1.4878 - acc: 0.9333 - val_loss: 3.0383 - val_acc: 0.5939
Epoch 110/300
414/414 [==============================] - 139s 335ms/step - loss: 1.4500 - acc: 0.9367 - val_loss: 2.8501 - val_acc: 0.6518
Epoch 111/300
414/414 [==============================] - 139s 335ms/step - loss: 1.4486 - acc: 0.9335 - val_loss: 3.2165 - val_acc: 0.5655
Epoch 112/300
414/414 [==============================] - 140s 339ms/step - loss: 1.4255 - acc: 0.9362 - val_loss: 3.2801 - val_acc: 0.5469
Epoch 113/300
414/414 [==============================] - 140s 337ms/step - loss: 1.4617 - acc: 0.9348 - val_loss: 2.8423 - val_acc: 0.6251
Epoch 114/300
414/414 [==============================] - 140s 339ms/step - loss: 1.4393 - acc: 0.9370 - val_loss: 3.0031 - val_acc: 0.6178
Epoch 115/300
414/414 [==============================] - 140s 338ms/step - loss: 1.4546 - acc: 0.9337 - val_loss: 2.9611 - val_acc: 0.6074
Epoch 116/300
414/414 [==============================] - 140s 339ms/step - loss: 1.4950 - acc: 0.9305 - val_loss: 3.0758 - val_acc: 0.6045
Epoch 117/300
414/414 [==============================] - 139s 335ms/step - loss: 1.4730 - acc: 0.9341 - val_loss: 3.0919 - val_acc: 0.5879
Epoch 118/300
414/414 [==============================] - 139s 336ms/step - loss: 1.4569 - acc: 0.9363 - val_loss: 2.9073 - val_acc: 0.6287
Epoch 119/300
414/414 [==============================] - 139s 336ms/step - loss: 1.4404 - acc: 0.9372 - val_loss: 2.9240 - val_acc: 0.6375
Epoch 120/300
414/414 [==============================] - 139s 336ms/step - loss: 1.4704 - acc: 0.9359 - val_loss: 2.9269 - val_acc: 0.5932
Epoch 121/300
414/414 [==============================] - 139s 335ms/step - loss: 1.4432 - acc: 0.9346 - val_loss: 3.0706 - val_acc: 0.5963
Epoch 122/300
414/414 [==============================] - 139s 335ms/step - loss: 1.4866 - acc: 0.9339 - val_loss: 3.0892 - val_acc: 0.5797
Epoch 123/300
414/414 [==============================] - 139s 337ms/step - loss: 1.4348 - acc: 0.9368 - val_loss: 3.0808 - val_acc: 0.5943
Epoch 124/300
414/414 [==============================] - 140s 339ms/step - loss: 1.4575 - acc: 0.9360 - val_loss: 2.9136 - val_acc: 0.6347
Epoch 125/300
414/414 [==============================] - 140s 338ms/step - loss: 1.4756 - acc: 0.9351 - val_loss: 3.3635 - val_acc: 0.5877
Epoch 126/300
414/414 [==============================] - 140s 338ms/step - loss: 1.4632 - acc: 0.9347 - val_loss: 2.7377 - val_acc: 0.6498
Epoch 127/300
414/414 [==============================] - 140s 339ms/step - loss: 1.4852 - acc: 0.9334 - val_loss: 3.2842 - val_acc: 0.5719
Epoch 128/300
414/414 [==============================] - 139s 337ms/step - loss: 1.4448 - acc: 0.9370 - val_loss: 3.1850 - val_acc: 0.6118
Epoch 129/300
414/414 [==============================] - 139s 336ms/step - loss: 1.4462 - acc: 0.9376 - val_loss: 2.9335 - val_acc: 0.6107
Epoch 130/300
414/414 [==============================] - 144s 348ms/step - loss: 1.4561 - acc: 0.9353 - val_loss: 3.3377 - val_acc: 0.5767
Epoch 131/300
414/414 [==============================] - 148s 358ms/step - loss: 1.4903 - acc: 0.9350 - val_loss: 3.0415 - val_acc: 0.6127
Epoch 132/300
414/414 [==============================] - 143s 344ms/step - loss: 1.4838 - acc: 0.9357 - val_loss: 2.8329 - val_acc: 0.6540
Epoch 133/300
414/414 [==============================] - 143s 345ms/step - loss: 1.4519 - acc: 0.9341 - val_loss: 3.0918 - val_acc: 0.5893
Epoch 134/300
414/414 [==============================] - 142s 343ms/step - loss: 1.5163 - acc: 0.9338 - val_loss: 2.7136 - val_acc: 0.6663