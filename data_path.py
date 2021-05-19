import os 

DIR_TRAIN = "dataset/train/"
DIR_VALID = "dataset/valid/"
DIR_TEST  = "dataset/test/"

classes = os.listdir(DIR_TRAIN)
# print("Total classes: ", classes)

train_count = 0 
valid_count = 0 
test_count  = 0

for _class in classes:
    train_count += len(os.listdir(DIR_TRAIN + _class))
    valid_count += len(os.listdir(DIR_VALID + _class))
    test_count  += len(os.listdir(DIR_TEST  + _class))

print("Total train images: ", train_count)
print("Total valid images: ", valid_count)
print("Test images: ", test_count)

train_imgs = []
valid_imgs = []
test_imgs  = []

for _class in classes:
    for img in os.listdir(DIR_TRAIN + _class):
        train_imgs.append(DIR_TRAIN + _class + "/" + img)
    for img in os.listdir(DIR_VALID + _class):
        valid_imgs.append(DIR_VALID + _class + "/" + img)
    for img in os.listdir(DIR_TEST +_class ):
        test_imgs.append(DIR_TEST + _class +"/" + img)

class_to_int = {classes[i]: i for i in range(len(classes))}