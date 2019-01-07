from model import *
from data import *
from keras.models import load_model


#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 忽略硬件加速的警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2,'data/Data/data01','image','label',data_gen_args,save_to_dir = None)

model = unet()
model_checkpoint = ModelCheckpoint('unet_hw.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint])


model = load_model('model-tgs-salt.h5')
f_names = glob.glob('data/Data/data02/image/*.png')
testGene = testGenerator("data/Data/data02/image/", len(f_names))
results = model.predict_generator(testGene,len(f_names),verbose=1)
saveResult("data/Data/data02/predict/",results)