import os
import random
import shutil

# need to keep this .ipynb file in the same directory as the images folder
# divede the images provided into training and validation set (8:2)
# create directories
def clone_data(image_path):
    if not os.path.exists(image_path):
        print('Path {} not exists'.format(image_path) )
        return
    print('Splitting dataset...')
    train_path = os.getcwd() + '/train/'
    val_path = os.getcwd() + '/val/'
    train_fake_path = train_path + 'fake/'
    train_real_path = train_path + 'real/'
    val_fake_path = val_path + 'fake/'
    val_real_path = val_path + 'real/'

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    if not os.path.exists(train_fake_path):
        os.makedirs(train_fake_path)
    if not os.path.exists(train_real_path):
        os.makedirs(train_real_path)
    if not os.path.exists(val_fake_path):
        os.makedirs(val_fake_path)
    if not os.path.exists(val_real_path):
        os.makedirs(val_real_path)

    # distribute 12000 images into different folders
    train_fake_num = 0
    train_real_num = 0

    val_fake_num = 0
    val_real_num = 0

    test_fake_num = 0
    test_real_num = 0

    # loop through images folder
    for rootpath, dirnames, filenames in os.walk(image_path):
        for dirname in dirnames:
            if dirname == 'fake_deepfake':
                print(' > fake_deepfake:')
                # generate 800 random number in the range [0, 3999] to represent those go to val
                # force pseudorandom split
                random.seed(4487)
                val_index = random.sample(range(0, 4000), 800)
                # directory full path
                image_folder = rootpath + dirname + '/'
                # loop all images in fake_deepfake folder
                imgfiles = os.listdir(image_folder)
                for imgfile in imgfiles:
                    srcpath = image_folder + imgfile
                    index = int(imgfile.split('.')[0])
                    if index in val_index:
                        newname = str(val_fake_num) + '.png'
                        dstpath = val_fake_path + newname
                        shutil.copyfile(srcpath, dstpath)
                        val_fake_num += 1
                    else:
                        newname = str(train_fake_num) + '.png'
                        dstpath = train_fake_path + newname
                        shutil.copyfile(srcpath, dstpath)
                        train_fake_num += 1
                print('done')
            elif dirname == 'fake_face2face':
                print(' > fake_face2face:')
                # generate 800 random number in the range [0, 3999] to represent those go to val
                # force pseudorandom split
                random.seed(4486)
                val_index = random.sample(range(0, 4000), 800)
                # directory full path
                image_folder = rootpath + dirname + '/'
                # loop all images in fake_face2face folder
                imgfiles = os.listdir(image_folder)
                for imgfile in imgfiles:
                    srcpath = image_folder + imgfile
                    index = int(imgfile.split('.')[0])
                    if index in val_index:
                        newname = str(val_fake_num) + '.png'
                        dstpath = val_fake_path + newname
                        shutil.copyfile(srcpath, dstpath)
                        val_fake_num += 1
                    else:
                        newname = str(train_fake_num) + '.png'
                        dstpath = train_fake_path + newname
                        shutil.copyfile(srcpath, dstpath)
                        train_fake_num += 1
                print('done')
            elif dirname == 'real':
                print(' > real:')
                # generate 800 random number in the range [0, 3999] to represent those go to val
                # force pseudorandom split
                random.seed(4485)
                val_index = random.sample(range(0, 4000), 800)
                # directory full path
                image_folder = rootpath + dirname + '/'
                # loop all images in real folder
                imgfiles = os.listdir(image_folder)
                for imgfile in imgfiles:
                    srcpath = image_folder + imgfile
                    index = int(imgfile.split('.')[0])
                    if index in val_index:
                        newname = str(val_real_num) + '.png'
                        dstpath = val_real_path + newname
                        shutil.copyfile(srcpath, dstpath)
                        val_real_num += 1
                    else:
                        newname = str(train_real_num) + '.png'
                        dstpath = train_real_path + newname
                        shutil.copyfile(srcpath, dstpath)
                        train_real_num += 1
                print('done')
    return train_path, val_path