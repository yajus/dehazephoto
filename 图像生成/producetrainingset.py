import matplotlib.pyplot as plt
from scipy import misc
import numpy as np
import math
import scipy
import os
#readphoto
indexnum=0
namen=os.listdir("./photo")
print(namen)
for filename in namen:
    indexnum += 1
    I = misc.imread('./photo/' + filename)
    print(filename)
    # print("1")
    # plt.imshow(I)
    # plt.show()
    I = np.array(I, dtype=float)/255  # 转换类型
    # print("2")
    # print(I)
    # profile
    patchsize = 16
    alpha = 0.5
    patchradius = patchsize / 2  # usless
    imgshape = I.shape
    cutbegin = 0
    cutend = min(imgshape[0], imgshape[1])  # 长宽较小值
    blockcount = (cutend - cutbegin) //patchsize # 块数
    t = np.random.rand(blockcount * blockcount, 1)
    print(blockcount)
    # test
    #print(cutend)
    #print(blockcount)
    # print(imgshape[0])
    # scipy.misc.imsave('./save1.png', I)
    # print(I.shape)

    k = 0  # t[k]

    # weidu=cutbegin-cutend
    # GTimg=np.zeros(weidu)

    # algrithm
    for i in range(cutbegin, blockcount*patchsize, patchsize):
        for j in range(cutbegin, blockcount*patchsize,patchsize):
            #  print('path:')
            #  print(j)
            imgpatch = I[i:i + patchsize, j:j + patchsize, :]
            #print(imgpatch.shape)
            # print(imgpatch)
            # GTimg[:,:,:,k]=imgpatch
            # print(imgpatch)
            hazepatch = imgpatch *t[k] + alpha * (1-t[k])
            # hazepatch=hazepatch*255
            # print(hazepatch)

            # hazepatch = imgpatch * 0.001 + alpha * (1 - 0.001)

            # hazepatch = imgpatch * t(k) + alpha * (1 - t(k))
            # GTimg1=np.append(GTimg1,hazepatch,axis=1)
            if (j == 0):
                GTimg1 = np.array(hazepatch)
            else:
                GTimg1 = np.append(GTimg1, hazepatch, axis=1)
            #scipy.misc.imsave(('./hazefree/img' + str(k) + '.bmp'), imgpatch)
            #scipy.misc.imsave(('./haze/img' + str(k) + '.bmp'), hazepatch)
            k = k + 1
        if (i == 0):
            GTimg = np.array(GTimg1)
        else:
            GTimg = np.append(GTimg, GTimg1, axis=0)


    # print('blockcount:',blockcount)
    # create block.npy
    # print("3")
    # plt.imshow(GTimg)
    # print(GTimg.shape)
    # plt.show()

    # #训练后图片
    plt.imsave('./trainphotodehaze/' + str(indexnum) + '.png', GTimg)
    test = misc.imread('./trainphotodehaze/' + str(indexnum) + '.png')
    # print("4")
    # plt.imshow(test)
    # print(test.shape)
    # plt.show()

    dehazeblock = GTimg.reshape([-1, 16, 16, 3])
    print("GTIMGshape:", GTimg.shape)




    # create originptoto
    originphoto = np.array(I[0:patchsize * blockcount, 0:patchsize * blockcount, :])
    # 保存：
    #训练前图片
    plt.imsave('./trainphotoorigin/' + str(indexnum) + '.png', originphoto)
    plt.imsave('./res/aaaa.png', originphoto)


    # saveall
    if (indexnum == 1):
        dehazeblock_all = np.array(dehazeblock)
        GTimg_all = np.array(originphoto.reshape([-1,16,16,3]))
        print('GTimg_allshape:',GTimg_all.shape)
        print('dehazeblockshape:', dehazeblock_all.shape)
        t_all = np.array(t.reshape([-1,1,1,1]))
    else:
        dehazeblock_all = np.append(dehazeblock_all, dehazeblock, axis=0)
        GTimg_all = np.append(GTimg_all, originphoto.reshape([-1,16,16,3]), axis=0)
        t_all = np.append(t_all, t.reshape([-1,1,1,1]), axis=0)
        print('shape:', GTimg_all.shape)
        print('dehazeblockshape:', dehazeblock_all.shape)



    # show
   # plt.imshow(GTimg)
    #plt.show()
    #plt.imshow(originphoto)
    #plt.show()


#保存
#雾图
#np.save('./originphoto.npy',GTimg_all)
np.save('./originphoto16.npy',GTimg_all)
np.save('./dehazephoto16.npy',dehazeblock_all)
np.save('./t_value.npy',t_all)



