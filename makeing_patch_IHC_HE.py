from openslide import OpenSlide
import cv2
def object_region(img):
    arry_img = np.array(img)
    arry_img = arry_img[:,:,:3]
    arry_sum=arry_img.cumsum()
    # raw데이터가 깨졌을경우 0,0,0으로 되기때문에 cutoff시킴
    arry_img[:,:,0][arry_img[:,:,0]==0]=255
    arry_img[:,:,1][arry_img[:,:,1]==0]=255
    arry_img[:,:,2][arry_img[:,:,2]==0]=255
    arry_img[:,:,0][arry_img[:,:,0]<85]=1 
    arry_img[:,:,1][arry_img[:,:,1]<85]=1
    arry_img[:,:,2][arry_img[:,:,2]<100]=1
    # arry_img[:,:,0] 에다 모든 mask 정보가 들어감.
    arry_img[:,:,0][arry_img[:,:,1]==1]=1
    arry_img[:,:,0][arry_img[:,:,2]==1]=1
    arry_img[arry_img!=1]=0
    arry_img1=np.zeros_like(arry_img)
    arry_img1[:,:,0]=arry_img[:,:,0]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    step1 = cv2.dilate(arry_img1,kernel,iterations = 2)
    step2 = cv2.erode(step1,kernel,iterations = 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
    step3 = cv2.dilate(step1,kernel,iterations = 2)
    return step3
    
def wsi_tiling(File, IHC_File, dest_imagePath, img_name, Tile_size, calibration_coord, tumor_mask = None):
    since = time.time()
    Slide = OpenSlide(File)
    IHC_Slide = OpenSlide(IHC_File)
    
    xr = float(Slide.properties['openslide.mpp-x'])
    yr = float(Slide.properties['openslide.mpp-y'])
    Stride = [round(Tile_size[0]/xr), round(Tile_size[1]/yr)]
    Dims = Slide.level_dimensions
    X = np.arange(0,Dims[0][0] + 1, Stride[0])
    Y = np.arrang(0,Dims[0][1] + 1, Stride[1])
    X, Y = np.meshgrid(X,Y)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    for i in range(150,X.shape[0] - 1):
        for j in range(X.shape[1] - 1):
            # original
            Tile = Slide.read_region((int(X[i,j]), int(Y[i,j])), 0, (Stride[0],Stride[1]))
            Tile = np.asarray(Tile)
            Tile = Tile[:,:,:3]
            # masked
            IHC_Tile = IHC_Slide.read_region((int(X[i,j]) - int(calibration_coord[0]), int(Y[i,j]) - int(calibration_coord[1])), 0, (Stride[0],Stride[1]))
            IHC_Tile = np.asarray(IHC_Tile)
            IHC_Tile = object_region(IHC_Tile)
            # 정보가 없는 patch 제거
            IHC_cutoff = IHC_Tile[:,:,0].cumsum()[-1]/len(IHC_Tile[:,:,0].cumsum())
            bn = np.sum(Tile[:,:,0] < 5) + np.sum(np.mean(Tile,axis = 2) > 245)
            if (np.std(Tile[:,:,0]) + np.std(Tile[:,:,1]) + np.std(Tile[:,:,2]))/3 > 18 and bn < Stride[0] * Stride[1] * 0.3 and IHC_cutoff > 0.05:
                tile_name = img_name.split('.')[0] + '_' +str(X[i,j]) + '_' + str(Y[i,j]) + '_' + str(Stride[0]) + '_' + str(Stride[1]) + '_' + '.png'
                IHC_tile_name = img_name.split('.')[0] + '_' +str(X[i,j]) + '_' + str(Y[i,j]) + '_' + str(Stride[0]) + '_' + str(Stride[1]) + '_' + 'IHC.png'
                img = Image.fromarray(Tile)
                img_masked = Image.fromarray(IHC_Tile)
                img.save(dest_imagePath +'/original/' + tile_name)
                img_masked.save(dest_imagePath +'/mask/' + tile_name)
    time_elapsed = time.time() - since
    print('Patch production Completed {:.0f}m {:.0f}s'.format(time_elapsed//60,time_elapsed%60))
                
        
               
