from dependencies import *


# df_train_data=pd.read_csv("D:\\covid\\train_data.csv")
# df_test_data=pd.read_csv("D:\\covid\\test_data.csv")

# print(df_train_data["Image_ID"][0])

def image_preprocessing1(path_1,path_2,path_3,path_4,data_frame):
    image_array=[]
    print("testing")   
    
    for img in range(len(data_frame)):

        if list(data_frame["Label"])[img]==2:

            path=path_1
        elif list(data_frame["Label"])[img]==3:

            path=path_2
        elif list(data_frame["Label"])[img]==0:

            path=path_3
        else:
            path=path_4

        img_path_match=os.path.join(path,list(data_frame["Image_ID"])[img])
        extract_img=cv2.imread(img_path_match)
        print(extract_img.shape)
        print("b")
        img_resize=cv2.resize(extract_img,(224,224))

        image_array.append(img_resize)

    return np.array(image_array)


# train_image=image_preprocessing("D:\\covid\\COVID_CT_All","D:\\covid\\CT_NonCOVID\\CT_NonCOVID",df_train_data)
# test_image=image_preprocessing("D:\\covid\\COVID_CT_All","D:\\covid\\CT_NonCOVID\\CT_NonCOVID",df_test_data)
# print(train_image.shape)

# print(test_image.shape)
