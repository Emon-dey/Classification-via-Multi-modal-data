from dependencies import *



# df_train_data=pd.read_csv("D:\\covid\\train_data.csv")
# df_test_data=pd.read_csv("D:\\covid\\test_data.csv")


def one_hot_encode(ref_label,label):
    
    ref_label=np.array(ref_label)
    label=np.array(label)

    onehot_encode=OneHotEncoder(sparse=False)
    interger_encoded=ref_label.reshape(len(ref_label),1)
    interger_encoded1=label.reshape(len(label),1)
    # interger_encoded1=onehot_encode.fit(interger_encoded)

    onehot_encoded=onehot_encode.fit(interger_encoded)

    onehot_encoded1=onehot_encode.transform(interger_encoded1)

    return onehot_encoded1


# one_hot_label_train=one_hot_encode(list(df_train_data["Label"]),list(df_train_data["Label"]))

# one_hot_label_test=one_hot_encode(list(df_train_data["Label"]),list(df_test_data["Label"]))

# print(one_hot_label_train.shape)

# print(one_hot_label_test.shape)
