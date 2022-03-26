import faiss
import pickle
import os
import numpy as np
import random



def load_lfw(train_embeddings_path, test_embeddings_path, lfw_full_list_path):
    with open(train_embeddings_path, 'rb') as train_f:
        train_vectors_dict = pickle.load(train_f)

    with open(test_embeddings_path, 'rb') as test_f:
        test_vectors_dict = pickle.load(test_f)
        
    with open(lfw_full_list_path, 'r') as f_l:
        people_images = f_l.readlines()

    person_class = {}
    for person_image in people_images:
        tmp = person_image.split(' ')
        person_class[tmp[0]] = int(tmp[1])

    train_vectors = np.array(list(train_vectors_dict.values()))
    train_vectors_info = np.array(list(train_vectors_dict.keys()))

    test_vectors = np.array(list(test_vectors_dict.values()))
    test_vectors_info = np.array(list(test_vectors_dict.keys()))
 
    train_vectors_labels = []
    test_vectors_labels = []

    for i in range(train_vectors_info.shape[0]):
        train_vectors_labels.append(person_class[train_vectors_info[i]])

    for j in range(test_vectors_info.shape[0]):
        test_vectors_labels.append(person_class[test_vectors_info[j]])

    train_vectors_labels = np.array(train_vectors_labels)
    test_vectors_labels = np.array(test_vectors_labels)

    faiss.normalize_L2(train_vectors.reshape(1, -1))
    faiss.normalize_L2(test_vectors.reshape(1, -1)) 

    return train_vectors, test_vectors, train_vectors_info, test_vectors_info

def load_data(embeddings_path, train_list_path, test_list_path):
    with open(embeddings_path, 'rb') as emb_file:
        full_dataset_embeddings_dict = pickle.load(emb_file)
    
    with open(train_list_path, 'r') as train_list_file:
        train_list = train_list_file.readlines()

    with open(test_list_path, 'r') as test_list_file:
        test_list = test_list_file.readlines()

    test_list = [test_item[:-1].split()[0] for test_item in test_list]
    train_list = [train_item[:-1].split()[0] for train_item in train_list]

    test_embeddings = {}
    train_embeddings = {}
    for test_item in test_list:
        test_embeddings[test_item] = full_dataset_embeddings_dict[test_item]

    for train_item in train_list:
        train_embeddings[train_item] = full_dataset_embeddings_dict[train_item]
    train_vectors = np.array(list(train_embeddings.values()))
    test_vectors = np.array(list(test_embeddings.values()))
    train_vectors_info = np.array(list(train_embeddings.keys()))
    test_vectors_info = np.array(list(test_embeddings.keys()))
    faiss.normalize_L2(train_vectors.reshape(1, -1))
    faiss.normalize_L2(test_vectors.reshape(1, -1))
    return train_vectors, test_vectors, train_vectors_info, test_vectors_info


def precision_top_n(I, test_vectors_data, train_vectors_data, n):
    assert n <= I.shape[1]
    assert I.shape[0] == test_vectors_data.shape[0]
    matchs = 0
    for i, test_vector_info in enumerate(test_vectors_data):
        test_vector_info = test_vector_info.split("/")
        test_vector_info_class = test_vector_info[0]
        neighbors_info = train_vectors_data[I[i]]
        for j, neighbor_info in enumerate(neighbors_info):
            neighbor_info = neighbor_info.split("/") 
            neighbors_info[j] = neighbor_info[0]
        if test_vector_info_class in neighbors_info:
            matchs += 1  
        
    precision = float(matchs / test_vectors_data.shape[0])
    return precision   



def get_memory(index):
    # write index to file
    faiss.write_index(index, './temp.index')
    # get file size
    file_size = os.path.getsize('./temp.index')
    # delete saved index
    os.remove('./temp.index')
    return file_size / 1e6

