
import faiss
import time
import pickle
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import load_data, precision_top_n, get_memory, load_lfw
import random



# class to make some compression-related experiments

class FaceEmbeddingsCompression:

    def __init__(self, embeddings_dimension:int, embeddings_path:str, 
                 train_list_path:str, test_list_path:str):
        self.embeddings_path = embeddings_path
        self.train_list_path = train_list_path
        self.test_list_path = test_list_path
        self.emb_dim = embeddings_dimension
        self.train_vectors = None
        self.train_vectors_info = None
        self.test_vectors = None
        self.test_vectors_info = None
        self.n_bits_values = list(range(1, 9, 1))
        self.n_subquantizers_values = [i for i in range(2, self.emb_dim) if self.emb_dim % i == 0]
        self.n_list_values = [i for i in range(20, 301, 20)]


    def set_parameter_values(self, parameter_name:str, values:list):
        if parameter_name == "n_bits":
            self.n_list_values = values    
        elif parameter_name == "n_subquantizers":
            self.n_subquantizers_values = values    
        elif parameter_name == "n_lists":
            self.n_list_values = values
        else:
            print("Parameter name is not recognized!")
            exit()

        
    
    def get_parameter_values(self, parameter_name:str):
        result = {}
        if parameter_name == "n_bits":
            result[parameter_name] = self.n_bits_values
        elif parameter_name == "n_subquantizers":
            result[parameter_name] = self.n_subquantizers_values
        elif parameter_name == "n_lists":
            result[parameter_name] = self.n_list_values
        elif parameter_name == "all":
            result.update(self.get_parameter_values("n_bits"))
            result.update(self.get_parameter_values("n_subquantizers"))
            result.update(self.get_parameter_values("n_lists"))
        else:
            print("Parameter name is not recognized!")
        
        return result
        


    def load_embeddings(self):
        self.train_vectors, self.test_vectors, self.train_vectors_info, self.test_vectors_info  = load_data(self.embeddings_path,
                                                                                                            self.train_list_path,
                                                                                                            self.test_list_path)

    def load_lfw(self, train_embeddings_path, test_embeddings_path, lfw_full_list_path):
        self.train_vectors, self.test_vectors, self.train_vectors_info, self.test_vectors_info  = load_lfw(train_embeddings_path,
                                                                                                           test_embeddings_path,
                                                                                                           lfw_full_list_path)


    def index_without_compression(self, nn_quantity, show_index_info=False):
        index = faiss.IndexFlatIP(self.emb_dim)
        index.add(self.train_vectors)
        search_start_time = time.time()
        D, I  = index.search(self.test_vectors, nn_quantity)
        search_end_time = time.time()
        memory = get_memory(index)
        precision_value = precision_top_n(I, self.test_vectors_info, self.train_vectors_info, nn_quantity)

        if show_index_info:
            print(f"Index contains {index.ntotal} embeddings")
            print("Time spent for searching"
                  " : {:.2f} seconds".format(search_end_time - search_start_time))
            print("Shape of D: {} and I: {} ".format(len(D), len(I)))
            print("Precision : {}".format(precision_value))
            print("Memory : {:.2f} MB".format(memory))


    def index_ivfpq(self, nn_quantity, n_list, n_probe_values, n_subquantizers,
                    n_bits, show_index_info=False):

        report = {}

        if type(n_probe_values) is not list: n_probe_values = [n_probe_values]
        search_time_values = []
        precision_values = []
        memory_values = []

        quantizer = faiss.IndexFlatIP(self.emb_dim)
        index = faiss.IndexIVFPQ(quantizer, self.emb_dim, n_list, n_subquantizers, n_bits)

        assert not index.is_trained
        train_start_time = time.time()
        index.train(self.train_vectors)
        train_end_time = time.time()       
        assert index.is_trained

        if show_index_info: print("Train time: {:.2f} seconds".format(train_end_time - train_start_time))
        report["train_time"] = train_end_time - train_start_time

        add_start_time = time.time()
        index.add(self.train_vectors)
        add_end_time = time.time()

        if show_index_info: print("Addition time: {:.2f} seconds".format(add_end_time - add_start_time))
        report["add_time"] = add_end_time - add_start_time

        for n_probe in n_probe_values:
            index.nprobe = n_probe

            if show_index_info: print("\tnprobe = {}".format(n_probe))

            search_start_time = time.time()
            D, I = index.search(self.test_vectors, nn_quantity)
            search_end_time = time.time()
            search_time_values.append(search_end_time-search_start_time)

            if show_index_info: print("\tSearch time: {:.2f} seconds".format(search_end_time-search_start_time), end=' ')

            precision = precision_top_n(I, self.test_vectors_info, self.train_vectors_info, nn_quantity)
            precision_values.append(precision)
            
            if show_index_info: print("Precision: {}".format(precision), end=' ')
            
            memory = get_memory(index)
            memory_values.append(memory)

            if show_index_info: print("Memory: {} MB".format(memory))            
        
        
        best_precision = max(precision_values)
        best_compressed_memory = min(memory_values)
        best_search_time = min(search_time_values)        
        
        report["precision"] = best_precision
        report["search_time"] = search_time_values[precision_values.index(best_precision)]
        report["nprobe"] = n_probe_values[precision_values.index(best_precision)]
        report["memory_size"] = memory_values[precision_values.index(best_precision)]


        if len(n_probe_values) > 1:
            print("-"*10)
            print("The highest precision: {:.5f} (nprobe={})".format(best_precision, n_probe_values[precision_values.index(best_precision)]))
            print("The smallest memory size: {} MB (nprobe={})".format(best_compressed_memory, n_probe_values[memory_values.index(best_compressed_memory)]))
            print("The fastest search: {:.2f}seconds (nprobe={})".format(best_search_time, n_probe_values[search_time_values.index(best_search_time)]))
            print("-"*10)

        return report
    

    def nbits_expr(self, nn_quantity, n_list, n_probe, n_subquantizers, show_process=False, plot_graph=False):  

        n_bits_expr_report = {}                
        precision_values = []
        search_time_values = []
        train_time_values = []
        add_time_values = []
        memory_sizes = []


        for n_bits in self.n_bits_values:
            if show_process: print(f"n_bits = {n_bits}")
            
            report = self.index_ivfpq(nn_quantity, n_list, n_probe, n_subquantizers,
                                                 n_bits, show_index_info=show_process)

            precision_values.append(report["precision"])
            search_time_values.append(report["search_time"])
            train_time_values.append(report["train_time"])
            add_time_values.append(report["add_time"])
            memory_sizes.append(report["memory_size"])
        
        if plot_graph:
            plt.figure(figsize=(10, 6), dpi=100)
            
            plt.subplot(211)
            plt.xlabel('Количество бит')
            plt.ylabel('Оценка качества')
            plt.plot(self.n_bits_values, precision_values, '-o')

            plt.subplot(212)
            plt.xlabel('Количество бит')
            plt.ylabel('Время (в секундах)')
            plt.plot(self.n_bits_values, train_time_values, '-o', label='Обучение')
            plt.plot(self.n_bits_values, add_time_values, '-o', label='Добавление')
            plt.plot(self.n_bits_values, search_time_values, '-o', label='Поиск')
            plt.legend()

            plt.show()

        n_bits_expr_report["precision"] = max(precision_values)
        precision_id = precision_values.index(n_bits_expr_report["precision"])
        n_bits_expr_report["n_bits"] = self.n_bits_values[precision_id]
        n_bits_expr_report["search_time"] = search_time_values[precision_id]
        n_bits_expr_report["train_time"] = train_time_values[precision_id]
        n_bits_expr_report["add_time"] = add_time_values[precision_id]
        n_bits_expr_report["memory_size"] = memory_sizes[precision_id]

        return n_bits_expr_report

    def n_subquantizers_expr(self, nn_quantity, n_list, n_probe, n_bits, show_process=False, plot_graph=False):
        
        n_subquantizers_expr_report = {}
        precision_values = []
        search_time_values = []
        train_time_values = []
        add_time_values = []
        memory_sizes = []

        for n_subquantizers in self.n_subquantizers_values:
            if show_process: print(f"n_subquantizers = {n_subquantizers}")
            
            report = self.index_ivfpq(nn_quantity, n_list, n_probe, n_subquantizers,
                                      n_bits, show_index_info=show_process)

            precision_values.append(report["precision"])
            search_time_values.append(report["search_time"])
            train_time_values.append(report["train_time"])
            add_time_values.append(report["add_time"])
            memory_sizes.append(report["memory_size"])

        if plot_graph:
            plt.figure(figsize=(10, 6), dpi=100)
            
            plt.subplot(211)
            plt.xlabel('Количество бит')
            plt.ylabel('Оценка качества')
            plt.plot(self.n_subquantizers_values, precision_values, '-o')

            plt.subplot(212)
            plt.xlabel('Количество бит')
            plt.ylabel('Время (в секундах)')
            plt.plot(self.n_subquantizers_values, train_time_values, '-o', label='Обучение')
            plt.plot(self.n_subquantizers_values, add_time_values, '-o', label='Добавление')
            plt.plot(self.n_subquantizers_values, search_time_values, '-o', label='Поиск')
            plt.legend()

            plt.show()

        n_subquantizers_expr_report["precision"] = max(precision_values)
        precision_id = precision_values.index(n_subquantizers_expr_report["precision"])
        n_subquantizers_expr_report["n_subquantizers"] = self.n_subquantizers_values[precision_id]
        n_subquantizers_expr_report["search_time"] = search_time_values[precision_id]
        n_subquantizers_expr_report["train_time"] = train_time_values[precision_id]
        n_subquantizers_expr_report["add_time"] = add_time_values[precision_id]
        n_subquantizers_expr_report["memory_size"] = memory_sizes[precision_id]
            
        return n_subquantizers_expr_report

    def n_list_nprobe_expr(self, nn_quantity, n_subquantizers, n_bits, show_process=False, plot_graph=False):
        
        n_list_nprobe_expr_report = {}

        best_n_probe_values = []
        precision_values = []
        search_time_values = []
        train_time_values = []
        add_time_values = []
        memory_sizes = []

        for n_list in self.n_list_values:
            
            if show_process: print(f"n_list = {n_list}")

            n_probe_values = list(range(int(n_list/10), n_list+1, int(n_list/10)))

            report = self.index_ivfpq(nn_quantity, n_list, n_probe_values, n_subquantizers,
                                      n_bits, show_index_info=show_process)

            precision_values.append(report["precision"])
            search_time_values.append(report["search_time"])
            train_time_values.append(report["train_time"])
            add_time_values.append(report["add_time"])
            memory_sizes.append(report["memory_size"])
            best_n_probe_values.append(report["nprobe"])

        if plot_graph:
            plt.figure(figsize=(20,15), dpi=120)

            plt.subplot(211)
            plt.plot(self.n_list_values, precision_values)
            for i in range(len(self.n_list_values)):
                plt.text(self.n_list_values[i], precision_values[i], best_n_probe_values[i],
                         color = 'y', fontsize=12)
            plt.xticks(self.n_list_values)
            plt.yticks(precision_values)
            plt.xlabel('Количество ячеек Вороного')
            plt.ylabel('Оценка качества')

            plt.subplot(212)
            plt.xlabel('Количество бит')
            plt.ylabel('Время (в секундах)')
            plt.plot(self.n_list_values, train_time_values, '-o', label='Обучение')
            plt.plot(self.n_list_values, add_time_values, '-o', label='Добавление')
            plt.plot(self.n_list_values, search_time_values, '-o', label='Поиск')
            plt.legend()

            plt.show()
        
        n_list_nprobe_expr_report["precision"] = max(precision_values)
        precision_id = precision_values.index(n_list_nprobe_expr_report["precision"])
        n_list_nprobe_expr_report["n_list"] = self.n_list_values[precision_id]
        n_list_nprobe_expr_report["n_probe"] = best_n_probe_values[precision_id]
        n_list_nprobe_expr_report["search_time"] = search_time_values[precision_id]
        n_list_nprobe_expr_report["train_time"] = train_time_values[precision_id]
        n_list_nprobe_expr_report["add_time"] = add_time_values[precision_id]
        n_list_nprobe_expr_report["memory_size"] = memory_sizes[precision_id]
            
        return n_list_nprobe_expr_report 



    
    def find_optim_params(self, nn_quantity, n_bits, n_subquantizers, show_process=False, plot_graph=False):
        
        if show_process:
            print(f"Finding optim parameters\nInitial values: {n_bits=} {n_subquantizers=}")
            
        report_n_list_n_probe = self.n_list_nprobe_expr(nn_quantity, n_subquantizers, n_bits, show_process=show_process, plot_graph=True)
        optim_n_list = report_n_list_n_probe["n_list"]
        optim_n_probe = report_n_list_n_probe["n_probe"]
        report_n_subquantizers = self.n_subquantizers_expr(nn_quantity, optim_n_list, optim_n_probe, n_bits, show_process=show_process, plot_graph=plot_graph)
        optim_n_subquantizers = report_n_subquantizers["n_subquantizers"]
        report_nbits = self.nbits_expr(nn_quantity, optim_n_list, optim_n_probe, optim_n_subquantizers, show_process=show_process, plot_graph=plot_graph)
        optim_n_bits = report_nbits["n_bits"]
        report_optim = self.index_ivfpq(nn_quantity, optim_n_list, optim_n_probe, optim_n_subquantizers, optim_n_bits, show_index_info=show_process)
        report_optim["n_list"] = optim_n_list
        report_optim["n_subquantizers"] = optim_n_subquantizers
        report_optim["n_bits"] = optim_n_bits
        if show_process: print(f"Find optim params report:\n{report_optim}")
        return report_optim

    def find_params_until_threshold(self, nn_quantity, threshold, show_process=False, plot_graph=False):

        n_bits = random.choice(self.n_bits_values)
        n_subquantizers = random.choice(self.n_subquantizers_values)       

        over_threshold = True
        history = []

        previous_precision = 0
        counter = 1

        while over_threshold:
            if show_process:
                print("-"*15)
                print(f"Iteration #{counter}")
            report = self.find_optim_params(nn_quantity, n_bits, n_subquantizers, show_process, plot_graph)            
            n_bits = report["n_bits"]
            n_subquantizers = report["n_subquantizers"]
            if np.abs(report["precision"] - previous_precision) * 100 <= threshold:
                over_threshold = False
            else:
                previous_precision = report["precision"]
            
            history.append(report)
            if show_process: print("-"*15)
            counter += 1
        return history

    def find_params_until_iterations(self, nn_quantity, iterations, show_process=False, plot_graph=False):
        
        n_bits = random.choice(self.n_bits_values)
        n_subquantizers = random.choice(self.n_subquantizers_values)
        
        history = []

        
        for i in range(iterations):
            if show_process:
                print("-"*15)
                print(f"Iteration #{i+1}")
            
            report = self.find_optim_params(nn_quantity, n_bits, n_subquantizers, show_process, plot_graph)            
            n_bits = report["n_bits"]
            n_subquantizers = report["n_subquantizers"]            
            history.append(report)
            
        return history 




    







            


   
        

        