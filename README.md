## Face Embeddings Compression

Face recognition is a set of various tasks, for example, searching an individual among given dataset (identification) or checking whether faces in two images belong
to the same individual or not (verification). Face embeddings -  special vectors that represent features of face - play great role in face recognition systems. However, 
storing vast amount of high-dimensional embeddings requires great financial resources along with computing ones. In order to tackle memory related bariers, it is prudent 
to shrink embeddings by using compression algorithms, among which the most common one is [Product Quantization](https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf). By varying the values of parameters of Product Quantization, 
it is possible to not only compress face embeddings in a more effective way, but also achieve good values of accuracy indice. In this repository, you can find useful tools to 
find optimal parameters of Product Quantization algorithm. It is highly recomended to move on [example](https://github.com/SamandarYokubov/face_embeddings_compression/blob/main/example/lfw_compression_experiments.ipynb),
which represents compression experiments of [LFW](http://vis-www.cs.umass.edu/lfw/)'s face embeddings.

