## Сжатие биометрических векторов лиц


Распознаванием лиц часто называют набор различных задач, например, поиск нужного человека среди множества изображений (идентификация) или проверка того, что на двух изображениях один и тот же человек (верификация). Особую роль в таких задачах играют биометрические вектора лиц, которые служат для представления различных
особенностей лиц в виде специальных векторов. Размерность векторов может отличаться в зависимости от методов, которые используются для их вычисления, но обычно равна 128 или 512. Однако, хранение большого количества биометрических векторов с большими размерностями требует немалых финансовых ресурсов для обеспечения достаточного дискового пространства и, во многих случаях, является невыгодным.

В целях уменьшения затрат на хранение, можно использовать различные методы сжатия векторов признаков, которые позволяют уменьшить требуемый объём памяти
в несколько раз. На сегодняшний день, известно множество методов сжатия векторов,
среди которых стоит отметить такой метод, как [Product Quantization](https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf), который основывается на идее разделения исходного вектора на несколько частей и кодирования
каждой части несколькими битами. Для применения различных способов сжатия векторов можно воспользоваться инструментами библиотеки [FAISS (Facebook AI Research
Similarity Search)](https://github.com/facebookresearch/faiss).

Используя инструменты данного репозитория Вы сможете провести эксперименты сжатия биометрических векторов 
и найти оптимальные значения параметров индекса IVFPQ для вашей базы данных. Для этого, сперва, рекомендуется изучить [пример](https://github.com/SamandarYokubov/face_embeddings_compression/blob/main/example/lfw_compression_experiments.ipynb), в котором был проведен эксперимент сжатия биометрических векторов лиц базы данных [LFW](http://vis-www.cs.umass.edu/lfw/).
