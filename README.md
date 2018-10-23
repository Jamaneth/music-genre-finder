Finding The Genre of Music Tracks with Convolutional Neural Networks
========

The goal of this project is to manage to create a model that can accurately identify the genre of an unknown song, among eight possible musical genres:
- Electronic
- Experimental
- Folk
- Hip-Hop
- Instrumental
- International
- Pop
- Rock

## Dataset
The dataset we will be using here comes from a dump of the Free Music Archive (FMA), available [in this Github repository](https://github.com/mdeff/fma). The complete description of this project may be found [in this paper](https://arxiv.org/abs/1612.01840).

To summarise, whole dataset is comprised of 106,574 of Creative Commons-licenced tracks from 16,341 artists and 14,854 albums, arranged in a hierarchical structure of 161 genres. However, in this notebook, we will be focusing on a **smaller subset** of this initial dataset: the dataset we will be focusing on will be comprised of **8,000 30 second clips**. Those clips belong to 8 top genres, and they are balanced – with exactly 1,000 clips per genre.


## General method
Our approach to identify the genre of a music track relies on the use of a Convolutional Neural Network (CNN), and was influenced by ([Dong 2018](https://arxiv.org/abs/1802.09697)):

- We first transform each sound track into a mel spectrogram. This is a way to transform a sound into an image, and it basically allows us to transform a sound-recognition problem into a computer vision problem.

- Each track is 30-second long, which is quite long: we therefore split each spectrogram into ten 3-second long spectrograms, based on the assumption that an average human can recognise the genre of a song fairly accurately in that timeframe, and therefore that a computer should be able to do so as well,

- We use those short spectrograms as an input to train the CNN.

Since we did not have a computer with a GPU, the model has been trained on [Google Colab](https://colab.research.google.com/) – in order to take advantage of a free GPU acceleration.

## Results
For the time being, our model can achieve **41% accuracy** on the validation dataset. This is an unquestionable progress when compared to the dummy classifier's performance (12.5%), but there is still a bit of room for improvement.
- The lack of available RAM on Google Colab (the notebook automatically crashes when the RAM used is higher than 12 GB) and the restriction on GPU usage makes it difficult to train the neural network on a big dataset,
- For this reason, we only managed to train the model on 1,000 of the 8,000 songs, both because the original dataset was too inconvenient to manipulate and because the neural network becomes very slow to train,
-  A potential improvement might be to check the performance of a neural network that has been pre-trained on a big image database, such as ImageNet.
