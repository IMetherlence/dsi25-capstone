<img src="http://imgur.com/1ZcRyrc.png" style="float: left; margin: 20px; height: 55px">  

# Capstone Project



# Music Generation From MIDI with Machine Learning Networks

## Contents

- [Executive Summary](#Executive-Summary)
- [Problem Statement](#Problem-Statement)
- [Background Research](#Background-Research)
- [Future Improvement](#Future-Improvement)
- [Import Data and EDA](#Import-Data-and-EDA)
- [Further EDA](#Further-EDA)
- [Shaping Data for our Model](#Shaping-Data-for-our-Model)
- [Running LSTM-RNN Model](#Running-LSTM-RNN-Model)
- [Generate Music](#Generate-Music)
- [Display and Produce Generated Music](#Display-and-Produce-Generated-Music)
- [Continuation](#Continuation)

---

# Executive Summary
Music generation via Machine Learning has been frequently discussed as computing power increased. The possibility of generating music from Machine Learning models yields opportunities to create music for those who do not have sufficient musical aptitude to write their own, as well as a good opportunity to teach students and hobbyist alike several key concepts in music.

For this project, I have taken MIDI files from https://www.classicalarchives.com/midi/composer/2156.html  
We will be focusing on mostly Beethoven's Sonata.

The music to be used on this project are composed by the famous composer Ludwig Van Beethoven. I have chosen Beethoven because Beethoven was trained on the classical harmonies with Joseph Haydn, and is one of the pioneers of the Romantic era, being one of the pre-Romantic Era composers.

Our MIDI data is imported and one-hot encoded and fed into our 2 models, Long-Short Term Memory Recurrent Neural Network model (LSTM-RNN) and Convolutional Neural Network (CNN). A big bulk of this project was to run the model with different epochs. Running high number of epochs results in huge computational demands and thus the lack of exploration in other possible models.

Upon listening to our generated music located in our `output` folder, I can hear that both our model is able to learn the concept of keys and key signatures. Generally both models have a downward trend on it's loss and a higher level of accuracy to our training data

On our LSTM-RNN model, we can hear more variation in the melody, which is more interesting. The piece is also rather musically sound as it follows a key signature. There is also use of borrowed chords and keys in our LSTM-RNN generated music.

Our CNN model, however, resulted in bad results. There are frequent repeated notes that indicates a lack of long-term memory for logical rules that we are hoping the model would pick up on.

---

# Problem Statement:
Content creators often face issues with music and audio when creating original content. They often need background music to use, which results in royalty payments, already setting up an upfront cost to even post contents. Some may even try to make their own music, only to find that the music used or composed for their content is too similar to another work. This creates several conflicts of interests in the media industry.

This project aims to mitigate this issue by generating original music for the content produced based in the music fed into the model. Also, this project is attempting to replicate styles of famous composers in the past so that it can stay within a theme for content creators to follow.

---

# Background Research
There is assymetry in the timing of several famous composers. While playing their old works may yield great performances, this project also aim to explore the possibility of training a model to compose music with specific pieces from a composer. In this project I will be using Ludwig Van Beethoven's Piano Sonatas to train our models. After training our models, I will generate an output music.

---

# Future Improvement
#### GPU Acceleration for faster model training
One of the biggest gripes of this project is that it was not ran on a GPU. Given time, I would have preferred to run the process on a GPU instead, which will speed the model up in orders of magnitude. However, Tensorflow was compiled with a much more powerful GPU and thus a different driver altogether and thus causes compatibility issues when attempting to run the model with CUDA acceleration.

#### Pre-train model on MIDI based on Harmonic logics and known musical patterns first
It may also be useful to train harmonic logic rules and known musical patterns into the model prior to training the model based on the composer's style. This may increase the musicality of the generated output.

#### Account for Velocity data in MIDI with additional dimension into our models
Also, this method discards velocity data, which may be a part of Beethoven's signature (Beethoven is known for large dynamic jumps). This could be circumvented by introducing another array of colours to represent velocity before feeding it into the models, however due to the intensity of running the models, it is not feasible to run with velocity data with the current hardware used.

#### Potential Data Cleaning Steps
It may be possible to clean the data further by taking into account of key signatures for each piece, then tagging them to a library of musical technical names, before feeding the data into a one-hot encoder and into our models. This may yield better results. However, as we are using Pretty_MIDI, which should take the keys into account, we will be dropping this method for now.

---
# Import Data and EDA
We need to import our MIDI data into the notebook. Let's start by creating a list containing the locations of all our MIDI data first.

There is no data cleaning step for this because we are extracting already clean data from the MIDI archives, which is a performance of Beethoven's Sonatas and has no null, missing or outlier datas to begin with.

We will proceed with a key-note EDA on each piece that we import.

Afterwards, we will perform one-hot encoding on our data into an array.

# Further EDA
Let's start exploring our data more in-depth. We know that there may be different key signatures for these pieces, as well as the frequency of each notes. Let's look at these data points

---

# Shaping Data for our Model
We need each excerpt of the pieces to be shaped properly so that our model can be trained. To do this, we create 2 sequences:
- An input sequence which contains a sub-sequence of length sequence_length.  
This sub-sequence ranges from the note t to the note t+sequence\_length-1  


- An output sequence which contains the following note to be predicted, the one at position t+sequence\_length. The training is therefore performed by giving to the model a set of sequences as input and asking the network to predict each time the note that should come right after this sequence.

# Running LSTM-RNN Model
Let's start by running our RNN-LSTM model on our data. We will run our model with these following parameters:  

1st layer units = 64  
Dropout layer = 0.3  
2nd layer units = 64  
Dropout layer = 0.3  
3rd layer units = 64  
Dense layer with 64 units  
Another Dropout layer = 0.3  
A Dense layer with a softmax activation  

# Train the LSTM model

Let's start our model training. The epoch number here can be changed to any numbers. Note that if we were to run with a high epoch, the model will take very long to run.

Our Accuracy graph is generally increasing the more training epochs the model goes through, which is expected. The Loss also decreases with each epochs. This indicates that the model is training properly.

Note that in Music Generation from Machine Learning Models, we do not want the model to be highly accurate. This results in overfitting and indicates that the model regurgitating training data, and is undesirable. Keep in mind that the objective of this project is to generate original music in the style of Beethoven, not to repeat exact materials or extracts from the Beethoven's Sonatas.

# Generate Music
Let's use our LSTM model to generate our music."# dsi25-capstone" 
