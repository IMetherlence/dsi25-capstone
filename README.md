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

Our MIDI data is imported and one-hot encoded and fed into our 2 models, Long-Short Term Memory Recurrent Neural Network model (LSTM-RNN) and Convolutional Neural Network (CNN). A big bulk of this project was to run the model with different epochs. Running high number of epochs results in huge computational demands and thus we have to optimise our model for computational requirements as well, thus the lack of further exploration with other models due to the enormous computational demands.

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

#### Polyphony in Generated Music
The generated music only produces monophonic output. We can explore the possibility of parallelising more MIDI data with accompaniments and melody to produce more polyphonic music. This can also be done with Baroque music, which usually consist of polyphonic melodies layered on one another.

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


# Train and Generate Music
Our models are Long-Short Term Memory Recurrent Neural Network (LSTM-RNN) and Convolutional Neural Network (CNN). Our One-hot encoded data is fed into the model and trained for different number of epochs to get a scale for the level of training for each model.  

After our model has been trained, we set out to generate music into our `output` folder.LSTM is capable of retaining information through epochs and thus is chosen for its better output in music. CNN model is able to learn the music initially however due to the lack of memory in the model, it is unable to retain logical patterns in our training pieces and thus generated music that initially sounded ok, albeit with a lot of repeated notes. Afterwhich, it simply started playing rather poorly with repeated notes.

Due to GitHub restrictions, the `output` folder will be left empty. Here is the link to the actual generated music:
https://drive.google.com/drive/folders/1D162XP9uDOglr2yP9tMDASadikF1L9Os?usp=sharing

---
# Comparison of Generated Music by Ear
It is clear that there are the presence of several repeated notes in CNN. The lack of any long short term memory on CNN has resulted in the melody sounding random at the start, before shifting to just single notes as it is unable to form further associations with the training music it is given. In comparison, LSTM-RNN performed much better as the data is fed back into the model and retained through epochs, allowing it to develop further musical learning and generate more complex music.

Loss and Accuracy trends for LSTM-RNN is a lot smoother, indicating that the model may be training properly to the underlying rules in the training data, compared to CNN which had more varied graduations in both curves.

---
# Conclusions
We have discovered that CNN LSTM is capable of producing listenable music that has learnt some level of musical rules. There are more levels of optimisation that can be done, and more models that can be explored such as musicGAN, PixelCNN, LSTM-CNN and HyperGAN, all which require much more computing power.

One important realisation is the computational requirement of running such a model. It is true that running the model may be a one-off as most people may use it to generate music once and use it. However, I cannot emphasize the need for repeatability and optimisation for consumers. While each run of the model may result in different original music, we want the music to follow the style and rules in the training material.

We can deploy the model to content creators who often find ideas with musical ideas that they have. In particular, we could parse the MIDI data from known sources, especially with the prevalence of fan-created music that often contains MIDI data. Afterwhich, the model could produce tracks by outputting a MIDI, in which content creators could use.

---
