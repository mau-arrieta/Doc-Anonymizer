# DOCUMENT ANONYMIZATION
Final project for the 2025 Postgraduate course on Artificial Intelligence with Deep Learning, UPC School, proposed by **Mauricio Arrieta**, **Adrià Buil**, **Xavi Rodríguez** and **Antoni Jordi Noguera**. 

Advised by **Pol Caselles**.

Table of Contents
=================


  * [Motivation](#motivation)
  * [Project Pipeline](#pipeline)
  * [Computer Vision](#computer-vision)
  * [Optical Character Recognition](#optical-character-recognition)
  * [Natural Language Processing](#natural-language-processing)
  * [Blurring](#blurring)
  * [How to Run](#how-to-run)



## MOTIVATION

In today’s digital age, the need to protect our personal and sensitive information has never been more urgent. Manual redaction is not only time-consuming but also prone to human error, making automated solutions a necessity. 

This project aims to develop a modular, deep learning-based system capable of detecting and anonymizing sensitive data and personal identifier information in scanned documents. By doing so, it seeks to create a practical, scalable tool for safeguarding privacy in real-world contexts like CVs, medical records, and administrative forms.

## PIPELINE

The proposal consists of four main modules, integrated into a pipeline that uses an image as input, and outputs the same image, with its sensitive text data blurred out. The four modules and their main functions are:

  1.Computer Vision: In charge of detecting text in an image, at the word level. Will receive a document image as input, and will output a list of images, consisting of crops of the original image where text is detected/predicted. Will also output the bounding box coordinates for these crops, as they will be needed to blur the image, if the word in that crop is considered to be sensitive information.
  
  2.Optical Character Recognition: Image to text capability. Will take an image containing a word as input, and predict the text written in that image. 
  
  3.Natural Language Processing: Takes words as input, and classifies them into two classes: sensitive or not sensitive. 
  
  4.Blurring: With the set of coordinates provided from the computer vision module, and the classification done by the NLP module, this module is in charge of blurring the words classified as sensitive in the original image.

  
![Alt text](https://github.com/mau-arrieta/Doc-Anonymizer/blob/main/images/General%20Pipeline%20Concept.png)



## COMPUTER VISION

### Hypothesis

It is possible to accurately detect and localize individual words in document images by learning visual patterns within these documents

We expect that a pre-trained detection model can be fine-tuned for our use case, and learn how to distinguish word regions from non-text background across a wide variety of documents. Achieving robust word-level detection will enable downstream models such as the OCR to correctly predict text from our image crops. 

Furthermore, the module is designed with a Faster R-CNN based model for out text detection task, as its two-stage design (region proposal + classification) makes it especially strong at precise localization, which is critical for detecting small, tightly packed word regions that simpler one-stage detectors often miss.


### Dataset

The dataset used for the word-box detection task is the [DocBank dataset](https://doc-analysis.github.io/docbank-page/). It was originally designed for textual and document layout tasks, and the version used for this project consists of 500K document pages, along with their corresponging annotations.

Annotations are based on 12 labels, which correspond to each area of the document layout:

![Alt text](https://github.com/mau-arrieta/Doc-Anonymizer/blob/main/images/docbankLayout.png)


These labels are then used to classify the annotations into these 12 classes. This data along with the bounding box pixel coordinates are included in the annotation files. 


![Alt text](https://github.com/mau-arrieta/Doc-Anonymizer/blob/main/images/docbankAnnotations.png)


The pixel coordinates for these bounding boxes is crucial for the project, as they determine wether the model is going to learn valuable information or if it's simply going to underfit for our task. According to [Docbank's official github](https://github.com/doc-analysis/DocBank/tree/master), these coordinates were normalized to 1000xx1000 grid, so we use the same normalization to maintain coherence with the dataset.


A visualization of the ground-truth bounding boxes with their correct normalization is shown next: 

![Alt text](https://github.com/mau-arrieta/Doc-Anonymizer/blob/main/images/ground-truth-labels.png)


The annotations included tokens our model didn't need to learn for this task (symbols, formatting sequences), so we filtered our annotation files by matching a REGEX which left us only those tokens that included words longer (or equal to) three letters, and that were made up of letters in the english alphabet (including those with hyphens and apostrophes).


The whole dataset was approximately 50gb, our model used 2.4% of it for fine-tuning due to storage constraints the team worked with. Our final version was fine-tuned with a total of 12k document images, with a 80-10-10 data partition for training,validation and testing, in that order.



### Architecture

### Metrics

### Results 

Runnning inference on the test portion of the dataset, it's possible to visually examine the results of the model. 

<img width="557" height="657" alt="image" src="https://github.com/user-attachments/assets/cce7e4e3-2406-4612-9d8d-d04adcb72472" />

The  bounding boxes are marked in red, and there are 109 of the total 200 predictions that have a confidence score over 0.7 or 70%. The scores represent the model’s estimated confidence that a given bounding box contains a valid object (excluding background). In the project's use case, this indicates that the box contains text. These values are derived from the classification head of the object detection model, which computes a probability distribution over object classes using a softmax layer. These predictions are then used to create the image crops, which are fed into the OCR model. An example of the cropped words list is shown next:


<img width="615" height="510" alt="image" src="https://github.com/user-attachments/assets/473d95f1-a19b-4244-90a9-01573b001b73" />


### Conclusions

In this module of the project, we explored the effectiveness of fine-tuning a pre-trained Faster R-CNN model for word-level text detection in document images. Despite using only a small portion of the full dataset, the model successfully learned to identify and localize word regions with a high degree of confidence, even in visually dense and varied layouts. By carefully selecting and preprocessing the data, we ensured that the model focused on relevant content, which improved the quality of the extracted crops for downstream OCR. The results support our initial hypothesis and show that adapting a general-purpose detection model to a specialized task like this is not only feasible, but also highly effective when guided by the right data and design choices.

## OPTICAL CHARACTER RECOGNITION

### Hypothesis

This section focuses on the OCR module of the pipeline. Its input consists of cropped word images provided by the computer vision module, and its task is to predict the corresponding text for each image.

The hypothesis is that a deep learning model can learn to map each word-level image to its text representation, enabling accurate image-to-text conversion at the word level.

### Dataset

The MJSynth dataset (also known as Synth90k) contains approximately 9 million synthetic images of cropped words rendered with diverse fonts, backgrounds, and distortions to simulate real-world text recognition scenarios. In this project, it has been downloaded from Hugging Face (priyank-m/MJSynth_text_recognition · Datasets at Hugging Face).
<img width="455" height="145" alt="image" src="https://github.com/user-attachments/assets/5798f0a7-b237-4471-b20f-07e118e17477" />
 
#### Sampling and Storage
To manage dataset size, a subset of num_images samples (i.e., 25,000) has been selected. The dataset was first shuffled and then sampled images were saved to Google with standardized filenames (e.g. img_00001.jpg). Additionally, a label file has been created in the same directory, storing the relative image paths and their corresponding text labels separated by a tab. This setup ensures fast and organized data loading in later stages.

#### Custom Dataset Class (Data Loader)
A custom PyTorch dataset class has been implemented to handle data loading efficiently. Within this class:
-	Image paths and labels are read from the label file.
-	Each image is loaded in grayscale mode and resized to 1x32x128 (CxHxW) to maintain consistent input dimensions.
-	Images are converted into tensors using standard PyTorch transforms.
-	
#### Label Encoding
A character vocabulary is built from all unique characters present in the sampled labels (unless predefined). Labels are encoded as sequences of character indices based on this vocabulary. To handle variable-length labels, padding is applied to reach a fixed maximum label length using a dedicated padding token index.
In practice, vocab_size = 62 (a-z: 26 characters + A-Z: 26 characters + 0-9: 10 characters).

#### Batch Preparation
A custom collate function has been defined to stack images into batch tensors and convert the list of encoded labels into tensors, enabling efficient and streamlined data feeding during model training.


### Architecture

In this project, different OCR model architectures have been explored, depending on how the input image is processed and decoded into text. The main approaches considered are:
•	Image-to-Sequence (img2seq):
The image is first encoded as a sequence of features, typically by slicing it column-wise after passing through a CNN. The full word is then predicted as a sequence using models trained with Connectionist Temporal Classification (CTC) Loss. This has been the main approach used in the project.
•	Sequence-to-Sequence (seq2seq):
In this setup, the image is encoded into a fixed representation, and a decoder generates one character at a time. These models are typically trained with Cross Entropy Loss and include architectures like GRUs or Transformers with attention. Although initially explored, this approach was not included in the final evaluation.
•	Encoder Variants:
Various CNN-based backbones have been used to extract features from images. These include simple custom CNNs (e.g. TinyCNN) and deeper networks (e.g. ResNet), depending on the model variant being tested.

#### Img2Seq: CNN + BiLSTM + CTC Loss

In this architecture, a Convolutional Neural Network (CNN) is used as a feature extractor to encode the input image into a sequence of feature vectors. These are then passed to a Bidirectional LSTM (BiLSTM), which models the temporal dependencies in both forward and backward directions.
The model is trained using Connectionist Temporal Classification (CTC) loss, which allows alignment-free training — a key feature when no explicit character-to-position alignment is available in OCR datasets. 
This architecture has proven to be robust and stable, providing consistently good results in the experiments.

<img width="372" height="434" alt="image" src="https://github.com/user-attachments/assets/03b635f3-41a3-4da1-849e-95fcf2772626" />
https://arxiv.org/pdf/1507.05717

**Type**	                **Configurations**
Transcription	           -
Bidirectional-LSTM	      #hidden units: 256
Bidirectional-LSTM	      #hidden units: 256
Map-to-Sequence	         -
AdaptiveAvgPool2d	       Output size: (1, None)
MaxPooling	              Window: (2,1), stride: (2,1)
BatchNormalization	      512 channels
ReLU	                    - 
Convolution	             #maps:512, k:3×3, s:1, p:1
MaxPooling	              Window: (2,1), stride: (2,1)
Convolution	             #maps:256, k:3×3, s:1, p:1
Convolution	             #maps:256, k:3×3, s:1, p:1
MaxPooling	              Window:2×2, stride:2
Convolution	             #maps:128, k:3×3, s:1, p:1
MaxPooling	              Window:2×2, stride:2
Convolution	             #maps:64, k:3×3, s:1, p:1
Input	                   128 × 32 grayscale image


#### Img2Seq: CNN + Attention + CTC Loss

This variant uses the same CNN backbone to extract spatial features from the input image. Instead of a BiLSTM, it incorporates a learned attention mechanism to focus on different parts of the image sequence during decoding.
The attention layer is followed by a sequence of dense layers, and the model is also trained using CTC loss. This combination allows the model to dynamically weight relevant image regions, improving interpretability and performance in cases where fixed receptive fields are limiting.


### Metrics

#### Character Error Raet (CER)

CER is calculated based on the concept of Levenshtein distance, which counts the minimum number of character-level operations required to transform the ground truth text (also called the reference text) into the OCR output.
It is represented by the formula:
CER = (S+D+I)/N, where S = Number of Substitutions, D = Number of Deletions, I = Number of Insertions, N = Number of characters in reference text (aka ground truth).
The output of this equation represents the percentage of characters in the reference text that was incorrectly predicted in the OCR output. The lower the CER value (with 0 being a perfect score), the better the performance of the OCR model.

#### Word Accuracy @k

Word accuracy @ k measures the proportion of word predictions that match the ground truth with up to k character differences (i.e. edit distance tolerance).
-	This metric is useful when small typos are acceptable, providing a more flexible evaluation aligned with user-perceived correctness in OCR tasks.
-	Additionally, it can be used in further applications such as identifying the most similar word within a distance of k, which is helpful for known error correction or approximate matching in post-processing pipelines.


### Results 

<img width="865" height="548" alt="image" src="https://github.com/user-attachments/assets/92b5431b-d1b2-4ab7-8f1d-05b99dfd0181" />

#### Number of Parameters



### Conclusions



## NATURAL LANGUAGE PROCESSING

### Architecture
### Results

## BLURRING

### Architecture
### Results

## HOW TO RUN
