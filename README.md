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

![Alt text](https://github.com/mau-arrieta/Doc-Anonymizer/blob/main/images/General%20Pipeline%20Concept.png)


These labels are then used to classify the annotations into these 12 classes. This data along with the bounding box pixel coordinates are included in the annotation files. 


![Alt text](https://github.com/mau-arrieta/Doc-Anonymizer/blob/main/images/General%20Pipeline%20Concept.png)


The pixel coordinates for these bounding boxes is crucial for the project, as they determine wether the model is going to learn valuable information or if it's simply going to underfit for our task. According to [Docbank's official github](https://github.com/doc-analysis/DocBank/tree/master), these coordinates were normalized to 1000xx1000 grid, so we use the same normalization to maintain coherence with the dataset.

![Alt text](https://github.com/mau-arrieta/Doc-Anonymizer/blob/main/images/General%20Pipeline%20Concept.png)

A visualization of the ground-truth bounding boxes with their correct normalization is shown next: 

![Alt text](https://github.com/mau-arrieta/Doc-Anonymizer/blob/main/images/General%20Pipeline%20Concept.png)


The annotations included tokens our model didn't need to learn for this task (symbols, formatting sequences), so we filtered our annotation files by matching a REGEX which left us only those tokens that included words longer (or equal to) three letters, and that were made up of letters in the english alphabet (including those with hyphens and apostrophes).


The whole dataset was approximately 50gb, our model used 2.4% of it for fine-tuning due to storage constraints the team worked with. Our final version was fine-tuned with a total of 12k document images, with a 80-10-10 data partition for training,validation and testing, in that order.



### Architecture

### Results 

### Conclusions

## OPTICAL CHARACTER RECOGNITION

### Hypothesis
### Dataset
### Architecture
### Results 
### Conclusions

## NATURAL LANGUAGE PROCESSING

### Architecture
### Results

## BLURRING

### Architecture
### Results

## HOW TO RUN
