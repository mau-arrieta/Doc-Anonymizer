# DOCUMENT ANONYMIZATION
Final project for the 2025 Postgraduate course on Artificial Intelligence with Deep Learning, UPC School, proposed by **Mauricio Arrieta**, **Adrià Buil**, **Xavi Rodríguez** and **Antoni Jordi Noguera**. 

Advised by **Pol Caselles**.

Table of Contents
=================


  * [Motivation](#motivation)
  * [End-to-end Architecture](#architecture)
  * [Computer Vision](#computer-vision)
  * [Optical Character Recognition](#optical-character-recognition)
  * [Natural Language Processing](#natural-language-processing)
  * [Blurring](#blurring)
  * [How to Run](#how-to-run)



## MOTIVATION

In today’s digital age, the need to protect our personal and sensitive information has never been more urgent. Manual redaction is not only time-consuming but also prone to human error, making automated solutions a necessity. 

This project aims to develop a modular, deep learning-based system capable of detecting and anonymizing sensitive data and personal identifier information in scanned documents. By doing so, it seeks to create a practical, scalable tool for safeguarding privacy in real-world contexts like CVs, medical records, and administrative forms.

## END-TO-END ARCHITECTURE

The proposal consists of four main modules, integrated into a pipeline that uses an image as input, and outputs the same image, with its sensitive text data blurred out. The four modules and their main functions are:

  1.Computer Vision: In charge of detecting text in an image, at the word level. Will receive a document image as input, and will output a list of images, consisting of crops of the original image where text is detected/predicted. Will also output the bounding box coordinates for these crops, as they will be needed to blur the image, if the word in that crop is considered to be sensitive information.
  
  2.Optical Character Recognition: Image to text capability. Will take an image containing a word as input, and predict the text written in that image. 
  
  3.Natural Language Processing: Takes words as input, and classifies them into two classes: sensitive or not sensitive. 
  
  4.Blurring: With the set of coordinates provided from the computer vision module, and the classification done by the NLP module, this module is in charge of blurring the words classified as sensitive in the original image.
  
A sample of the planned architecture is shown here:
![Alt text](https://github.com/mau-arrieta/Doc-Anonymizer/blob/main/images/General%20Pipeline%20Concept.png)



## COMPUTER VISION

### Hypothesis
### Dataset
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
