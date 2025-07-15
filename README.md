# DOCUMENT ANONYMIZATION
Final project for the 2025 Postgraduate course on Artificial Intelligence with Deep Learning, UPC School, proposed by **Mauricio Arrieta**, **Adrià Buil**, **Xavi Rodríguez de la Rubia** and **Antoni Jordi Noguera**. 

Advised by **Pol Caselles**.

Table of Contents
=================


  * [1. Motivation](#motivation)
  * [2. Project Pipeline](#pipeline)
  * [3. Computer Vision](#computer-vision)
  * [4. Optical Character Recognition](#optical-character-recognition)
  * [5. Natural Language Processing](#natural-language-processing)
  * [6. Blurring](#blurring)
  * [7. How to Run](#how-to-run)



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

### 3.1. Hypothesis

It is possible to accurately detect and localize individual words in document images by learning visual patterns within these documents

We expect that a pre-trained detection model can be fine-tuned for our use case, and learn how to distinguish word regions from non-text background across a wide variety of documents. Achieving robust word-level detection will enable downstream models such as the OCR to correctly predict text from our image crops. 

Furthermore, the module is designed with a Faster R-CNN based model for out text detection task, as its two-stage design (region proposal + classification) makes it especially strong at precise localization, which is critical for detecting small, tightly packed word regions that simpler one-stage detectors often miss.


### 3.2. Dataset

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



### 3.3. Architecture

### 3.4. Metrics

Faster R-CNN breaks the text detection problem into two stages: finding possible text regions and classifying them. The losses reflect each of these tasks — whether it's finding a text region at all (train_loss_objectness), correctly identifying it as text (train_loss_classifier), or drawing the box in the right place (train_loss_box_reg). Together, these make up the total training loss, helping us monitor how well the model is learning to detect text.


<img width="1382" height="634" alt="image" src="https://github.com/user-attachments/assets/0891e3b8-ff03-482e-9c2a-172a01bb5790" />


<img width="1382" height="634" alt="image" src="https://github.com/user-attachments/assets/01123a16-a728-43df-9440-a2439e30eac2" />


### 3.5. Results 

Runnning inference on the test portion of the dataset, it's possible to visually examine the results of the model. 


<img width="563" height="616" alt="image" src="https://github.com/user-attachments/assets/bdd8ec78-da14-4d72-a08f-d9465492c3d0" />


The  bounding boxes are marked in red, and there are 109 of the total 200 predictions that have a confidence score over 0.7 or 70%. The scores represent the model’s estimated confidence that a given bounding box contains a valid object (excluding background). In the project's use case, this indicates that the box contains text. These values are derived from the classification head of the object detection model, which computes a probability distribution over object classes using a softmax layer. These predictions are then used to create the image crops, which are fed into the OCR model. An example of the cropped words list is shown next:


<img width="615" height="510" alt="image" src="https://github.com/user-attachments/assets/473d95f1-a19b-4244-90a9-01573b001b73" />


### 3.6. Conclusions

In this module of the project, we explored the effectiveness of fine-tuning a pre-trained Faster R-CNN model for word-level text detection in document images. Despite using only a small portion of the full dataset, the model successfully learned to identify and localize word regions with a high degree of confidence, even in visually dense and varied layouts. By carefully selecting and preprocessing the data, we ensured that the model focused on relevant content, which improved the quality of the extracted crops for downstream OCR. The results support our initial hypothesis and show that adapting a general-purpose detection model to a specialized task like this is not only feasible, but also highly effective when guided by the right data and design choices.


## OPTICAL CHARACTER RECOGNITION

### 4.1. Hypothesis

This section focuses on the OCR module of the pipeline. Its input consists of cropped word images provided by the computer vision module, and its task is to predict the corresponding text for each image.

The hypothesis is that a deep learning model can learn to map each word-level image to its text representation, enabling accurate image-to-text conversion at the word level.

### 4.2. Dataset

The MJSynth dataset (also known as Synth90k) contains approximately 9 million synthetic images of cropped words rendered with diverse fonts, backgrounds, and distortions to simulate real-world text recognition scenarios. In this project, it has been downloaded from Hugging Face (priyank-m/MJSynth_text_recognition · Datasets at Hugging Face).

<img width="455" height="145" alt="image" src="https://github.com/user-attachments/assets/5798f0a7-b237-4471-b20f-07e118e17477" />
 
#### 4.2.1. Sampling and Storage
To manage dataset size, a subset of num_images samples (i.e., 25,000) has been selected. The dataset was first shuffled and then sampled images were saved to Google with standardized filenames (e.g. img_00001.jpg). Additionally, a label file has been created in the same directory, storing the relative image paths and their corresponding text labels separated by a tab. This setup ensures fast and organized data loading in later stages.

#### 4.2.2. Custom Dataset Class (Data Loader)
A custom PyTorch dataset class has been implemented to handle data loading efficiently. Within this class:
-	Image paths and labels are read from the label file.
-	Each image is loaded in grayscale mode and resized to 1x32x128 (CxHxW) to maintain consistent input dimensions.
-	Images are converted into tensors using standard PyTorch transforms.
#### 4.2.3. Label Encoding
A character vocabulary is built from all unique characters present in the sampled labels (unless predefined). Labels are encoded as sequences of character indices based on this vocabulary. To handle variable-length labels, padding is applied to reach a fixed maximum label length using a dedicated padding token index.
In practice, vocab_size = 62 (a-z: 26 characters + A-Z: 26 characters + 0-9: 10 characters).

#### 4.2.4. Batch Preparation
A custom collate function has been defined to stack images into batch tensors and convert the list of encoded labels into tensors, enabling efficient and streamlined data feeding during model training.


### 4.3. Metrics defined for evaluation

#### 4.3.1. Character Error Rate (CER)

CER is calculated based on the concept of Levenshtein distance, which counts the minimum number of character-level operations required to transform the ground truth text (also called the reference text) into the OCR output.
It is represented by the formula:
CER = (S+D+I)/N, where S = Number of Substitutions, D = Number of Deletions, I = Number of Insertions, N = Number of characters in reference text (aka ground truth).
The output of this equation represents the percentage of characters in the reference text that was incorrectly predicted in the OCR output. The lower the CER value (with 0 being a perfect score), the better the performance of the OCR model.


#### 4.3.2. Word Accuracy @k

Word accuracy @ k measures the proportion of word predictions that match the ground truth with up to k character differences (i.e. edit distance tolerance).
-	This metric is useful when small typos are acceptable, providing a more flexible evaluation aligned with user-perceived correctness in OCR tasks.
-	Additionally, it can be used in further applications such as identifying the most similar word within a distance of k, which is helpful for known error correction or approximate matching in post-processing pipelines.


### 4.4. Architectures and Results

In this project, different OCR model architectures have been explored, depending on how the input image is processed and decoded into text. The main approaches considered are:
- Image-to-Sequence (img2seq):
The image is first encoded as a sequence of features, typically by slicing it column-wise after passing through a CNN. The full word is then predicted as a sequence using models trained with Connectionist Temporal Classification (CTC) Loss. This has been the main approach used in the project.
- Sequence-to-Sequence (seq2seq):
In this setup, the image is encoded into a fixed representation, and a decoder generates one character at a time. These models are typically trained with Cross Entropy Loss and include architectures like GRUs or Transformers with attention. Although initially explored, this autoregressive approach was not included in the final evaluation.
- Encoder Variants:
Various CNN-based backbones have been used to extract features from images. These include simple custom CNNs (e.g. TinyCNN) and deeper networks (e.g. ResNet), depending on the model variant being tested.

#### 4.4.1. Image-to-Sequence approach:

The objective of this notebook (OCR_img_to_seq.ipynb) is trying to replicate a classical CRNN architecure, which consists of 3 structural blocks: the convolutional layers, the recurrent layers, and a transcription layer. To explore other architectures, the Recurrent Layer has been sobtituted by an Attention module and results have been compared into WandB.

#### 4.4.1.1. CRNN: CNN + BiLSTM + CTC Loss

In this architecture, a Convolutional Neural Network (CNN) is used as a feature extractor to encode the input image into a sequence of feature vectors. These are then passed to a Bidirectional LSTM (BiLSTM), which models the temporal dependencies in both forward and backward directions.
The model is trained using Connectionist Temporal Classification (CTC) loss, which allows alignment-free training — a key feature when no explicit character-to-position alignment is available in OCR datasets. 
This architecture has proven to be robust and stable, providing consistently good results in the experiments.

<img width="372" height="434" alt="image" src="https://github.com/user-attachments/assets/21788728-13cf-477d-af72-7d2892e538b1" />

https://arxiv.org/pdf/1507.05717

<img width="235" height="278" alt="image" src="https://github.com/user-attachments/assets/ec7d4139-db18-466b-a5ce-e24451f44659" />


#### 4.4.1.2. CAN: CNN + Attention + CTC Loss

This variant uses the same CNN backbone to extract spatial features from the input image. Instead of a BiLSTM, it incorporates a learned attention mechanism to focus on different parts of the image sequence during decoding.
The attention layer is followed by a sequence of dense layers, and the model is also trained using CTC loss. This combination allows the model to dynamically weight relevant image regions, improving interpretability and performance in cases where fixed receptive fields are limiting.


#### 4.4.1.3. Results Comparision

Training configuration:

*num_images = 25,000*
*epochs = 15*
*batch_size = 32*

<img width="1022" height="552" alt="image" src="https://github.com/user-attachments/assets/f597a001-0cb6-4692-aa72-fe415cf09a1a" />

Model Size:
- CAN has significantly fewer parameters than CRNN (~2.5M vs. ~4.5M), making it more lightweight and potentially more efficient for deployment.
  
Training Dynamics:
- Both models show rapid loss reduction within the first few epochs, indicating fast convergence.
- Training loss and validation loss decrease smoothly for both architectures, with no signs of overfitting.
- CAN exhibits a slightly lower validation loss than CRNN in early epochs, although the trend stabilizes later.

Character Error Rate (CER):
- Both models achieve a CER below 0.2, which indicates solid performance.
- CAN has a small edge in CER during initial epochs but converges to similar values as CRNN after epoch 10.

Word Accuracy:
- Word Accuracy @0 (strict match) is consistently higher for CRNN, suggesting stronger performance when exact word prediction is required.
- At Word Accuracy @1 and @2, both models converge to similar levels (~0.8–0.9), showing that they are comparably robust when minor character-level errors are tolerated.

Based on the results:
- Although both models performed well, the CRNN (CNN + BiLSTM + CTC) showed better word-level accuracy, especially for exact predictions (Word Accuracy @0).
- The CAN model (CNN + Attention + CTC), while lighter in parameters, was ultimately discarded for further experimentation due to its slightly lower precision in strict matching tasks.

As a result, the CRNN architecture was selected as the baseline for the next phase of the project.
In the next steps, the focus shifted to exploring different CNN encoders, including:
- Lightweight models like TinyCNN, and
- More advanced alternatives like Transformer-based encoders, to evaluate how the choice of visual feature extractor impacts OCR performance.

### **CRNN (Convolutional Recurrent Neural Network)**

-   **Stack:** CNN feature extractor + BiLSTM sequence model + FC head.

-   **Purpose:** Classic, widely used for OCR (scene and handwritten text).

-   **Details:** Uses several Conv2d layers, then two bidirectional LSTM layers, outputting per-timestep character logits for CTC decoding.

### **ResNet18 + BiLSTM (Hybrid)**

-   **Stack:** Pretrained ResNet18 as the feature extractor (truncated and spatially adapted) → BiLSTM → FC head.

-   **Purpose:** Leverages deeper CNNs (ResNet18) for better image features, especially with real-world or noisy images.

-   **Details:** Custom backbone class adapts ResNet18, reduces spatial stride, and adds adaptation layers. Features are collapsed and passed to BiLSTM and then FC/CTC.

| Run | Stride | Width | Freeze | Epochs | CER | Word Acc@0 | Notes |
| `RESNET18 V1` | 2 | 128 | 0 | 30 | 11.8 % | 0.700 | baseline transfer‑learning |
| `**RESNET18 V2**` | 1 | 256 | 0 | 50 | **8.62 %** | 0.710 | stride‑1, extra conv, BiLSTM 384 |

**Observations**

-   Removing final stride keeps spatial resolution → +3 pp CER vs V1.

-   No freezing outperforms early‑freeze attempts; full gradient flow crucial.

-   Balanced accuracy vs speed → good server‑side model.

### **ViT-Tiny + BiLSTM (Vision Transformer Hybrid)**

-   **Stack:** Pretrained ViT-tiny (patch-based transformer encoder for images) as the feature extractor → BiLSTM → FC head.

-   **Purpose:** Explores transformer-based global context for OCR, especially valuable on complex, distorted, or synthetic data.

-   **Details:** Handles patch resizing and positional embedding adjustment, then features are projected as sequences to BiLSTM and decoded via FC/CTC.

| Run | Width | Freeze | Aug | CER | Word Acc@0 | Notes |
| `Vit Freeze` | 128 | gradual | Baseline | 10.5 % | 0.680 | first transformer attempt |
| `Vit Aug` | 128 | 0 | Aug | 18 % | 0.419 | unstable -- over‑aug at low width |
| `Vit‑6H` | 192 | 0 | Aug | 8.9 % | 0.730 | finer patches, slower |
| `**Vit Tiny Final MAP**` | 256 | 3 epochs | MAP | **8.77 %** | 0.696 | width↑ + elastic aug |

**Insights**

-   ViT needs **≥256 px width** + elastic aug to compete with CNNs.

-   3‑epoch freeze did **not** improve validation loss; kept for reproducibility.

-   Transformers lag CNNs in FPS, but offer global context and easier multi‑language extension.


### Conclusions



## NATURAL LANGUAGE PROCESSING

### Architecture

For the natural language processing task, SpaCy will be used as a pre-built, tried and tested model for text classification. SpaCy is an open-source library for Natural Language Processing (NLP) in Python. It provides fast and efficient tools for tasks like tokenization, part-of-speech tagging, named entity recognition (NER), and text classification.

For classification, we leverage spaCy’s efficient NLP pipeline to assign a semantic label to each word, indicating whether it contains personal identifier information (PII) or not. We use the lightweight en_core_web_sm model, which provides named entity recognition (NER) capabilities suitable for detecting common types of sensitive content (e.g., names, locations, organizations). To maximize inference speed, we enable GPU acceleration when available via PyTorch, ensuring low-latency processing even at scale. Classification is performed in batches using spaCy’s nlp.pipe() method, allowing for fast, streaming inference across large datasets. The resulting entity label—if detected—is appended to the original OCR entry, preserving both the image and text information for each word. This integrated pipeline enables precise and efficient identification of sensitive content for downstream redaction or anonymization directly on document images.

### Results

Results are varied as by this part of the pipeline our words come from both cropping of the original image (Computer vision module) and the text-detection from the OCR module. 

<img width="554" height="73" alt="image" src="https://github.com/user-attachments/assets/c459734e-0f70-4f68-abfd-328532505162" />

In this example we got a true positive, where the text "paper" is classified into the "None" (not sensitive) class.

<img width="524" height="73" alt="image" src="https://github.com/user-attachments/assets/ba287542-f654-4cc7-a091-34b92720ae3a" />

In this example we got another true positive, where the name "Aanye" is classified into the "Person" (sensitive) class.

<img width="518" height="73" alt="image" src="https://github.com/user-attachments/assets/ac5ad296-172e-4f9c-b7fc-f3fd3cdeeeeb" />

There are some other subtleties to take into account, for example, the word "Will" which could either refer to a name, or the verb used to express intentions. In this case SpaCy classifies it into "None" but for a more robust model, it might be better to take context into account when classifying sensitive information. A sliding window approach with variable window size might even be worth looking into, but is out of the scope of the current project.



## BLURRING

### Architecture
### Results

## HOW TO RUN
