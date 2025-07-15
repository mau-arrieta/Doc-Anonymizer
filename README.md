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

  
![Alt text](https://github.com/mau-arrieta/Doc-Anonymizer/blob/main/doc-images/General%20Pipeline%20Concept.png)


## COMPUTER VISION

### 3.1. Hypothesis

It is possible to accurately detect and localize individual words in document images by learning visual patterns within these documents

We expect that a pre-trained detection model can be fine-tuned for our use case, and learn how to distinguish word regions from non-text background across a wide variety of documents. Achieving robust word-level detection will enable downstream models such as the OCR to correctly predict text from our image crops. 

Furthermore, the module is designed with a Faster R-CNN based model for out text detection task, as its two-stage design (region proposal + classification) makes it especially strong at precise localization, which is critical for detecting small, tightly packed word regions that simpler one-stage detectors often miss.


### 3.2. Dataset

The dataset used for the word-box detection task is the [DocBank dataset](https://doc-analysis.github.io/docbank-page/). It was originally designed for textual and document layout tasks, and the version used for this project consists of 500K document pages, along with their corresponging annotations.

Annotations are based on 12 labels, which correspond to each area of the document layout:

![Alt text](https://github.com/mau-arrieta/Doc-Anonymizer/blob/main/doc-images/docbankLayout.png)


These labels are then used to classify the annotations into these 12 classes. This data along with the bounding box pixel coordinates are included in the annotation files. 


![Alt text](https://github.com/mau-arrieta/Doc-Anonymizer/blob/main/doc-images/docbankAnnotations.png)


The pixel coordinates for these bounding boxes is crucial for the project, as they determine wether the model is going to learn valuable information or if it's simply going to underfit for our task. According to [Docbank's official github](https://github.com/doc-analysis/DocBank/tree/master), these coordinates were normalized to 1000xx1000 grid, so we use the same normalization to maintain coherence with the dataset.


A visualization of the ground-truth bounding boxes with their correct normalization is shown next: 

![Alt text](https://github.com/mau-arrieta/Doc-Anonymizer/blob/main/doc-images/ground-truth-labels.png)


The annotations included tokens our model didn't need to learn for this task (symbols, formatting sequences), so we filtered our annotation files by matching a REGEX which left us only those tokens that included words longer (or equal to) three letters, and that were made up of letters in the english alphabet (including those with hyphens and apostrophes).

We also tried to filter the annotations by keeping the annotations that we needed and grouping them into a single label. So it would be easier for the model to distribute the weights on the learning process, since it doesn't have the same weight to detect a element with label paragraph than caption. At the end, we couldn't apply said change over the annotations from DocBank and we had to train it but we took it into account on the metrics evaluation.

The whole dataset was approximately 50gb, our model used 2.4% of it for fine-tuning due to storage constraints the team worked with. Our final version was fine-tuned with a total of 12k document images, with a 80-10-10 data partition for training,validation and testing, in that order.

### 3.3. Architecture

Our implemented architecture is based on the Faster R-CNN framework for object detection, which integrates three main components:

1. **Feature extraction backbone: ResNet-18**.   We use ResNet-18 as the backbone convolutional neural network. ResNet-18 consists of an initial convolution + pooling layer followed by 4 residual blocks. Each residual block contains convolutional layers with skip connections (adding input directly to output), which stabilizes training even with increased depth. The backbone outputs a high-level feature map, typically 1/32 of the original input resolution, but rich in semantic information. We chose ResNet-18 as the backbone because it’s lightweight, ensuring faster training and inference, which is critical given the high-resolution nature of document images and the residual connections allow training deeper networks efficiently, but ResNet-18 is deep enough to capture important hierarchical document features (lines → paragraphs → figures).

2. **Region Proposal Network (RPN)**. The RPN takes the feature map from ResNet-18 and slides small 3x3 convolutions across it. It predicts: objectness scores, whether each anchor (small sliding window) likely contains an object, and bounding box deltas, adjustments to anchors to better fit objects. The RPN uses multiple anchors at each position (different scales and aspect ratios), generating thousands of candidate regions (ROIs).

3. **ROI Heads (Classifier + Regressor)**. The top N proposals from the RPN are then processed by ROI Pooling/ROI Align which consists on crops + resizes each proposed region from the feature map to a fixed size (e.g., 7x7). A series of fully connected layers (MLP) predict: Class scores (multi-class softmax over our labels like "title", "paragraph", "figure", etc) and bounding box refinements for each predicted class.

#### 3.3.1 How it all connects
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Input image

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ResNet-18 backbone

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|──▶ RPN ───────────────────▶ proposals (regions of interest)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ROI&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Align ROI Classifier + Regressor

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Feature crops&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Class scores & refined boxes

#### 3.3.2 Architectural decisions & customitzation
We chose ResNet-18 as the backbone because it's lightweight, ensuring faster training and inference, which this key point is critical taking into consideration the high-resolution nature of document images. 

Previously we tried to train it with a backbone of ResNet-50 and the performance leaved much to desire in comparisson to this model. With the same amount of data ResNet-18 outperformed ResNet-50 on this use case. To be able to determine it, we trained with both backbones and saw a significant difference in the losses on each backbone version.

On both we used a pretrained backbone on ImageNet, so it would accelerate the training process by leveraging features learned on large datasets. Then we fine-tuned all layers (not frozen), so the network could adapt to the low-level features (edge cases) to document-specific patterns.


#### 3.3.3 RPN & ROI customization
Since document layouts vary so much in element size and shape, it's critical that the RPN anchors cover multiple scales and aspect ratios, so that small lines of text and large table can both be proposed. The ROI heads, on another hand, can accurately classify each region into our document-specific categories and refine the bounding boxes for pixel-level alignment. 

We customized the anchor sizes of the RPN so it ensures that this head covers very small to very large object, which affects over our dataset since in documents there are tiny author names or equations and large tables or figures. Also, the aspect ratios have been customized so each scale supports tall, square, and wide objects.

We have increased proposals (rpn_pre_nms_top_n_train, rpn_post_nms_top_n_train, rpn_pre_nms_top_n_test and rpn_post_nms_top_n_test) significantly higher than defaults to handle the many text regions per page.

We used MultiScaleRoIAlign on 4 levels, which combines different feature map scales to handle multi-sized objects.


### 3.4. Metrics

Faster R-CNN breaks the text detection problem into two stages: finding possible text regions and classifying them. The losses reflect each of these tasks — whether it's finding a text region at all (train_loss_objectness), correctly identifying it as text (train_loss_classifier), or drawing the box in the right place (train_loss_box_reg). Together, these make up the total training loss, helping us monitor how well the model is learning to detect text.

![Alt text](https://github.com/mau-arrieta/Doc-Anonymizer/blob/main/doc-images/resnet-18-loss-metrics.png)

In this case we saw that the train loss is around 0.46 and the validation loss is around 0.46, which indicates that the model on the 10th epoch has a valid result over the training with the stablished hyper parameters.

After that we analyzed the metrics of the trained model. We have perceived that by class there is a certain relationship on the amount of times the type has been registered in the annotations and it's value on the metrics. In this case, paragraph it's clearly the label with most appearances and best metrics:

![Alt text](https://github.com/mau-arrieta/Doc-Anonymizer/blob/main/doc-images/resnet-18-class-metrics.png)

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

### 4.4.2  Multiple Arquitecture -CTC

### Why **CRNN → ResNet-18 → ViT-Tiny** were chosen

| Family                     | Concrete model we use                               | Why it belongs in the benchmark                                                                                                                                                                                                                                                                                                         | Key references in the labs                                                                                                              |
| -------------------------- | --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **Recurrent-CNN baseline** | **CRNN** (small custom CNN + 2 × Bi-LSTM)           | *Historical anchor.*<br>• Purely “from-scratch”, no ImageNet weights → shows what the dataset itself can teach the network.<br>• Extremely light (<1 M params) and easy to train/interpret.          | *lab\_mlp\_cnn* → basic CNN building blocks.<br>*lab\_rnn* → Bi-LSTM sequencing.
| **Transfer-learning CNN**  | **ResNet-18** (pre-trained, frozen-then-fine-tuned) | *Strong classical baseline.*<br>• Convolutional features are well matched to local stroke patterns in text.<br>• ImageNet pre-training means we need < fewer labelled lines to reach low CER.<br>• Still modest size (≈11 M params), fast on Colab GPUs.                                                                                | *lab\_transfer\_learning* shows how to freeze, then unfreeze.<br>*lab\_contrastive* motivates using powerful encoders with small heads. |
| **Vision Transformer**     | **ViT-Tiny Patch16** (pre-trained)                  | *Modern research variant.*<br>• Self-attention can capture long-range relations → good for long text strings and wide aspect ratios.<br>• Patch-wise design simplifies to a 1-D sequence → meshes naturally with CTC after minimal reshaping.<br>• ViT-Tiny (192-dim, 12 layers) keeps GPU memory within Colab limits, unlike ViT-Base. | *lab\_diffusion* & *NLP Transformer labs* demonstrate positional embedding tricks we re-used (interpolating PE).                        |

_We retained the CRNN baseline for continuity and did two deliberate upgrades:_

*   **ResNet-18** to leverage transfer-learning from natural-image CNNs while keeping a lightweight model.
    
*   **ViT-Tiny** to test whether modern transformer backbones improve OCR on wide text lines; we added positional-embedding interpolation so it runs at 32×160 inputs.
    

These three checkpoints give us a clear picture of **classical vs. transfer-learning vs. transformer** performance on the exact same dataset and training script.

#### 4.4.2.1  CRNN Family  — CNN + BiLSTM + CTC

Our CRNN baseline follows the classical pipeline proposed by Shi et al. (2017):

1. **CNN Feature Extractor**    
   5 convolutional blocks (64 → 512 channels) with interleaved BatchNorm, ReLU and 2×2 max-pooling.  
   Output feature map: **[B, 512, H=1, W]**.

2. **Sequence Mapper**    
   The height axis is squeezed (H = 1), leaving a width-wise sequence **[W, 512]** that represents the word from left to right.

3. **Bidirectional LSTM × 2**    
   Hidden = 256 per direction → output dim = 512.

4. **Classification Head**    
   Linear layer → logits of size *(vocab + blank)* at every time-step.

5. **Loss**    
   Connectionist Temporal Classification (CTC) with blank = 62, ignoring PAD = 63.

#### 4.4.2.2  Experiments

| Run (wandb) | Image Width | Augmentation | Batch | Max LR | Epochs | Notes |
|-------------|------------|--------------|-------|--------|--------|-------|
| **CRNN Baseline** | 128 px | Baseline | 128 | 1 e-3 | 30 | first scratch model |
| **crnn Final MAP** | 192 px | *MAP* (elastic) | 64 | 1 e-3 | 30 | +dropout 0.20, One-Cycle LR |

Key tweaks that closed the gap:

* **Wider input (128 → 192 px)** — doubles the LSTM time-steps, giving more context per word.
* **Elastic-distortion augmentation** — combats synthetic-to-real domain gap.
* **Dropout 0.20** on LSTM outputs — mitigates over-fitting without hurting convergence.

#### 4.4.2.3  Results & Discussion (22 k test set)

| Metric | CRNN Baseline | **crnn Final MAP** |
|--------|---------------|--------------------|
| Character Error Rate (↓) | 15.20 % | **6.72 %** |
| Word Accuracy @0 (↑) | 0.580 | **0.766** |
| Word Accuracy @1 (↑) | 0.710 | 0.880 |
| Word Accuracy @2 (↑) | 0.780 | **0.926** |
| Inference FPS (BS=1) | 250 | **250** |

*The final CRNN nearly halves the CER of the baseline while keeping its class-leading speed (≈ 250 fps on an RTX 4000).*

> **Take-away:** A well-regularised, width-scaled CRNN is hard to beat for word-level OCR when inference speed and model size matter. The gains justify retaining it as the reference point for the ResNet-18 and ViT-Tiny explorations.


#### 4.4.2.2  ResNet-18 with Deep CNN Backbone

We swap the 5-conv encoder of the CRNN for a **transfer-learned ResNet-18** to test whether deeper convolutional features boost recognition:

1. **ResNet-18 Backbone**  
   * Layers 0–3 kept as in ImageNet pre-train.  
   * **Stride fix:** the final down-sampling in *layer4* is removed (stride 1 instead of 2) to preserve horizontal resolution.  
   * One extra 3 × 3 conv (512 → 512) + ReLU refines the feature map.

2. **Sequence Mapper**  
   Height squeezed → sequence **[W, 512]** (≈ 64 time-steps with 256 px input).

3. **BiLSTM × 2**  
   Hidden = **384** per direction (larger than CRNN to exploit richer features).

4. **CTC Head & Loss**  
   Same as CRNN (vocab + blank).

<div align="center">
  <!-- schematic omitted; identical head to CRNN -->
</div>

##### Experiments

| Run (wandb) | Stride | Image Width | Freeze | Epochs | CER ↓ | Word Acc@0 ↑ | Notes |
|-------------|--------|------------|--------|--------|-------|--------------|-------|
| **RESNET18 V1** | 2 | 128 px | none | 30 | 11.8 % | 0.700 | first TL baseline |
| **RESNET18 V2** | **1** | **256 px** | none | 50 | **8.62 %** | **0.710** | stride-1 + wider + BiLSTM 384 |

Key findings  
* Removing the last stride **adds ×2 time-steps** yet only +2 MB params.  
* Wider input (256 px) further lowers CER (~3 pp).  
* Early experiments with **3-epoch freezing** hurt convergence → final model keeps **all layers trainable** from the start with a discriminative LR (1e-3 head / 3e-4 backbone).

##### Results & Discussion (22 k test set)

| Metric | RESNET18 V1 | **RESNET18 V2** |
|--------|-------------|-----------------|
| Character Error Rate (↓) | 11.80 % | **8.62 %** |
| Word Accuracy @0 (↑) | 0.700 | **0.710** |
| Word Accuracy @1 (↑) | 0.794 | **0.846** |
| Word Accuracy @2 (↑) | 0.833 | **0.914** |
| Inference FPS (BS = 1) | 190 | **175** |

*ResNet18-V2 slices CER by **~27 %** over V1 and narrows the gap with CRNN to 1.9 pp, trading ~30 % inference speed.  
Its balanced accuracy vs speed makes it the recommended CNN backbone when hardware allows a slightly heavier model.*


#### 4.4.2.3  ViT-Tiny Vision-Transformer Backbone

To test non-convolutional feature extractors we replaced the CNN encoder with **ViT-Tiny** (Dosovitskiy et al., 2021):

1. **Patch Embedding**  
   * 16 × 16 patches → 192-D embeddings.  
   * With a 256 px-wide image this yields **16 × 16 = 256 tokens** (plus CLS).

2. **Transformer Encoder**  
   12 layers, 12 heads, GELU activations.  
   Pre-trained on ImageNet-21k, fine-tuned end-to-end.

3. **Sequence Mapper**  
   CLS is discarded; the remaining patch tokens are reshaped into a **[W, 192]** sequence (left→right order).

4. **BiLSTM × 2**  
   Hidden = 256 each direction (kept small to offset ViT cost).

5. **CTC Head & Loss**  
   Same as CRNN/ResNet (vocab+blank).

##### Experiments

| Run (wandb) | Image Width | Freeze | Augmentation | CER ↓ | Word Acc@0 ↑ | Notes |
|-------------|-------------|--------|--------------|-------|--------------|-------|
| `Vit Freeze` | 128 px | gradual (layer-wise) | Baseline | 10.5 % | 0.680 | first stabilisation attempt |
| `Vit Aug` | 128 px | none | strong | 18 % | 0.419 | over-augmented, unstable |
| `Vit-6H` | 192 px | none | strong | 8.9 % | 0.730 | patch 4, 6 heads, slower |
| **`Vit Tiny Final MAP`** | **256 px** | **first 3 epochs** | **MAP (elastic)** | **8.77 %** | **0.696** | best transformer balance |

Tuning insights  

* **Width matters** — 256 px input halves CER vs 128 px.  
* **Early freeze (3 epochs)** did **not** lower over-fit but avoided initial divergence; kept for reproducibility.  
* Patch-4, 6-head variant improves strict accuracy (0.730) but drops FPS (115 → 130).

##### Results & Discussion (22 k test set)

| Metric | Vit Freeze | Vit-6H | **Vit Tiny Final** |
|--------|------------|--------|--------------------|
| Character Error Rate (↓) | 10.50 % | 8.90 % | **8.77 %** |
| Word Accuracy @0 (↑) | 0.680 | **0.730** | 0.696 |
| Word Accuracy @1 (↑) | 0.800 | 0.830 | **0.839** |
| Word Accuracy @2 (↑) | 0.860 | 0.890 | **0.906** |
| Inference FPS (BS=1) | 140 | 115 | **130** |

*ViT-Tiny Final closes to within **0.15 pp CER** of ResNet18-V2 while retaining 74 % of CRNN’s speed.  
Transformers remain more data-hungry but become competitive once image width and elastic augmentation are raised.*

---

#### 4.4.2.4  Results Comparison

| Model (22 k test set) | Params | CER ↓ | Word Acc @0 ↑ | Word Acc @1 ↑ | Word Acc @2 ↑ | Inference FPS* |
|-----------------------|--------|-------|---------------|---------------|---------------|----------------|
| **CRNN-Final**        | 7 M    | **6.72 %** | **0.766** | 0.880 | **0.926** | **250** |
| **ResNet18-V2**       | 26 M   | 8.62 % | 0.710 | **0.846** | 0.914 | 175 |
| **ViT-Tiny Final**    | 30 M   | 8.77 % | 0.696 | 0.839 | 0.906 | 130 |

**Key take-aways**

1. **Accuracy vs Speed**  
   *CRNN-Final* delivers the **lowest CER** and highest word accuracy while maintaining the **fastest inference** (≈ 250 fps).  
   *ResNet18-V2* trails by ~1.9 pp CER but gains +6 pp Word Acc @1 over CRNN, offering a balanced server-side option.  
   *ViT-Tiny Final* matches ResNet on character-level accuracy yet remains slower and slightly behind on strict word matches.

2. **Model Capacity**  
   The two transfer-learning backbones (ResNet, ViT) require ≈ 4–5× more parameters than CRNN. Gains plateau once width = 256 px and BiLSTM ≥ 256 hidden.

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
