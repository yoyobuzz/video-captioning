
# Video Caption Generator

## Overview  
This project focuses on developing an automatic video captioning system capable of generating natural language captions to describe events occurring within a video. The solution integrates both **Computer Vision (CV)** and **Natural Language Processing (NLP)** domains, processing sequential frames from videos and generating coherent captions that describe the content.

## Motivation  
The primary motivation behind this project was to explore the field of **multimodal learning** by working across CV and NLP domains. An effective video captioning model has multiple real-world applications, such as:  
- **Content indexing and search:** Enable keyword-based searching of videos by auto-generating captions.  
- **Financial applications:** Can indirectly support indexing and analysis of financial videos or meetings.

## Dataset  
We used the **MSVD dataset** (Microsoft Video Description Dataset) as it offers short videos (6-7 seconds) with multiple human-annotated captions. The brevity and variety of the dataset suited the scope of this project. Other datasets were evaluated but were longer or required additional preprocessing.

## Project Workflow  

1. **Data Collection**  
   - Dataset: MSVD  
   - Captions and videos were paired, creating input-output examples for training.  

2. **Data Preprocessing**  
   - **Video preprocessing:**  
     - Extracted features from video frames using a **VGG16 convolutional neural network** pre-trained on ImageNet.  
     - Features were saved for further processing.  
   - **Text preprocessing:**  
     - Tokenized and padded captions to ensure uniform input.  
     - Restricted captions to a length of 6-12 words.  
     - Created video-caption pairs for the data loader.

3. **Model Architecture**  
   - Followed a **Seq2Seq (Sequence-to-Sequence) architecture** using **LSTMs**:  
     - **Encoder:** LSTM that processes sequential video frames and extracts temporal patterns.  
     - **Decoder:** LSTM that generates natural language captions based on the encoded video features.  
     - Tokenizer: Fitted on all captions to generate the vocabulary used by the model.  
   - **Training:**  
     - Decoder inputs: Ground-truth captions were used during training.  
     - **Beam Search** was applied during inference to generate high-quality captions.  
   - **Evaluation:**  
     - Model performance was evaluated using the **BLEU score**.

4. **Challenges and Improvements**  
   - **Dataset challenges:** Deciding on the right dataset was time-consuming, and preprocessing videos was computationally intensive.  
   - **Tokenizer issues:** The initial tokenizer had limited vocabulary coverage. We improved this by increasing the vocabulary size and refining tokenization strategies.  
   - **Model architecture:** We considered using a **Recurrent Neural Network (RecNet)** architecture, but preprocessing complexity delayed its implementation. Future improvements include:  
     - Adding **embedding layers** during text preprocessing.  
     - Incorporating **attention mechanisms** in the model.  
     - Implementing RecNet or Transformer-based architectures for better results.

## Installation and Setup  
1. Clone the repository:  
   ```bash
   git clone https://github.com/yoyobuzz/video-captioning.git
   cd video-captioning
   ```

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

4. Setup the config.py file:
   ```python
   train_path = "Data/Testing Videos"
   test_path = "Data/Training Videos"
   max_length = 12
   batch_size = 10
   learning_rate = 0.0007
   epochs = 150
   latent_dim = 512
   num_encoder_tokens = 4096
   num_decoder_tokens = 1500
   time_steps_encoder = 80
   time_steps_decoder = 12
   max_probability = -1
   save_model_path = "model_final"
   validation_split = 0.15
   search_type = 'greedy'
   ```

3. Run the training script:  
   ```bash
   python train.py
   ```

4. Run the testing script to generate captions for test data:  
   ```bash
   python test.py
   ```

## Results  
The model was able to generate coherent captions for short videos, with moderate performance measured using BLEU scores. However, there is room for improvement with better preprocessing and more advanced architectures.

## Future Work  
- Add **attention mechanisms** to the Seq2Seq architecture.  
- Experiment with **Transformer-based models** for better sequence modeling.  
- Use **pre-trained embeddings** (like GloVe or Word2Vec) to improve caption quality.  
- Optimize preprocessing to speed up the video feature extraction pipeline.