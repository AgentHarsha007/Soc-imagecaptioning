
# Image Captioning Project

This project focuses on generating captions for images using a CNN-LSTM model. The dataset used is the Flickr 8k dataset. The following steps outline the process:

## Steps for Final Project:

1. **Importing Necessary Packages**
   - Load all required libraries and modules for data handling, model building, and training. This includes TensorFlow, Keras, NumPy, and other essential packages.

2. **Loading the Dataset**
   - Downloaded the Flickr 8k dataset, which contains 8,000 images and five captions per image. The dataset is split into training and testing sets.

3. **Text Preprocessing**
   - Clean and preprocess the text data to prepare it for model training. This includes:
     - Tokenization: Splitting captions into words.
     - Removing special characters and punctuation.
     - Converting all text to lowercase.

4. **Extracting Feature Vectors from Images**
   - Uses a pre-trained Convolutional Neural Network (CNN) model,I had used Xception to extract feature vectors from the images. This step reduces the dimensionality of the images while preserving important features.
   - ![Alt text](./delta.png)

5. **Tokenizing the Vocabulary**
   - Tokenize the captions to create a vocabulary of unique words. Convert each word to a numerical index and vice versa. This step is crucial for feeding the text data into the model.

6. **Created Data Generator**
   - Build a data generator to efficiently feed the model with batches of images and corresponding captions during training. This helps in handling large datasets without loading everything into memory at once.

7. **CNN-LSTM Model**
   - Construct the image captioning model by combining the CNN for image feature extraction and the LSTM for sequence prediction. The architecture typically includes:
     - A CNN model to process images and extract features.
     - An embedding layer to convert word indices to dense vectors.
     - An LSTM layer to process the sequence of words.
     - A dense layer to predict the next word in the sequence.
      ![Alt text](./image.png)

8. **Testing the Model**
   - Evaluate the model's performance on a test dataset by generating captions for new images and comparing them to the actual captions. Use metrics like BLEU score to quantify the model's accuracy.
## Getting Started

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd image-captioning-project
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the Flickr 8k dataset from [Flickr 8k Dataset](https://forms.illinois.edu/sec/1713398) and place it in the appropriate directory.

4. Run the project:
   ```bash
   python main.py
   ```

## Acknowledgments

- [Flickr 8k Dataset](https://forms.illinois.edu/sec/1713398)
- TensorFlow and Keras libraries
- Any other resources or tutorials that were instrumental in developing this project.

## License

Include information about the project's license if applicable.
