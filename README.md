# Facial Emotion Detection using CNN (FER2013 + RAF-DB)

This project focuses on building a Convolutional Neural Network (CNN)-based model for facial emotion recognition by combining two benchmark datasets: **FER2013** and **RAF-DB**. The model is trained to classify facial expressions into seven emotion categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

## 🧠 Key Features

- ✅ Merged dataset training using **FER2013** and **RAF-DB**
- ✅ Preprocessing with grayscale resizing and normalization
- ✅ CNN architecture with Batch Normalization, Dropout, and Data Augmentation
- ✅ Early stopping and model checkpointing for efficient training
- ✅ Visualization of Confusion Matrix and classification performance metrics

## 📁 Emotion Classes

- Angry  
- Disgust  
- Fear  
- Happy  
- Neutral  
- Sad  
- Surprise

## 🧰 Technologies & Libraries Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy, Matplotlib, Seaborn
- Scikit-learn (for evaluation)
- ImageDataGenerator (for augmentation)

## 📦 Dataset Info

- **FER2013**: Pre-sorted by emotion category, grayscale images resized to 48x48.  
- **RAF-DB**: Additional labeled emotion dataset to improve diversity and accuracy.  
- Both datasets are manually loaded and preprocessed before training.

## 📊 Model Architecture (CNN)

- 3 Convolutional + MaxPooling blocks with ReLU & BatchNormalization
- Flatten → Dense (512 units) → Dropout (0.5)
- Final Softmax layer for 7-class classification

## 🚀 Training Highlights

- Optimizer: Adam  
- Loss Function: Categorical Crossentropy  
- Batch Size: 64  
- Epochs: 60  
- Callbacks: `EarlyStopping`, `ModelCheckpoint`  

## 📈 Evaluation

- Accuracy Score on Test Set  
- Confusion Matrix with Seaborn  
- Classification Report (Precision, Recall, F1-Score)

## 🖼️ Sample Visualization (Confusion Matrix)
[

![Image](https://github.com/user-attachments/assets/9b34fc8e-a76b-4d56-9604-6f9bb5dc08ec)

](url)




