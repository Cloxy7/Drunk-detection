# IR-Based Drunk Detection Using Deep Learning and Facial Cues
This is a deep learning project that combines physiological and thermal face cues to detect signs of intoxication. The model uses pixel-level features and domain-specific features like Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and thermal asymmetry to improve accuracy on small datasets.

There are 9 steps to follow to go through the detection:
1. Basic setup and importing libraries
2. Upload IR images
3. Defining cue extraction functions
4. Preprocessing and landmarks extractions
5. Create a custom dataset class
6. Define the drunken model
7. Train and evaluate the model
8. Metric calcualtion
9. Visualize model predictions

# 1. Basic setup and importing libraries
To begin, install the required libraries and download the pretrained facial landmark model.
Install core libraries like dlib, torchvision, and matplotlib required for landmark detection, deep learning, and visualization.
Get the shape_predictor_68_face_landmarks.dat file used by dlib to extract 68 facial landmarks from IR images.
Load necessary Python modules and set the training device

# 2. Upload IR images from the dataset
The images are taken from the Kaggle dataset linked below -
https://www.kaggle.com/datasets/kipshidze/drunk-vs-sober-infrared-image-dataset?resource=download

After uploading the images from the dataset, define the ground truth table

1 - Drunk
0 - Sober

# 3. Defining cue extraction functions
Extract the following cues from the landmarks:
- EAR (Eye Aspect Ratio): Eye openness
- EOR (Eye Opening Ratio): Simpler eye openness ratio
- MAR (Mouth Aspect Ratio): Mouth openness (e.g., yawning)
- TAI (Thermal Asymmetry Index): Temp difference between left and right cheeks
- CNTD (Cheek-Nose Temperature Diff): Temp difference between cheeks and nose
Each is calculated using Euclidean distances between key landmarks.

# 4. Preprocessing and landmark extractions
Make sure that all the images are grayscale images and convert them to RGB and apply transformers that resize, normalize, and convert to a tensor.

# 5. Create a custom dataset class
Create a custom Dataset that returns three items per sample:
- Preprocessed image tensor (for CNN)
- Extracted 5D cue tensor (for fusion)
- Ground truth label (0 = sober, 1 = drunk)

# 6. Define Drunken Model
DrunkNet is a lightweight deep learning model designed to detect intoxication by combining:
- Visual features from IR facial images (via MobileNetV2)
- Thermal and physiological cues (EAR, EOR, MAR, TAI, CNTD)
  
It uses a pretrained MobileNetV2 as a frozen feature extractor to process IR facial images, preserving prior facial knowledge without retraining. A custom fully connected head learns from the image features, while a separate branch processes five key facial cues (EAR, EOR, MAR, TAI, CNTD). These two branches are then concatenated and passed through a final classifier to predict drunk or sober states. 

# 7. Train and evaluate the model
The model is trained using CrossEntropyLoss and the Adam optimizer, with only the custom classification layers being updated while MobileNetV2 remains frozen. During training, each batch includes both an IR image and its corresponding cue vector, passed through the model to produce predictions. After computing the loss, gradients are backpropagated, and the optimizer updates the trainable layers. Evaluation is done using accuracy, recall, and F1-score to assess both overall performance and sensitivity to impaired cases.

So the flow is like
forward pass    - get prediction
loss            - diff between actual and predicted value
backpropagation - calculate gradients using loss
optimizer       - update weights to reduce error

# 8. Metric calculation
After training, the model is evaluated using accuracy, recall, and F1-score, which reflect both the model’s correctness and its ability to detect intoxicated states accurately.

# 9. Visualize model predictions
In this final step, we visualize the model’s predictions by feeding it IR facial images and their associated cues. The image is first converted to RGB, cue values are extracted, and both are passed through the trained model in evaluation mode. The model's output is then converted into a predicted label (“Drunk” or “Sober”), which is displayed along with the original image using matplotlib. This helps validate the model’s real-world behavior in an interpretable way.















