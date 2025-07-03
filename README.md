# IR-Based Drunk Detection Using Deep Learning and Facial Cues
This is a deep learning project that combines physiological and thermal face cues to detect signs of intoxication. The model uses pixel-level features and domain-specific features like Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and thermal asymmetry to improve accuracy on small datasets.

There are 9 steps to follow to go through the detection:
1. Basic setup and importing libraries
2. Upload IR images
3. Defining cue extraction functions
4. Preprocessing and landmarks extractions
5. Create a custom dataset class
6. Define drunken model
7. Train and evaluate the model
8. Run the full training program
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




















