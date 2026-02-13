# Fake_Ai_Generated_Image-Text_Detector
A machine learning–based system to detect AI-generated images and videos using deep learning models, helping identify synthetic and manipulated media.Built with Python, deep learning models, and a focus on real-world media authenticity.

Project Overview

AI Content Detector is a Streamlit-based web application that identifies whether an image or text is AI-generated or human-created.
The project combines computer vision, deep learning, and linguistic heuristics, with special support for Indian languages.
This project was built to address the growing challenge of deepfakes and AI-generated content, especially in digital media and online platforms.

Key Features:

🖼 Image Detection
  Detects AI-generated vs real images
  Two analysis modes:
    Heuristic-based analysis (image size, aspect ratio, color distribution)
    Deep Learning analysis using a custom CNN (PyTorch)
  Displays confidence scores and final verdict

📝 Multi-Lingual Text Detection
  Detects AI-written vs human-written text
  Supports 22+ Indian languages
  Automatic language detection
  Advanced text metrics:
  Perplexity
  Burstiness
  Syntactic complexity
  Detailed AI & human writing indicators

🌐 Multi-Language UI
  English 🇬🇧
  Hindi 🇮🇳
  Easy language switching from sidebar

🎨 Modern UI
  Custom CSS styling
  Interactive dashboards
  Clear result visualization
  Confidence progress bars
  
🧠Technologies Used
  Python  
  Streamlit – Web UI
  PyTorch – Deep learning
  TorchVision  
  PIL (Pillow) – Image processing  
  NumPy
  Regex (re) – Language detection  
  Custom CNN model

  Home Page: 
  
  ![image](https://github.com/omsalokhe/Fake_Ai_Generated_Image-Text_Detector/blob/6372a73a25d9b8dd6c5547952c396c345855feea/images/output1.png)

  
  ![image](https://github.com/omsalokhe/Fake_Ai_Generated_Image-Text_Detector/blob/6372a73a25d9b8dd6c5547952c396c345855feea/images/output2.png)

  
  ![image](https://github.com/omsalokhe/Fake_Ai_Generated_Image-Text_Detector/blob/6372a73a25d9b8dd6c5547952c396c345855feea/images/output3.png)

  
  ![image](https://github.com/omsalokhe/Fake_Ai_Generated_Image-Text_Detector/blob/6372a73a25d9b8dd6c5547952c396c345855feea/images/userinterface1.png)

  
  ![image](https://github.com/omsalokhe/Fake_Ai_Generated_Image-Text_Detector/blob/6372a73a25d9b8dd6c5547952c396c345855feea/images/userinterface2.png)


# To run the Fake AI Generated Image-Text Detector project from GitHub, follow these steps to set up the environment and launch the application on your local machine:
 
# 1. Clone the Repository
  Open your terminal or command prompt and clone the project to your computer using the following command:
  
  Bash
  git clone https://github.com/omsalokhe/Fake_Ai_Generated_Image-Text_Detector.git
  cd Fake_Ai_Generated_Image-Text_Detector
  
# 2. Set Up a Virtual Environment (Recommended)
  It is best practice to use a virtual environment to keep dependencies organized:

Windows:

  Bash
  python -m venv venv
  .\venv\Scripts\activate
  
Mac/Linux:

  Bash
  python3 -m venv venv
  source venv/bin/activate
  
# 3. Install Required Dependencies
Install the necessary Python libraries used in the project:

  Bash
  pip install streamlit torch torchvision pillow numpy

Streamlit: Powers the web interface and interactive dashboard.


PyTorch & TorchVision: Used for the custom CNN deep learning model.


Pillow (PIL): Handles image processing and analysis.


NumPy: Manages numerical operations for detection metrics.

# 4. Run the Application
Launch the Streamlit server to open the web-based interface:

Bash
streamlit run app.py
