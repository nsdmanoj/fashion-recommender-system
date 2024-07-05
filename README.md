Fashion Recommender System

Overview
This repository contains a fashion recommender system designed to help users discover visually similar fashion items based on uploaded images. The system utilizes deep learning techniques for feature extraction and similarity detection, making use of a pre-trained ResNet50 model and the k-nearest neighbors algorithm.

Key Features
Advanced Feature Extraction: The system leverages the ResNet50 convolutional neural network, pre-trained on the ImageNet dataset, to extract high-level features from fashion images. These features capture important visual characteristics such as patterns, textures, and shapes.

Efficient Recommendation Engine: Recommendations are generated using the k-nearest neighbors algorithm, which efficiently matches the features of uploaded images with precomputed features of fashion items in the dataset. This allows for real-time recommendations based on visual similarity.

Interactive Web Interface: Built with Streamlit, the system provides an intuitive web interface where users can upload images of fashion items. Upon upload, the system displays visually similar fashion items along with their details and links for further exploration.

Precomputed Data: To optimize performance, precomputed feature vectors (embeddings.pkl) and corresponding filenames (filenames.pkl) are used. These files are essential for quick retrieval and comparison of features during the recommendation process.

Scalability and Adaptability: The system is designed to be scalable, allowing for easy integration with different fashion datasets and adaptable to various fashion recommendation use cases. It provides a foundation that can be extended and customized to suit specific needs.
Usage
Run the Application:

code
streamlit run app.py
This command starts the Streamlit server and launches the web interface.

Upload an Image:
Use the file uploader in the web interface to upload an image of a fashion item.
![Screenshot 2024-07-03 185857](https://github.com/nsdmanoj/fashion-recommender-system/assets/114307491/08afb754-8da2-4135-bdef-0280fad6643e)


View Recommendations:
Once uploaded, the system processes the image, extracts its features using ResNet50, and displays a list of visually similar fashion items.

Explore Recommendations:
Click on recommended images to view details such as product information, prices, and links to purchase or learn more.
