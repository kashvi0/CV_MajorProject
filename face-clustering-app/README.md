# Face Clustering App

Welcome to the Face Clustering App! This application allows you to cluster and visualize faces detected in images using various clustering algorithms.

## Features

- Upload images containing faces.
- Choose from multiple clustering algorithms (DBSCAN, KMeans, Agglomerative).
- View clustered faces in montages.
- Explore cluster visualization plots.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Arun-Raghav-S/face-clustering
    ```

2. Navigate to the project directory:

    ```bash
    cd face-clustering
    ```

3. Create and activate a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Unix/Linux/Mac
    venv\Scripts\activate      # On Windows
    ```

4. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

5. **Additional Setup (Optional)**:
    - To generate new facial encodings, run:
      ```bash
      python encodings.py
      ```
    - To specify a different dataset name, use the `--dataset` flag.

6. **Changing Face Detection Model**:
    - You can switch between face detection models by modifying the code in `cluster_faces.py`.

7. **Note**: Ensure that the dlib wheel file used is compatible with Python version 11.0. Download the corresponding wheel file from [here](https://github.com/z-mahmud22/Dlib_Windows_Python3.x).

## Usage

1. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

2. Access the app in your web browser at [http://localhost:8501](http://localhost:8501).

3. Upload images containing faces and explore the clustering results!

## Folder Structure

- `app.py`: Main application file containing the Streamlit app code.
- `cluster_faces.py`: Script for clustering faces in images.
- `requirements.txt`: File containing the dependencies required to run the app.
- `encodings.pickle`: Serialized file containing facial encodings.
- `output/`: Directory containing clustering results, including montages and cluster visualization plots.

