import streamlit as st
from cluster_faces1 import main as cluster_main
from PIL import Image
import os

# Function to run clustering and generate montages
def run_clustering_and_montages(encodings_path):
    return cluster_main(encodings_path)

# Function to display uploaded images in a gallery view
def display_uploaded_images(image_paths):
    st.subheader("Uploaded Images")

    # Determine number of columns in the grid
    num_columns = 5

    # Calculate number of rows needed
    num_images = len(image_paths)
    num_rows = (num_images - 1) // num_columns + 1

    # Get the position of the slider
    window_start = st.sidebar.slider("Scroll to view images:", 0, max(0, num_images - num_columns), 0)

    # Create a container with a fixed height and horizontal scroll
    with st.container():
        st.write('<div style="overflow-x: auto; display: flex;">', unsafe_allow_html=True)

        for i in range(num_rows):
            images_in_row = image_paths[i * num_columns: (i + 1) * num_columns]
            st.image([Image.open(image_path) for image_path in images_in_row], caption=[f"Image {index + 1}" for index in range(len(images_in_row))], width=150)

        st.write('</div>', unsafe_allow_html=True)


# Function to display silhouette scores for each clustering method
def display_silhouette_scores(silhouette_scores):
    st.subheader("Silhouette Scores")
    for method, score in silhouette_scores.items():
        st.write(f"{method.capitalize()}: {score}")

# Main function for Streamlit app
def main():
    st.title("Facial Clustering App")

    # File uploader for user to upload images
    uploaded_files = st.file_uploader("Upload images", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

    if uploaded_files:
        # Create a directory to store uploaded images
        os.makedirs('uploaded_images', exist_ok=True)

        # Save the uploaded images to the directory
        image_paths = []
        for uploaded_file in uploaded_files:
            with open(os.path.join("uploaded_images", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getvalue())
            image_paths.append(os.path.join("uploaded_images", uploaded_file.name))

        # Display uploaded images in a gallery view
        display_uploaded_images(image_paths)

        # Button to start clustering
        if st.button("Cluster Images"):
            # Run clustering and generate montages
            st.text("Clustering images...")
            silhouette_scores=run_clustering_and_montages(encodings_path="encodings.pickle")
            st.text("Clustering complete!")

            # Display silhouette scores
            # silhouette_scores = {"dbscan": 0.75, "kmeans": 0.82, "agglomerative": 0.79}  # Placeholder scores
            display_silhouette_scores(silhouette_scores)

            # Button to explore clustering results
                    # Button to explore clustering results
        # Button to explore clustering results
        st.subheader("Explore Clustering Results")
        for method in ["DBSCAN", "KMeans", "Agglomerative"]:
            with st.expander(method):
                st.subheader(f"{method} Clusters")
                montages_folder = os.path.join("output", method.lower(), "montages")
                if os.path.exists(montages_folder):
                    image_files = [file for file in os.listdir(montages_folder) if file.endswith(".jpg")]
                    if image_files:
                        selected_image = st.select_slider(f"Select Image ({method})", options=image_files)
                        selected_image_path = os.path.join(montages_folder, selected_image)
                        img = Image.open(selected_image_path)
                        st.image(img, caption=selected_image, use_column_width=True)

                # Display cluster visualization plots
                visualization_path = os.path.join("output", method.lower(), f"cluster_visualization_{method.lower()}.png")
                if os.path.exists(visualization_path):
                    st.subheader(f"{method} Cluster Visualization")
                    st.image(visualization_path, caption=f"{method} Cluster Visualization", use_column_width=True)


if __name__ == "__main__":
    main()
