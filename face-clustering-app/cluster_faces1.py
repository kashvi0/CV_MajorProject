# import the necessary packages
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GridSearchCV
from imutils import build_montages
import numpy as np
import argparse
import pickle
import cv2
import os
from constants import FACE_DATA_PATH, ENCODINGS_PATH, CLUSTERING_RESULT_PATH
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import face_recognition

def move_image(image, id, labelID, method):
    path = os.path.join(CLUSTERING_RESULT_PATH, 'output', method, 'labels', f'label{labelID}')
    if not os.path.exists(path):
        os.makedirs(path)
    filename = str(id) + '.jpg'
    cv2.imwrite(os.path.join(path, filename), image)
    return

def save_cluster_visualization(embeddings, labels, method, output_folder):
    # Visualize clusters using t-SNE
    plt.figure(figsize=(10, 8))
    for label in np.unique(labels):
        if label == -1:
            continue  # Skip outliers
        idx = np.where(labels == label)
        plt.scatter(embeddings[idx, 0], embeddings[idx, 1], label=f'Cluster {label}')
    plt.title(f'Cluster Visualization ({method})')
    plt.legend()
    plt.savefig(os.path.join(output_folder, f'cluster_visualization_{method}.png'))
    plt.close()  #
def main(encodings_path):
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--encodings", default=encodings_path, help="path to serialized db of facial encodings")
    ap.add_argument("-j", "--jobs", type=int, default=-1, help="# of parallel jobs to run (-1 will use all CPUs)")
    args = vars(ap.parse_args())

    # load the serialized face encodings + bounding box locations from
    # disk/encodings pickle file, then extract the set of encodings to so we can cluster on them
    print("[INFO] loading encodings...")
    data = pickle.loads(open(args["encodings"], "rb").read())
    data = np.array(data)
    encodings = [d["encoding"] for d in data]

    # Load known face encodings
    known_encodings = []
    known_names = []

    # Create the 'output' folder if it doesn't exist
    output_folder = os.path.join(CLUSTERING_RESULT_PATH, 'output')
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'dbscan'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'kmeans'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'agglomerative'), exist_ok=True)

    # define clustering techniques
    clustering_methods = {
        "dbscan": DBSCAN(metric="euclidean", n_jobs=args["jobs"])
    }
    silhouette_scores = {}

    # loop over each clustering technique
    for method, clt in clustering_methods.items():
                        # Define the parameter grid for DBSCAN
        # Define the range of hyperparameters
        eps_values = np.arange(0.1, 1.0, 0.1)  # Adjust the range of eps values as needed
        min_samples_values = range(2, 11)  # Adjust the range of min_samples values as needed

        best_score = -1
        best_params = None

        # Iterate over all combinations of hyperparameters
        for eps in eps_values:
            for min_samples in min_samples_values:
                # Instantiate DBSCAN with the current hyperparameters
                dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=args["jobs"])

                # Fit DBSCAN to the data
                clt = dbscan.fit(encodings)

                # Calculate silhouette score
                # Compute silhouette score only if there are at least two unique labels
                silhouette_avg = 0
                if len(np.unique(clt.labels_)) > 1:
                    silhouette_avg = silhouette_score(encodings, clt.labels_)
                    print(f"[INFO] Silhouette score for {method}: {silhouette_avg:.3f}")
                else:
                    print("[INFO] Skipping silhouette score calculation due to insufficient number of clusters.")


                # Check if this combination of hyperparameters yields a better silhouette score
                if silhouette_avg > best_score:
                    best_score = silhouette_avg
                    best_params = {'eps': eps, 'min_samples': min_samples}

        # Use the best hyperparameters to fit DBSCAN
        best_dbscan = DBSCAN(eps=best_params['eps'], min_samples=best_params['min_samples'], metric="euclidean", n_jobs=args["jobs"])
        clt = best_dbscan.fit(encodings)

        print(f"[INFO] Best DBSCAN hyperparameters: {best_params}")
        print(f"[INFO] Best silhouette score: {best_score:.3f}")

        # Now, you can proceed with clustering using the best DBSCAN estimator



        labelIDs = np.unique(clt.labels_)
        num_clusters = len(np.where(labelIDs > -1)[0])
        print(f"[INFO] # unique faces using {method}: {num_clusters}")

        # Create the 'montages' and 'labels' folders if they don't exist
        output_method_folder = os.path.join(output_folder, method)
        os.makedirs(output_method_folder, exist_ok=True)
        montages_folder = os.path.join(output_method_folder, 'montages')
        os.makedirs(montages_folder, exist_ok=True)
        labels_folder = os.path.join(output_method_folder, 'labels')
        os.makedirs(labels_folder, exist_ok=True)

        for labelID in labelIDs:
            print(f"[INFO] Processing label ID: {labelID} using {method}")
            label_folder_path = os.path.join(labels_folder, f'label{labelID}')
            os.makedirs(label_folder_path, exist_ok=True)
            idxs = np.where(clt.labels_ == labelID)[0]
            faces = []
            for i in idxs:
                print(f"[INFO] Processing image: {i} using {method}")
                image = cv2.imread(data[i]["imagePath"])
                (top, right, bottom, left) = data[i]["loc"]
                face = image[top:bottom, left:right]
                move_image(image, i, labelID, method)
                face = cv2.resize(face, (96, 96))
                faces.append(face)

                # Recognize face
                encoding = encodings[i]
                matches = face_recognition.compare_faces(known_encodings, encoding)
                if any(matches):
                    matched_indices = [i for (i, match) in enumerate(matches) if match]
                    names = [known_names[i] for i in matched_indices]
                    print(f"[INFO] Image {i} matched known individuals: {', '.join(names)}")
                else:
                    print(f"[INFO] Image {i} did not match any known individuals")

            montage = build_montages(faces, (96, 96), (5, 5))[0]
            montage_path = os.path.join(montages_folder, f'montage_label{labelID}.jpg')
            cv2.imwrite(montage_path, montage)
            print(f"[INFO] Montage saved for label ID: {labelID} using {method}")

        # Compute silhouette score
        silhouette_avg = silhouette_score(encodings, clt.labels_)
        print(f"[INFO] Silhouette score for {method}: {silhouette_avg:.3f}")
        silhouette_scores[method] = silhouette_avg

        # # Visualize clusters using t-SNE
        
        tsne = TSNE(n_components=2, random_state=42)
        embeddings = tsne.fit_transform(np.array(encodings))
        save_cluster_visualization(embeddings, clt.labels_, method, output_method_folder)
        # plt.figure(figsize=(10, 8))
        # for label in np.unique(clt.labels_):
        #     if label == -1:
        #         continue  # Skip outliers
        #     idx = np.where(clt.labels_ == label)
        #     plt.scatter(embeddings[idx, 0], embeddings[idx, 1], label=f'Cluster {label}')
        # plt.title(f'Cluster Visualization ({method})')
        # plt.legend()
        # plt.savefig(os.path.join(output_method_folder, 'cluster_visualization.png'))
        # plt.show()

    # Define KMeans and AgglomerativeClustering based on the number of clusters determined by DBSCAN
    print("number of clusters: ", num_clusters)
    clustering_methods = {
        "kmeans": KMeans(n_clusters=num_clusters),
        "agglomerative": AgglomerativeClustering(n_clusters=num_clusters, linkage="ward")
    }

    # Loop over the remaining clustering techniques
    for method, clt in clustering_methods.items():
        output_method_folder = os.path.join(output_folder, method)
        os.makedirs(output_method_folder, exist_ok=True)
        montages_folder = os.path.join(output_method_folder, 'montages')
        os.makedirs(montages_folder, exist_ok=True)
        labels_folder = os.path.join(output_method_folder, 'labels')
        os.makedirs(labels_folder, exist_ok=True)

        print(f"[INFO] clustering using {method}...")
        clt.fit(encodings)

        labelIDs = np.unique(clt.labels_)
        numUniqueFaces = len(np.where(labelIDs > -1)[0])
        print(f"[INFO] # unique faces using {method}: {numUniqueFaces}")
        for labelID in labelIDs:
            print(f"[INFO] Processing label ID: {labelID} using {method}")
            label_folder_path = os.path.join(labels_folder, f'label{labelID}')
            os.makedirs(label_folder_path, exist_ok=True)
            idxs = np.where(clt.labels_ == labelID)[0]
            faces = []
            for i in idxs:
                print(f"[INFO] Processing image: {i} using {method}")
                image = cv2.imread(data[i]["imagePath"])
                (top, right, bottom, left) = data[i]["loc"]
                face = image[top:bottom, left:right]
                move_image(image, i, labelID, method)
                face = cv2.resize(face, (96, 96))
                faces.append(face)

            montage = build_montages(faces, (96, 96), (5, 5))[0]
            montage_path = os.path.join(montages_folder, f'montage_label{labelID}.jpg')
            cv2.imwrite(montage_path, montage)
            print(f"[INFO] Montage saved for label ID: {labelID} using {method}")

        # Compute silhouette score
        silhouette_avg = silhouette_score(encodings, clt.labels_)
        print(f"[INFO] Silhouette score for {method}: {silhouette_avg:.3f}")
        silhouette_scores[method] = silhouette_avg

        tsne = TSNE(n_components=2, random_state=42)
        embeddings = tsne.fit_transform(np.array(encodings))
        save_cluster_visualization(embeddings, clt.labels_, method, output_method_folder)

        # plt.figure(figsize=(10, 8))
        # for label in np.unique(clt.labels_):
        #     if label == -1:
        #         continue  # Skip outliers
        #     idx = np.where(clt.labels_ == label)
        #     plt.scatter(embeddings[idx, 0], embeddings[idx, 1], label=f'Cluster {label}')
        # plt.title(f'Cluster Visualization ({method})')
        # plt.legend()
        # plt.savefig(os.path.join(output_method_folder, 'cluster_visualization.png'))
        # plt.show()
    return silhouette_scores

if __name__ == "__main__":
    main()