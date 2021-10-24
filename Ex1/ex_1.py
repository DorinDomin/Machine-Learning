import numpy as np
import scipy.io.wavfile
import sys

# print compressed file
def printComp(clusters,new_centers,fs):
    new_values = []
    for c in clusters:
        new_values.append(new_centers[c])
    scipy.io.wavfile.write("compressed.wav", fs, np.array(new_values, dtype=np.int16))
# main
def main():
    # reading
    sample, centroids = sys.argv[1], sys.argv[2]
    centroids = np.loadtxt(centroids)
    itr = 0
    convergence = False
    # reading
    fs, y = scipy.io.wavfile.read(sample)
    # training set
    x = np.array(y.copy())
    # open output file
    output = open(f"output.txt", "w")
    training_size = x.shape[0]
    num_of_clusters = centroids.shape[0]
    # centroids centers initialization
    new_centers = np.array(centroids.copy())
    prev_centers = np.zeros(num_of_clusters)
    clusters = np.zeros((num_of_clusters, training_size))
    x_distance_calc = np.zeros((training_size, num_of_clusters))
    while itr < 30 and not convergence:
        # closest centroid assignment
        for i in range(training_size):
            x_distance_calc[i, :] = np.linalg.norm(x[i] - new_centers, axis=1)
        clusters = np.argmin(x_distance_calc, axis=1)
        # saving for convergence checking
        prev_centers = np.array(new_centers.copy())
        # update centers
        for i in range(num_of_clusters):
            # get all points of a cluster
            arr = np.where(clusters == i)
            # no change in the position of the centroid
            if (len(arr[0]) == 0):
                new_centers[i] = prev_centers[i]
            # update position
            else:
                new_centers[i] = np.round(np.mean(x[arr[0]], axis=0))
                # print itr
        output.write(f"[iter {itr}]:{','.join([str(i) for i in new_centers])}\n")
        itr += 1
        # check for convergence
        if(np.linalg.norm(new_centers - prev_centers)==0):
            convergence = True
    output.close()
    # saving
    printComp(clusters,new_centers,fs)
if __name__ == "__main__":
    main()