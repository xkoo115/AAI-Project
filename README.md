# AAI-Project
Project files for AAI project.

We performed with processing operations on the data, and the processed data is saved in the data file.

We performed an FFT transform on the audio file and kept half of the data (according to the FFT symmetry, the half part already contains all the features).

In order to ensure the data balance, we randomly one noise for each original data, to ensure the ratio of 1:1 between the noisy data and the original data, and all the original data are covered to ensure its maximum diversity.

We have saved the pre-trained model in Recogniser.pt with the device cuda. Our pre-trained model obtained an accuracy of 88.8% on the validation set.

We have saved the results of test in out_test.txt and the results of test_noisy in out_test_noisy.txt.

LibriSpeech-SI is used for the dataset.

The pre-processed data is saved in data/noise and data/origin by default, and these two folders need to be created in advance.
