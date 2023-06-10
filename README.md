## Installation Instructions

To install and run this repository on your machine, please follow these steps:

1. Clone the repository to your local machine using Git.
2. Navigate to the directory containing the cloned repository using your terminal.
3. Run the following command to start the installation process:

```
user$ ./process.sh 
```

This will install the necessary libraries with correct python versions required to run the following scripts successfully:

- `gather.py` - Used to gather and index data from specified URLs on the internet. The script exports an `extracted_text.txt` file containing the gathered data.
- `clean.py` - Used to tidy up and organize the data collected in `extracted_text.txt`.
- `train.py` - Used to tokenize and train the model. This script communicates with `dataset.py` (torch tokenizer) and exports a `trained_model` for testing and production.

4. Once the libraries are installed, define and change the hyperparameters for training in `train.py`.

That's it! You should now be able to use the scripts in this repository to gather, clean, and train data for LifeDeFied projects.
