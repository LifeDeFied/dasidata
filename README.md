Within your terminal run the following command. 

# user$ ./process.sh 

This will run and install.sh the needed libraries with correct python versions in order to run the following:

# gather.py (gather and index data from specified URLs on the internet)
- exports an extracted_text.txt

# clean.py (tidy up and organize the data gathered in extracted_text.txt)

# train.py (tokenize and train the model)
- communicates with dataset.py (torch tokenizer)

A trained_model is then exported for testing and production

# Define the hyperparameters for training in train.py
