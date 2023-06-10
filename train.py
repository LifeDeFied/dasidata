import os
import torch
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer
)
from torch import tensor
from torch.utils.data import DataLoader
from dataset import TextIterableDataset
from torch.nn.utils.rnn import pad_sequence

sentences = [
    "Cybersecurity threats are constantly evolving, and it's crucial to stay up-to-date with the latest best practices.",
    "Smart contract security is an important consideration when deploying any decentralized application on a blockchain platform.",
    "Blockchain technology provides a highly secure and transparent way to store and share data across a network of peers.",
    "The rise of decentralized finance (DeFi) has brought new security challenges, particularly around the management of private keys.",
    "One key advantage of blockchain-based systems is that they can be designed to be inherently resistant to many types of cyberattacks.",
    "To ensure the security of your smart contracts, it's important to conduct thorough testing and auditing before deployment.",
    "The use of multi-signature wallets and other advanced security features can help protect against malicious actors in the DeFi space.",
    "Blockchain networks rely on complex cryptographic algorithms to maintain their security and integrity.",
    "As more financial transactions move onto blockchain platforms, cybersecurity will become even more critical for businesses and individuals alike.",
    "Keeping your software tools and hardware devices up-to-date with the latest security patches is an essential part of maintaining overall cybersecurity and protecting against potential vulnerabilities."
]

# Load the pre-trained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize your input data
input_ids = []
for sentence in sentences:
    encoded_sent = tokenizer.encode(sentence, add_special_tokens=True)
    input_ids.append(torch.tensor(encoded_sent))

# Set the maximum sequence length
max_seq_length = 512  # maximum allowed by BERT
input_ids = pad_sequence(
    input_ids, 
    batch_first=True, 
    padding_value=tokenizer.pad_token_id, 
)

sequence1 = [1, 2, 3, 4]
sequence2 = [5, 6, 7]

# Convert list of lists to list of tensors
sequences = [torch.tensor(seq) for seq in [sequence1, sequence2]]

# Pad the sequences to the same length
padded_seqs = pad_sequence(sequences, batch_first=True)

print(padded_seqs)

tensor([[1, 2, 3, 4],
        [5, 6, 7, 0]])

# Apply the model on your input
outputs = model(input_ids)

# Load the pre-trained GPT-2 model and tokenizer
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')
model_gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')

# Set the device to CPU if available, else use CPU
device = torch.device("cpu")
model_gpt2.to(device)

# Define the hyperparameters for training
batch_size = 2
epochs = 3
learning_rate = 1e-5

# Load the text data using the custom iterable dataset class and DataLoader
train_dataset = TextIterableDataset(tokenizer_gpt2, file_path='extracted_text.txt', block_size=128)
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=lambda x: pad_sequence(x, batch_first=True))

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model_gpt2.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Train the model for the specified number of epochs
for epoch in range(epochs):
    model_gpt2.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model_gpt2(batch, labels=batch)
        loss = criterion(outputs.logits.view(-1, tokenizer_gpt2.vocab_size), batch.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
print('Epoch {} | Average Loss: {:.4f}'.format(epoch+1, total_loss))

# Save the trained model
output_dir = 'trained_model'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
model_gpt2.save_pretrained(output_dir)
tokenizer_gpt2.save_pretrained(output_dir)
