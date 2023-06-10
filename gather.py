import requests
from bs4 import BeautifulSoup

import requests

response = requests.get('https://www.google.com')
print(response.status_code)

# Enter the URLs you want to extract data from in a list
urls = ["https://github.com/sigp/solidity-security-blog"]

# Create an empty list to hold all the extracted text
extracted_text = []

for url in urls:
    # Send a request to the URL and get the response
    response = requests.get(url)
    
    # Use BeautifulSoup library to parse html content of the page
    soup = BeautifulSoup(response.content, features="html.parser")
    
    # Extract all the text from the page and append it to the extracted_text list
    extracted_text.append(soup.get_text())
    
# Join all the extracted text into one string with line breaks between each URL's text
all_extracted_text = "\n\n".join(extracted_text)

# Write the extracted text to a .txt file
with open("extracted_text.txt", "w") as f:
    f.write(all_extracted_text)
