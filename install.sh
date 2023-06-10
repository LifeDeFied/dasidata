#!/bin/bash

# Check if Python is already installed
if which python3 >/dev/null 2>&1; then
    echo "Python is already installed on this system"
else
    # Check OS type
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            # Linux OS
            echo "Detected Linux OS"
            wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz
            tar -xvf Python-3.10.0.tgz
            cd Python-3.10.0/
            ./configure --enable-optimizations
            make -j 8
            sudo make altinstall
            cd ../
            rm -rf Python-3.10.0.tgz Python-3.10.0/
    elif [[ "$OSTYPE" == "darwin"* ]]; then
            # Mac OS
            echo "Detected Mac OS"
            brew install python@3.10
    else
            # Unsupported OS
            echo "Unsupported OS type"
    fi
fi

# Required Python libraries
LIBRARIES=(
    beautifulsoup4==4.12.2
    brotlipy==0.7.0
    certifi==2021.10.8
    cffi==1.14.6
    click==8.1.3
    cmake==3.26.3
    conda==23.3.1
    contourpy==1.0.7
    cryptography==3.4.8
    cycler==0.11.0
    filelock==3.12.0
    fonttools==4.39.4
    fsspec==2023.5.0
    huggingface-hub==0.14.1
    idna==3.3
    Jinja2==3.1.2
    joblib==1.2.0
    jsonpointer==2.1
    kiwisolver==1.4.4
    lit==16.0.5
    MarkupSafe==2.1.2
    matplotlib==3.7.1
    mpmath==1.3.0
    networkx==3.1
    nltk==3.8.1
    numpy==1.24.3
    nvidia-cublas-cu11==11.10.3.66
    nvidia-cuda-cupti-cu11==11.7.101
    nvidia-cuda-nvrtc-cu11==11.7.99
    nvidia-cuda-runtime-cu11==11.7.99
    nvidia-cudnn-cu11==8.5.0.96
    nvidia-cufft-cu11==10.9.0.58
    nvidia-curand-cu11==10.2.10.91
    nvidia-cusolver-cu11==11.4.0.1
    nvidia-cusparse-cu11==11.7.4.91
    nvidia-nccl-cu11==2.14.3
    nvidia-nvtx-cu11==11.7.91
    pandas==2.0.1
    Pillow==9.5.0
    pluggy==1.0.0
    pycparser==2.20
    pyparsing==3.0.9
    PySocks==1.7.1
    python-dateutil==2.8.2
    pytz==2023.3
    PyYAML==6.0
    regex==2023.5.5
    requests==2.26.0
    ruamel.yaml==0.17.16
    ruamel.yaml.clib==0.2.6
    six==1.16.0
    soupsieve==2.4.1
    sympy==1.12
    tokenizers==0.13.3
    toolz==0.11.2
    torch==1.10.0
    tqdm==4.62.2
    transformers==4.29.2
    triton==2.0.0
    typing_extensions==4.5.0
    tzdata==2023.3
    urllib3==1.26.7
    zstandard==0.15.2
)

# Install required Python libraries
for LIBRARY in "${LIBRARIES[@]}"
do
    pip3 show $LIBRARY >/dev/null 2>&1 || pip3 install $LIBRARY
done

# Generate requirements file
pip3 freeze > requirements.txt

# Print message indicating that installation was successful
echo "Python libraries have been installed. A requirements.txt file has been generated."
