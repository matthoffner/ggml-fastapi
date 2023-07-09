FROM nvidia/cuda:11.0.3-base-ubuntu20.04

ENV MODEL_NAME=""
ENV DEFAULT_MODEL_FILE=""
ENV MODEL_USER="TheBloke"
ENV DEFAULT_MODEL_BRANCH="main"
ENV MODEL_URL="https://huggingface.co/${MODEL_USER}/${MODEL_NAME}/raw/${DEFAULT_MODEL_BRANCH}/${DEFAULT_MODEL_FILE}"
ENV PATH="/usr/local/cuda/bin:$PATH"

RUN apt update && \
    apt install --no-install-recommends -y build-essential python3 python3-pip wget curl git && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN apt-get install -y wget && \
    wget -qO- "https://cmake.org/files/v3.18/cmake-3.18.0-Linux-x86_64.tar.gz" | tar --strip-components=1 -xz -C /usr/local

COPY requirements.txt ./

RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

RUN CT_CUBLAS=1 pip install ctransformers --no-binary ctransformers

RUN wget -O /app/${DEFAULT_MODEL_FILE} ${MODEL_URL}

RUN useradd -m -u 1000 user

RUN mkdir -p /home/user/app && mv /app/${DEFAULT_MODEL_FILE} /home/user/app

RUN chown user:user /home/user/app/${DEFAULT_MODEL_FILE}

USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH \
    MODEL_NAME=${MODEL_NAME} \
    MODEL_FILE=/home/user/app/${DEFAULT_MODEL_FILE} \
    MODEL_TYPE=${DEFAULT_MODEL_TYPE}

WORKDIR $HOME/app

COPY --chown=user . .

RUN ls -al

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
