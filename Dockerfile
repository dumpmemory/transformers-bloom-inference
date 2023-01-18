FROM nvidia/cuda:11.6.1-runtime-ubi8 as base

# taken form pytorch's dockerfile
RUN curl -L -o ./miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ./miniconda.sh && \
    ./miniconda.sh -b -p /opt/conda && \
    rm ./miniconda.sh

ENV PYTHON_VERSION=3.9 \
    PATH=/opt/conda/envs/inference/bin:/opt/conda/bin:${PATH}

# create conda env
RUN conda create -n inference python=${PYTHON_VERSION} pip -y

# change shell to activate env
SHELL ["conda", "run", "-n", "inference", "/bin/bash", "-c"]

FROM base as conda

# change shell to activate env

# update conda
RUN conda update -n base -c defaults conda -y
# cmake
RUN conda install -c anaconda cmake -y

# update conda
RUN conda update -n base -c defaults conda -y

# necessary stuff
RUN pip install torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116 \
    transformers \
    deepspeed==0.7.7 \
    accelerate \
    gunicorn \
    flask \
    flask_api \ 
    pydantic \
    huggingface_hub \
	grpcio-tools==1.50.0 \
    --no-cache-dir

# clean conda env
RUN conda clean -ya

# change this as you like 🤗
ENV TRANSFORMERS_CACHE=/cos/HF_cache \
    HUGGINGFACE_HUB_CACHE=${TRANSFORMERS_CACHE}

FROM conda as app

WORKDIR /src
RUN chmod -R g+w /src

ENV PORT=5000 \
    UI_PORT=5001
EXPOSE ${PORT}
EXPOSE ${UI_PORT}

CMD git clone https://github.com/huggingface/transformers-bloom-inference.git && \
    cd transformers-bloom-inference && \
    # install grpc and compile protos
    make gen-proto && \
    make bloom-560m
