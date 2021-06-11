# Base image
FROM pytorch/pytorch

# Install necessary package dependencies
COPY ./requirements.txt /workspace
RUN apt-get update && apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*
RUN conda install -c conda-forge hdbscan && \
    python3 -m pip install -r /workspace/requirements.txt

# Set jupyter notebook theme
RUN jt -t onedork -fs 95 -tfs 11 -nfs 115 -cellw 88% -T && \
    jupyter nbextension enable --py widgetsnbextension

# Add Tini (What is advantage of Tini? https://github.com/krallin/tini/issues/8)
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--"]

# Defaults for executing the container (will be executed under Tini)
# CMD ["/bin/bash", "--login", "-i"]
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

# The container listens to port 8888
EXPOSE 8888