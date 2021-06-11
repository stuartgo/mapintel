FROM python:3.7.4-stretch

WORKDIR /home/user

RUN apt-get update && apt-get install -y curl git pkg-config cmake

# Install PDF converter
RUN wget --no-check-certificate https://dl.xpdfreader.com/xpdf-tools-linux-4.03.tar.gz && \
    tar -xvf xpdf-tools-linux-4.03.tar.gz && cp xpdf-tools-linux-4.03/bin64/pdftotext /usr/local/bin

RUN apt-get install libpoppler-cpp-dev pkg-config -y --fix-missing

# install packages
COPY requirements.txt /home/user/
RUN pip install -r requirements.txt

# Copy API code
COPY . /home/user/api

EXPOSE 8000

# cmd for initializing the container
CMD ["api/init_container.sh"]