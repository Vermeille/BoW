FROM ubuntu:14.04

ENV CC gcc
ENV CXX  g++

# What is the best practice regarding workdir?
WORKDIR /root

# TODO: this was intended as a very general dev platform, but the http interface doesn't need
# all of this. Purge the unnecessary
RUN apt-get update &&  apt-get install -y \
         automake \
         autoconf \
         autoconf-archive \
         cmake \
         gcc \
         g++ \
         git \
         libboost-all-dev \
         libgoogle-glog-dev \
         libgflags-dev \
         make \
         pkg-config

# Http Interface runs on GNU libmicrohttpd
RUN git clone https://github.com/Metaswitch/libmicrohttpd --depth=1 && \
    cd libmicrohttpd && \
    chmod +x ./configure && \
    ./configure --enable-doc=no && make && make install && \
    cd .. && \
    rm -rf libmicrohttpd

RUN apt-get install -y gdb valgrind

RUN git clone https://github.com/Vermeille/http-interface && \
    cd http-interface && \
    git checkout ff9093e34 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make install && \
    cd ../.. && \
    rm -rf http-interface


RUN locale-gen fr_FR.UTF-8
RUN git clone https://github.com/Vermeille/nlp-common && \
    cd nlp-common && \
    git checkout 56e8c14c91 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make install && \
    cd ../.. && \
    rm -rf nlp-common && echo i

ADD . /root

EXPOSE 8888

RUN mkdir build && cd build && cmake .. && make

ENTRYPOINT ["valgrind", "./build/bow"]
