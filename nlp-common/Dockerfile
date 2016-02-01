FROM ubuntu:15.10

ENV CC  gcc
ENV CXX g++

WORKDIR /root

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
         libeigen3-dev \
         libgflags-dev \
         make \
         pkg-config

RUN ln -s /usr/bin/aclocal-1.15 /usr/bin/aclocal-1.14
RUN ln -s /usr/bin/automake-1.15 /usr/bin/automake-1.14

# Http Interface runs on GNU libmicrohttpd
RUN git clone https://github.com/Metaswitch/libmicrohttpd --depth=1 && \
    cd libmicrohttpd && \
    chmod +x ./configure && \
    ./configure --enable-doc=no && make && make install && \
    cd .. && \
    rm -rf libmicrohttpd

RUN apt-get install -y gdb valgrind

RUN locale-gen fr_FR.UTF-8

EXPOSE 8888

ADD . /root

RUN mkdir -p build && cd build && cmake .. && make

