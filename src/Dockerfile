# Use an official Python runtime as the base image
FROM python:3.9.1-slim

WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libssl-dev libffi-dev libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt /usr/src/app/
COPY mnist_exp_2.h5 /app/model/
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# Copy project files
COPY . /usr/src/app/
