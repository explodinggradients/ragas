FROM python:3.9-slim
RUN apt-get update && apt-get install -y git make
COPY . /app
WORKDIR /app
RUN pip install -e /app/
ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=$OPENAI_API_KEY
RUN make run-benchmarks
