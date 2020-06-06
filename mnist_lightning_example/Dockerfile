from python:3.7-slim

RUN apt-get update && apt-get install --no-install-recommends -y default-libmysqlclient-dev gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install pipenv

WORKDIR /app

# Copy project dependencies
COPY Pipfile* /app/

# Install project dependencies
RUN pipenv install --system

# Download language shit
RUN python -m spacy download en_core_web_sm

# Copy project files
COPY . /app/

ENTRYPOINT ["sh", "-c"]

CMD ["echo sup && python /app/server.py"]