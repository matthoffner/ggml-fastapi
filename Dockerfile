FROM python:latest

ENV PYTHONUNBUFFERED 1

EXPOSE 8000

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt


COPY --chown=user . $HOME/app

RUN ls -al

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
