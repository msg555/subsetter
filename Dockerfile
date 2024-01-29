FROM python:3.12-alpine

RUN pip install -U pip

WORKDIR /subsetter

COPY . ./
RUN python3 -m pip install -e .

RUN adduser -S ctruser

USER ctruser
WORKDIR /config

ENTRYPOINT ["python3", "-m", "subsetter"]
