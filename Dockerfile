FROM python:3.12-alpine AS subsetter

RUN pip install -U pip tqdm

WORKDIR /subsetter

COPY . ./
RUN python3 -m pip install -e .[all]

RUN adduser -S ctruser

USER ctruser
WORKDIR /config

ENTRYPOINT ["python3", "-m", "subsetter"]
