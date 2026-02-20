FROM ubuntu:latest
LABEL authors="namit"

ENTRYPOINT ["top", "-b"]