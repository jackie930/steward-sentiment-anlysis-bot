FROM tensorflow/tensorflow:1.15.2-gpu-py3

ENV LANG=en_US.utf8
COPY . /tmp
WORKDIR /tmp

RUN pip install boto3
CMD ["python","/tmp/task.py",\
"-i=s3://test-output-v1/test.txt",\
"-e=bert-sentiment-anylsis-2020-06-19-08-50-59-780",\
"-o=test-output-v1",\
"-id=${YOUR KEY ID}",\
"-secret=${YOUR SECRET KEY}",\
"-r=${REGION}"]