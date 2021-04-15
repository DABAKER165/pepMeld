FROM python:3.8.0-slim as builder
RUN apt-get update \
&& apt-get install gcc -y \
&& apt-get clean
COPY requirements.txt /app/requirements.txt
WORKDIR app
RUN pip install --user -r requirements.txt
COPY . /app

# Here is the production image
FROM python:3.8.0-slim as app
COPY --from=builder /root/.local /root/.local
# COPY --from=builder /app/main.py /app/main.py
ENV PATH=/root/.local/bin:$PATH
ADD https://github.com/DABAKER165/pepMeld/archive/master.tar.gz /pepMeld-master.tar.gz
# COPY pepMeld.tar.gz /
RUN cd / && tar -xzf pepMeld-master.tar.gz && rm pepMeld-master.tar.gz && mv pepMeld-master pepMeld
WORKDIR /pepMeld
CMD ["/bin/bash"]

