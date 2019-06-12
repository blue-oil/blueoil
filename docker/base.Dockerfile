FROM nvidia/cuda:10.0-cudnn7-runtime

RUN apt-get update && apt-get -y install --no-install-recommends \
	python3 python3-setuptools python3-wheel python3-pip python3-pil libgomp1 && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10

RUN pip3 install \
	tensorflow==1.13.1 \
	tensorflow-gpu==1.13.1 \
	click==6.7 \
	easydict==1.6 \
	matplotlib==2.2.2 \
	pandas==0.19.2 \
	pytablewriter==0.35 \
	PyYAML==4.2b4 \
	jinja2==2.10.1 && \
	rm -rf /root/.cache
