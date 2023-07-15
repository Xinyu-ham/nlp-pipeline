docker build -f Dockerfile.train -t train .
docker run --env-file=.env --gpu=all -p 8888:8888 --memory=12G train