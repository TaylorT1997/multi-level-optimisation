# Environment Setup

## Docker Container

```
docker run -d -v "/${PWD}:/workspace" -p 8080:8080 --name "ml-workspace" --env AUTHENTICATE_VIA_JUPYTER="mytoken" --env WORKSPACE_SSL_ENABLED="true" --shm-size 8g --restart always --gpus all dagshub/ml-workspace-gpu:latest
```

## Anaconda

```
conda create -y --name mlo python=3.6 --file requirements.txt
```

```
conda list -e > requirements.txt
```