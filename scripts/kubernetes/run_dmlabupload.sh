#!/bin/bash -e

MLFLOW_EXPERIMENT_NAME=$1
ENVID=$2
SHARD_FROM=$3
SHARD_TO=$4

TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
TAG=$(git describe --tags | sed 's/-g[a-z0-9]\{7\}//')
RND=$(base64 < /dev/urandom | tr -d '[A-Z/+]' | head -c 6)

docker build . -f Dockerfile.tf -t eu.gcr.io/human-ui/dmlabupload:$TAG
docker push eu.gcr.io/human-ui/dmlabupload:$TAG

cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: dmlabupload-$TIMESTAMP
  namespace: default
spec:
  backoffLimit: 2
  template:
    spec:
      volumes:
        - name: google-cloud-key
          secret:
            secretName: mlflow-worker-key
      containers:
        - name: dreamer
          imagePullPolicy: Always
          image: eu.gcr.io/human-ui/dmlabupload:$TAG
          env:
            - name: GOOGLE_APPLICATION_CREDENTIALS
              value: /var/secrets/google/key.json
            - name: MLFLOW_TRACKING_URI
              value: http://mlflow-service.default.svc.cluster.local
            - name: MLFLOW_EXPERIMENT_NAME
              value: ${MLFLOW_EXPERIMENT_NAME}
          volumeMounts:
            - name: google-cloud-key
              mountPath: /var/secrets/google
          command:
            - python3 
          args:
            - pydreamer/offline_upload_rludmlab.py
            - --env
            - "${ENVID}"
            - --shard_from
            - "${SHARD_FROM}"
            - --shard_to
            - "${SHARD_TO}"
            - --resume_id
            - $RND
          resources:
            requests:
              memory: 1000Mi
              cpu: 1500m
      restartPolicy: Never
EOF