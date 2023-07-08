#!/bin/bash
aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
aws configure set default.region $AWS_REGION

export CLUSTER_NAME=$1
export NODE_TYPE=${2:-m1.large}
export NODE_COUNT=${3:-4}
export AZ1=$AWS_REGION"a"
export AZ2=$AWS_REGION"b"
envsubst < ./kubernetes/eks-cluster-config/eks.yaml.template > ./kubernetes/eks-cluster-config/eks.yaml

echo "Creating cluster $CLUSTER_NAME with $NODE_COUNT x $NODE_TYPE nodes"
echo "Attaching to $VPC_NAME: $VPC_ID"
eksctl create cluster -f ./kubernetes/eks-cluster-config/eks.yaml
eksctl upgrade cluster --name $CLUSTER_NAME --region $AWS_REGION
aws eks update-kubeconfig --region region-code --name my-cluster

echo ""
echo "Deploying Kubernetes Metrics Server ..."
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# # Kubeflow Training Operator
envsubst < ./kubernetes/kubeflow/clusterrole-hpa-access.yaml.template > ./kubernetes/kubeflow/clusterrole-hpa-access.yaml
envsubst < ./kubernetes/kubeflow/clusterrolebinding-training-operator-hpa-access.yaml.template > ./kubernetes/kubeflow/clusterrolebinding-training-operator-hpa-access.yaml
echo ""
echo "Deploying Kubeflow Training Operator ..."
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.5.0"
kubectl apply -f ./kubernetes/kubeflow/clusterrole-hpa-access.yaml
kubectl apply -f clusterrolebinding-training-operator-hpa-access.yaml

# Etcd
echo ""
echo "Deploying etcd ..."
envsubst < ./kubernetes/deployments/etcd-deployment.yaml.template > ./kubernetes/deployments/etcd-deployment.yaml
kubectl apply -f ./kubernetes/deployments/etcd-deployment.yaml

kubectl get pods


