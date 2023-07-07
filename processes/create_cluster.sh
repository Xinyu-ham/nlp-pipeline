aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
aws configure set default.region $AWS_REGION

eksctl create cluster -f ./kubernetes/eks-cluster-config/eks.yaml
aws eks update-kubeconfig --name $CLUSTER_NAME

eksctl upgrade cluster --name $CLUSTER_NAME

echo ""
echo "Deploying Kubernetes Metrics Server ..."
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# Kubeflow Training Operator
echo ""
echo "Deploying Kubeflow Training Operator ..."
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.5.0"
kubectl apply -f ./kubernetes/jobs/clusterrole-hpa-access.yaml
kubectl apply -f ./kubernetes/jobs/clusterrolebinding-training-operator-hpa-access.yaml

# Etcd
echo ""
echo "Deploying etcd ..."
kubectl apply -f ./kubernetes/deployments/etcd-deployment.yaml

kubectl get pods