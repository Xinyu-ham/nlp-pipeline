#!/bin/bash
echo "Deleting EFS.."
FILE_SYSTEM_ID=$(aws efs describe-file-systems --query 'FileSystems[*].FileSystemId' --output json | jq -r .[0] )
if [ -z "$FILE_SYSTEM_ID"]; then
        echo "No EFS Filesystems found."
else
        echo "Deleting EFS mount targets for File System $FILE_SYSTEM_ID ..."
        MOUNT_TARGETS="$(aws efs describe-mount-targets --file-system-id $FILE_SYSTEM_ID --query MountTargets[].MountTargetId --output text)"
        MT=$(echo $MOUNT_TARGETS)
        for t in $MT; do echo Deleting mount target $t; aws efs delete-mount-target --mount-target-id $t; done 
        sleep 15
        echo "Deleting EFS file system $FILE_SYSTEM_ID ..."
        aws efs delete-file-system --file-system-id $FILE_SYSTEM_ID
fi

echo ""
echo 'Deleted EFS ID: $FILE_SYSTEM_ID.'

echo ""
echo "Deleting Clutser $CLUSTER_NAME .."
eksctl delete cluster -f kubernetes/eks-cluster-config/eks.yaml --disable-nodegroup-eviction
echo "It will take a few minutes for CloudFormation to delete the cluster."
echo "You can check the status in the CloudFormation console."

echo ""
VPC_ID=$(aws eks describe-cluster --name $CLUSTER_NAME --query "cluster.resourcesVpcConfig.vpcId" --output text)
MOUNT_TARGET_GROUP_NAME="eks-efs-group-${CLUSTER_NAME}"
echo "Deleting security group $MOUNT_TARGET_GROUP_NAME ..."
MOUNT_TARGET_GROUP_ID=$(aws ec2 describe-security-groups --filter Name=vpc-id,Values=$VPC_ID Name=group-name,Values=$MOUNT_TARGET_GROUP_NAME --query 'SecurityGroups[*].[GroupId]' --output text)
aws ec2 delete-security-group --group-id $MOUNT_TARGET_GROUP_ID

echo ""
echo "Deleting NAT Gateways ..."
NAT_GATEWAYS=$(aws ec2 describe-nat-gateways --filter Name=vpc-id,Values=$VPC_ID --query 'NatGateways[*].NatGatewayId' --output text)
aws ec2 delete-nat-gateway --nat-gateway-id $NAT_GATEWAYS

echo "Exited."