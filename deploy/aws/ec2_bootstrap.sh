#!/usr/bin/env bash
# EC2 user-data / bootstrap for the inference host.
# Target: Ubuntu 24.04 LTS, ap-south-1 (Mumbai), t3.large or larger.
#
# Attach an instance profile carrying deploy/aws/iam_policy.json — that is what
# lets the box pull from ECR and read/write S3 without any stored credentials.
set -euo pipefail

APP_DIR=/opt/nvda-forecasting
EBS_DEVICE=/dev/nvme1n1        # verify with `lsblk` before first run
MLFLOW_MOUNT=/mnt/mlflow

echo "==> System packages"
apt-get update && apt-get upgrade -y
apt-get install -y ca-certificates curl gnupg unzip jq

echo "==> Docker Engine + Compose plugin"
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
systemctl enable --now docker
usermod -aG docker ubuntu

echo "==> AWS CLI v2"
curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp && /tmp/aws/install --update && rm -rf /tmp/aws /tmp/awscliv2.zip

echo "==> SSM Agent (deployments arrive over SSM, so port 22 stays closed)"
snap install amazon-ssm-agent --classic || true
snap start amazon-ssm-agent || systemctl enable --now amazon-ssm-agent

echo "==> EBS volume for MLflow metadata"
if [ -b "$EBS_DEVICE" ]; then
  blkid "$EBS_DEVICE" >/dev/null 2>&1 || mkfs -t ext4 "$EBS_DEVICE"
  mkdir -p "$MLFLOW_MOUNT"
  grep -q "$MLFLOW_MOUNT" /etc/fstab \
    || echo "$EBS_DEVICE $MLFLOW_MOUNT ext4 defaults,nofail 0 2" >> /etc/fstab
  mount -a
  chown -R 10002:10002 "$MLFLOW_MOUNT"
else
  echo "WARNING: $EBS_DEVICE not found — MLflow will use a Docker volume instead"
fi

echo "==> CloudWatch agent"
curl -fsSL https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb \
  -o /tmp/cwagent.deb && dpkg -i -E /tmp/cwagent.deb && rm -f /tmp/cwagent.deb
cp "$APP_DIR/deploy/aws/cloudwatch_agent.json" /opt/aws/amazon-cloudwatch-agent/bin/config.json 2>/dev/null || true
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config -m ec2 -s -c file:/opt/aws/amazon-cloudwatch-agent/bin/config.json || true

echo "==> Application"
mkdir -p "$APP_DIR"
cd "$APP_DIR"
# In practice this is a git clone or an S3 sync of the release bundle.
# git clone https://github.com/<org>/nvda-forecasting.git .

echo "==> Log rotation (unbounded container logs fill the root volume)"
cat > /etc/docker/daemon.json <<'JSON'
{
  "log-driver": "json-file",
  "log-opts": {"max-size": "10m", "max-file": "3"}
}
JSON
systemctl restart docker

echo "==> Done. Start the stack with:"
echo "    cd $APP_DIR && docker compose -f deploy/docker-compose.yml up -d"
