#!/usr/bin/env bash
set -e
# ensure conda is set up
source /opt/conda/etc/profile.d/conda.sh
conda activate mbiomekit
exec "$@"
