#!/bin/bash
set -e

# Kiểm tra và tạo thư mục
directories=("data" "models" "results" "logs")
for dir in "${directories[@]}"; do
    mkdir -p "/app/$dir"
done

# Chạy migrations hoặc setup nếu cần
if [ "$1" = 'init' ]; then
    python setup.py
    exit 0
fi

# Chạy command được truyền vào
exec "$@"