.PHONY: up down shell logs build restart pin ressh check-env tb

# 在后台启动容器
up:
	docker compose up -d

# 停止并删除容器
down:
	docker compose down

# 进入运行中的容器内部
shell:
	docker exec -it nanoamd bash

tb:
	docker exec -d nanoamd tensorboard --logdir /home/ubuntu/workspace/mainrun/mainrun/experiments --host=0.0.0.0 --port=6006

# 实时查看容器日志
logs:
	docker compose logs -f

# nanoamd 启动失败时看退出原因
logs-nanoamd:
	docker compose logs nanoamd

# 重新构建镜像 (当你修改了 Dockerfile 后需要跑这个)
build:
	docker compose build

# 重启容器 (先关闭再启动)
restart: build down up

ressh:
	ssh-keygen -R "[192.168.2.209]:2222"

pin:
    pip install datasets --proxy="" -i https://pypi.tuna.tsinghua.edu.cn/simple


check-env:
	rocminfo
	python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
