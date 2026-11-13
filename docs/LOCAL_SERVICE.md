# 固定测试地址与自动启动

测试入口：<http://127.0.0.1:8768/>。

这是当前 Mac 的本地地址。登录电脑后自动加载模型；服务意外退出时由 macOS 重新启动，不依赖聊天窗口一直打开。电脑关机、睡眠或尚未登录时不能保证可访问，也不能在另一台电脑或手机上用这个地址访问这台 Mac。

在项目目录执行：

```sh
.venv/bin/python scripts/manage_local_service.py install
.venv/bin/python scripts/manage_local_service.py status
```

首次从手动服务切换时，应先停止原来的 `scripts/start_objects.sh` 进程，避免占用同一个端口。安装脚本不会擅自终止端口上的其他进程。注册成功与模型就绪分开报告；模型加载完成后 `status` 显示“新版模型：已就绪”。

启动配置保存在当前用户的 `~/Library/LaunchAgents/com.lingjing.vision-workbench.plist`；日志位于项目内的 `var/service/`。仅监听 `127.0.0.1`，没有开放公网端口。配置引用项目和虚拟环境的实际绝对路径，移动项目后需要重新安装配置。

取消自动启动：

```sh
.venv/bin/python scripts/manage_local_service.py uninstall
```

取消后仍可以用 `sh scripts/start_objects.sh` 手动启动。模型、照片和复核记录不会随取消操作删除。
