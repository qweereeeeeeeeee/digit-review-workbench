"""Manage a per-user macOS LaunchAgent for the local recognition interface."""
import argparse
import json
import os
from pathlib import Path
import plistlib
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
LABEL = 'com.lingjing.vision-workbench'
DOMAIN = f'gui/{os.getuid()}'
SERVICE = f'{DOMAIN}/{LABEL}'
PLIST = Path.home() / 'Library/LaunchAgents' / f'{LABEL}.plist'
URL = 'http://127.0.0.1:8768/'


def launchctl(*args, check=True):
    return subprocess.run(['/bin/launchctl', *args], check=check,
                          capture_output=True, text=True)


def install():
    python = ROOT / '.venv/bin/python'
    if not python.is_file():
        raise RuntimeError('缺少项目虚拟环境，请先按 README 安装依赖。')
    logs = ROOT / 'var/service'
    logs.mkdir(parents=True, exist_ok=True)
    settings = {
        'Label': LABEL,
        'ProgramArguments': [str(python), '-m', 'uvicorn', 'objects.api:app',
                             '--host', '127.0.0.1', '--port', '8768', '--no-access-log'],
        'WorkingDirectory': str(ROOT),
        'RunAtLoad': True,
        'KeepAlive': True,
        'ThrottleInterval': 30,
        'EnvironmentVariables': {'PYTHONUNBUFFERED': '1',
                                 'PATH': '/usr/bin:/bin:/usr/sbin:/sbin'},
        'StandardOutPath': str(logs / 'stdout.log'),
        'StandardErrorPath': str(logs / 'stderr.log'),
    }
    PLIST.parent.mkdir(parents=True, exist_ok=True)
    temporary = PLIST.with_suffix('.plist.tmp')
    with temporary.open('wb') as stream:
        plistlib.dump(settings, stream)
    temporary.chmod(0o600)
    if launchctl('print', SERVICE, check=False).returncode == 0:
        launchctl('bootout', SERVICE)
    temporary.replace(PLIST)
    launchctl('bootstrap', DOMAIN, str(PLIST))
    print('已注册登录启动和退出后重启；模型仍需加载，请用 status 检查。')
    print(URL)


def status():
    result = launchctl('print', SERVICE, check=False)
    print('自动启动：' + ('已注册' if result.returncode == 0 else '未注册'))
    for line in result.stdout.splitlines():
        if line.strip().startswith(('state =', 'pid =', 'last exit code =')):
            print(line.strip())
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(URL + 'api/health', timeout=5) as response:
            health = json.load(response)
        ready = health.get('status') == 'ready' and health.get('general_vision') is True
        print('新版模型：' + ('已就绪' if ready else '未就绪'))
        print(URL)
        return 0 if ready and result.returncode == 0 else 1
    except OSError as exc:
        print('服务尚不可访问：' + str(exc))
        return 1


def uninstall():
    if launchctl('print', SERVICE, check=False).returncode == 0:
        launchctl('bootout', SERVICE)
    PLIST.unlink(missing_ok=True)
    print('已移除自动启动；项目、模型和识别记录均保留。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['install', 'status', 'uninstall'])
    action = parser.parse_args().action
    if sys.platform != 'darwin':
        parser.error('此脚本仅用于 macOS；其他系统请使用 scripts/start_objects.sh。')
    try:
        return {'install': install, 'status': status, 'uninstall': uninstall}[action]()
    except subprocess.CalledProcessError as exc:
        print('launchctl 操作失败：' + exc.stderr.strip(), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
