"""Prepare and seal a source/video/history delivery, excluding private workspace data."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import plistlib
import re
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STYLE = 'body{font:16px/1.85 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;color:#253347;max-width:960px;margin:48px auto;padding:0 24px}h1{font-size:30px}h2{font-size:22px;margin-top:36px}a{color:#245d8c}code,pre{background:#f1f4f7;padding:3px 6px;border-radius:3px}pre{white-space:pre-wrap;padding:18px}p{margin:16px 0}nav{background:#f3f6f9;padding:20px;border-radius:8px}li{margin:10px 0}'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def render_markdown(source):
    blocks = []
    for block in source.strip().split('\n\n'):
        tag = 'h1' if block.startswith('# ') else 'h2' if block.startswith('## ') else 'p'
        if tag != 'p':
            block = block.split(' ', 1)[1]
        value = html.escape(block)
        value = re.sub(r'`([^`]+)`', r'<code>\1</code>', value)
        value = re.sub(r'&lt;(https?://[^&]+)&gt;', r'<a href="\1">\1</a>', value)
        blocks.append(f'<{tag}>{value}</{tag}>')
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>项目说明</title><style>'+STYLE+'</style><body>'+''.join(blocks)+'</body></html>'


def prepare(destination, exam):
    destination.mkdir(parents=True, exist_ok=False)
    source = destination/'03_项目源码'
    source.mkdir()
    # Only tracked project files are eligible. User photos, README copies,
    # live SQLite databases, tokens, caches and virtual environments stay out.
    paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    for relative in filter(None, paths):
        path = ROOT/relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError('不打包符号链接或缺失文件：'+relative)
        target = source/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    weights = []
    for folder, manifest in [('florence', 'florence-source.json'), ('ppocr', 'ppocr-source.json')]:
        data = json.loads((ROOT/'artifacts/vision'/manifest).read_text())
        for entry in data['files']:
            relative = Path('artifacts/vision')/folder/entry['name']
            original = ROOT/relative
            assert sha(original) == entry['sha256'], relative
            target = source/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
            weights.append(str(Path('03_项目源码')/relative))
    for name in ['resnet50-11ad3fa6.pth', 'yolo11s.pt']:
        relative = Path('artifacts/objects')/name
        shutil.copy2(ROOT/relative, source/relative)
        weights.append(str(Path('03_项目源码')/relative))
    videos = destination/'02_运行视频'
    videos.mkdir()
    shutil.copy2(ROOT/'artifacts/vision/demo/新版模型实际运行.mp4', videos/'新版通用识别_实际运行.mp4')
    shutil.copy2(ROOT/'artifacts/review/demo-v2/运行演示.mp4', videos/'手写ResNet数字版_实际运行.mp4')
    shutil.copy2(ROOT/'docs/VIDEO.md', videos/'录制说明.md')
    report = (ROOT/'docs/DELIVERY_REPORT.md').read_text()
    (destination/'01_项目说明.md').write_text(report)
    (destination/'01_项目说明.html').write_text(render_markdown(report))
    checks = destination/'04_考核与验证'
    checks.mkdir()
    shutil.copy2(exam, checks/'第一阶段考核任务.pdf')
    shutil.copy2(ROOT/'docs/ACCEPTANCE.md', checks/'验收对应表.md')
    shutil.copy2(ROOT/'docs/GENERAL_VISION_CHECK.md', checks/'通用识别实际结果.md')
    shutil.copytree(ROOT/'artifacts/delivery', checks/'本次验证')
    history = destination/'05_开发历史'
    history.mkdir()
    subprocess.run(['git', 'bundle', 'create', str(history/'本地真实开发历史.bundle'), '--all'], cwd=ROOT, check=True)
    log = subprocess.check_output(['git', 'log', '--reverse', '--format=%H %aI %s'], cwd=ROOT, text=True)
    (history/'本地提交记录.txt').write_text(log)
    remote = ROOT/'var/packaging/remote-history.json'
    if not remote.is_file():
        raise RuntimeError('缺少实际获取的远程提交快照')
    shutil.copy2(remote, history/'GitHub提交快照.json')
    (history/'历史说明.txt').write_text('本地 bundle 保留原始提交对象和时间；远程快照来自 GitHub。由于早期通过连接器分阶段发布，本地与远程的提交 SHA 不一一相同，不冒充同一条历史。\n还原本地历史：git clone 本地真实开发历史.bundle restored-project\n仓库：https://github.com/qweereeeeeeeeee/digit-review-workbench\n')
    with (destination/'打开本机测试界面.webloc').open('wb') as stream:
        plistlib.dump({'URL':'http://127.0.0.1:8768/'}, stream)
    (destination/'仓库链接.txt').write_text('https://github.com/qweereeeeeeeeee/digit-review-workbench\n')
    instructions = '''# 灵境1：第一阶段交付包

先打开 00_交付导航.html，或双击 01_项目说明.html 阅读完整说明。

02_运行视频：基础手写 ResNet 演示和本次新版实际运行视频。
03_项目源码：代码、依赖、训练数据与结果、测试、模型文件及来源。
04_考核与验证：原任务 PDF、验收表、本次实际校验结果。
05_开发历史：本地 Git bundle、提交日志、GitHub 提交快照。

当前这台 Mac：双击“打开本机测试界面.webloc”，需要原服务已启动。
另一台机器：进入 03_项目源码，准备 Python 3.12，然后运行：
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
sh scripts/start_objects.sh
浏览器打开 http://127.0.0.1:8768/ 。数字考核基础版改为 sh scripts/start.sh，端口 8766。

完整包已包含模型文件，但仍需安装 Python 依赖，不是无需环境的独立 App。
精简包须额外运行：
.venv/bin/python scripts/download_object_model.py
.venv/bin/python scripts/download_vision_models.py
下载需要联网；完成后照片识别在本机执行。网络不能下载权重时使用完整包。

正式验证环境是 Intel macOS/Python 3.12；没有宣称在 Windows、Apple Silicon 或全新操作系统上完成验证。
不得把原 UCI 小图测试分数当成手机照片准确率；新版也有漏框、重复框和 OCR 误报。
历史文档中待补写的个人经历并非已完成证明；当前项目说明的感悟来自真实反馈整理，代码理解仍须本人准备。
不含用户私有照片、数据库、机器虚拟环境和临时缓存；测试示例均有公开来源。
'''
    (destination/'00_先读我.txt').write_text(instructions)
    homepage = '''<h1>灵境1 · 第一阶段交付包</h1><p>图片识别与复核。基础手写 ResNet 与新版物体、文字识别均保留。</p>
<nav><a href="01_项目说明.html">阅读项目说明与个人感悟</a> · <a href="https://github.com/qweereeeeeeeeee/digit-review-workbench">GitHub 仓库</a> · <a href="http://127.0.0.1:8768/">打开本机测试界面</a></nav>
<h2>先看运行结果</h2><p>基础版展示数字预测与复核；新版展示火箭、多区域识别及重复框处理。两个视频均来自实际运行，操作自动化、无合成配音。</p>
<ul><li><a href="02_运行视频/手写ResNet数字版_实际运行.mp4">手写 ResNet 数字版视频</a></li><li><a href="02_运行视频/新版通用识别_实际运行.mp4">新版通用识别视频</a></li></ul>
<h2>代码与检查材料</h2><ul><li><a href="03_项目源码/README.md">源码运行说明</a></li><li><a href="04_考核与验证/第一阶段考核任务.pdf">原考核要求</a></li><li><a href="04_考核与验证/验收对应表.md">逐项验收对应表</a></li><li><a href="05_开发历史/本地提交记录.txt">真实开发提交记录</a></li><li><a href="00_先读我.txt">安装步骤与完整文件说明</a></li></ul>
<h2>第一次在另一台电脑运行</h2><p>进入“03_项目源码”，准备 Python 3.12。完整包带模型，但仍需要安装依赖。精简包需要先按 README 下载模型。</p><pre>python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
sh scripts/start_objects.sh</pre><p>然后访问 http://127.0.0.1:8768/。这个地址是各自电脑的本地地址，不能当作公网演示网址。</p>
<h2>结果的边界</h2><p>原 UCI 测试集 267/270 的结果不能代表手机照片。新版仍有误报、遗漏和重复框。所有已报告的失败均保留；本人代码理解需要在答辩中实际展示。</p>'''
    (destination/'00_交付导航.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>灵境1 交付导航</title><style>'+STYLE+'</style><body>'+homepage+'</body></html>')
    (destination/'打包来源.json').write_text(json.dumps({'local_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(), 'large_model_paths': weights, 'private_data_included': False},ensure_ascii=False,indent=2)+'\n')
    print(destination)


def seal(destination):
    info = json.loads((destination/'打包来源.json').read_text())
    excluded = set(info['large_model_paths'])
    paths = sorted(p for p in destination.rglob('*') if p.is_file())
    if any(p.suffix in {'.sqlite3', '.pyc'} or '.venv' in p.parts or p.is_symlink() for p in paths):
        raise RuntimeError('交付目录含不应打包的运行文件')
    entries = [{'path':str(p.relative_to(destination)), 'bytes':p.stat().st_size, 'sha256':sha(p)}
               for p in paths if p.name != '文件校验清单.json']
    (destination/'文件校验清单.json').write_text(json.dumps(entries,ensure_ascii=False,indent=2)+'\n')
    paths = sorted(p for p in destination.rglob('*') if p.is_file())
    result = []
    for kind in ['完整交付包', '精简提交包']:
        archive = destination.parent/f'{destination.name}-{kind}.zip'
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
            for p in paths:
                relative = str(p.relative_to(destination))
                if kind == '精简提交包' and relative in excluded:
                    continue
                if kind == '精简提交包' and relative == '文件校验清单.json':
                    z.writestr('灵境1/文件校验清单.json',json.dumps([e for e in entries if e['path'] not in excluded],ensure_ascii=False,indent=2)+'\n')
                else:
                    z.write(p, str(Path('灵境1')/relative))
        with zipfile.ZipFile(archive) as z:
            bad = z.testzip()
            if bad: raise RuntimeError('ZIP 校验失败：'+bad)
        result.append({'file':str(archive), 'bytes':archive.stat().st_size, 'sha256':sha(archive)})
        print(result[-1], flush=True)
    (destination.parent/(destination.name+'-压缩包校验.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'seal'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exam', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        if not args.exam or not args.exam.is_file(): parser.error('prepare 需要有效的 --exam PDF 路径')
        prepare(args.output, args.exam)
    else:
        seal(args.output)
