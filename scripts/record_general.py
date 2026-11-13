"""Record the current UI against real models and an isolated demonstration database."""
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import imageio_ffmpeg
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/vision/demo'


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runtime = ROOT / 'var/browser-runtime'
    encoder = runtime / 'ffmpeg-1011/ffmpeg-mac'
    encoder.parent.mkdir(parents=True, exist_ok=True)
    if not encoder.exists():
        encoder.symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
    os.environ['PLAYWRIGHT_BROWSERS_PATH'] = str(runtime)
    rows = []
    errors = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path=os.environ.get('CHROME_PATH', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'),
            headless=True)
        context = await browser.new_context(viewport={'width': 1440, 'height': 960},
                    locale='zh-CN', record_video_dir=str(ROOT/'var/packaging/capture'),
                    record_video_size={'width': 1440, 'height': 960})
        page = await context.new_page()
        page.set_default_timeout(180000)
        page.on('pageerror', lambda e: errors.append(str(e)))
        await page.goto(os.environ.get('DEMO_URL', 'http://127.0.0.1:8770'), wait_until='networkidle')
        await page.wait_for_function("document.querySelector('#health').textContent.includes('OCR')")
        await asyncio.sleep(2)
        for name in ['rocket', 'astronaut', 'coffee']:
            path = ROOT / f'artifacts/objects/examples/{name}.png'
            await page.locator('#file').set_input_files(str(path))
            await page.wait_for_function("!document.querySelector('#recognize').disabled")
            await asyncio.sleep(1)
            async with page.expect_response(lambda r: '/api/recognize' in r.url and r.request.method == 'POST') as response:
                await page.locator('#recognize').click()
            response = await response.value
            assert response.ok, await response.text()
            record = await response.json()
            await page.wait_for_function("!document.querySelector('#recognize').disabled")
            assert record['result']['vision']
            rows.append({'sample': name, 'image_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'result': record['result']})
            await page.screenshot(path=str(OUT / f'{name}.png'))
            print(name, record['result']['vision']['seconds'], flush=True)
            await asyncio.sleep(5)
        await page.locator('#final-label').fill('咖啡杯、勺子')
        await page.locator('#note').fill('同一把勺子出现重复框，核对后只记一次。')
        await asyncio.sleep(2)
        await page.locator('#confirm').click()
        await page.wait_for_function("document.querySelector('#saved').textContent.includes('已确认')")
        await asyncio.sleep(4)
        await page.reload(wait_until='networkidle')
        await page.locator('.record').first.click()
        await page.wait_for_function("document.querySelector('#final-label').value==='咖啡杯、勺子'")
        await asyncio.sleep(4)
        assert not errors, errors
        await context.close()
        video = await page.video.path()
        await browser.close()
    target = OUT/'新版模型实际运行.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-i', video,
                    '-vf', 'crop=1440:872:0:0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-crf', '23', '-movflags', '+faststart', str(target)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    report = {'recorded_utc': datetime.now(timezone.utc).isoformat(),
              'recording': 'Continuous Playwright-driven real browser; actual HTTP inference; real wait time; no synthetic narration or generated images.',
              'scope': 'Public photographs only; isolated database. Digit ResNet demonstration is supplied separately.',
              'video_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
              'page_errors': errors, 'samples': rows}
    (OUT/'recording.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(target, flush=True)


if __name__ == '__main__':
    asyncio.run(main())
