"""Continuous browser recording of real local inference on attributed public photos."""
import asyncio,os,json,subprocess,time
from pathlib import Path
import imageio_ffmpeg
from playwright.async_api import async_playwright
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/objects/demo'
async def main():
 runtime=ROOT/'var/browser-runtime';encoder=runtime/'ffmpeg-1011/ffmpeg-mac';encoder.parent.mkdir(parents=True,exist_ok=True)
 if not encoder.exists():encoder.symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
 os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(runtime)
 async with async_playwright() as p:
  b=await p.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless=True)
  c=await b.new_context(viewport={'width':1440,'height':1000},locale='zh-CN',record_video_dir=str(OUT/'capture'),record_video_size={'width':1440,'height':1000})
  page=await c.new_page();await page.goto('http://127.0.0.1:8768',wait_until='networkidle')
  await asyncio.sleep(2)
  for name,label in [('coffee','杯子'),('astronaut','人'),('rocket',None)]:
   await page.locator('#file').set_input_files(str(ROOT/f'artifacts/objects/examples/{name}.png'));await asyncio.sleep(2)
   await page.locator('#recognize').click();await page.wait_for_function("!document.querySelector('#recognize').disabled")
   await asyncio.sleep(6)
   if name=='astronaut':
    await page.locator('#final-label').fill('人、头盔');await page.locator('#note').fill('头盔被误报成球，人工更正。');await asyncio.sleep(2)
    await page.locator('#confirm').click();await page.wait_for_function("document.querySelector('#saved').textContent.includes('已确认')");await asyncio.sleep(3)
   if name=='rocket':
    await page.locator('#unknown').click();await page.wait_for_function("document.querySelector('#saved').textContent.includes('无法判断')");await asyncio.sleep(3)
  await c.close();video=await page.video.path();await b.close()
 subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-i',video,'-vf','crop=1440:912:0:0','-c:v','libx264','-pix_fmt','yuv420p','-crf','23','-movflags','+faststart',str(OUT/'物体识别演示.mp4')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
 (OUT/'recording.json').write_text(json.dumps(dict(source='continuous Playwright browser recording, real HTTP API and official checkpoint',samples=['coffee','astronaut','rocket'],generated_images=False,edited_inference=False,includes_failures=['helmet false positive as sports ball','rocket no detection']),ensure_ascii=False,indent=2)+'\n')
 print('Recorded real object detection walkthrough')
if __name__=='__main__':asyncio.run(main())
