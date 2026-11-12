"""Exercise actual UI and API; only public photographs used."""
import asyncio,json,os
from pathlib import Path
from playwright.async_api import async_playwright
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/objects/demo'

async def main():
 OUT.mkdir(parents=True,exist_ok=True);errors=[]
 async with async_playwright() as p:
  browser=await p.chromium.launch(executable_path=os.environ.get('CHROME_PATH','/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'),headless=True)
  page=await browser.new_page(viewport={'width':1440,'height':1000},locale='zh-CN')
  page.on('pageerror',lambda e:errors.append(str(e)))
  await page.goto('http://127.0.0.1:8768',wait_until='networkidle')
  await page.wait_for_function("document.querySelector('#health').textContent.includes('已就绪')")
  await page.locator('#file').set_input_files(str(ROOT/'artifacts/objects/examples/coffee.png'))
  await page.locator('#recognize').click()
  await page.locator('#detections').get_by_text('杯子',exact=False).wait_for()
  await page.screenshot(path=str(OUT/'coffee-detection.png'),full_page=True)
  await page.locator('#final-label').fill('杯子、勺子、桌子')
  await page.locator('#note').fill('公开照片的浏览器流程检查')
  await page.locator('#confirm').click()
  await page.wait_for_function("document.querySelector('#saved').textContent.includes('已确认')")
  await page.reload(wait_until='networkidle')
  await page.locator('.record').first.click()
  await page.wait_for_function("document.querySelector('#saved').textContent.includes('已确认')")
  assert await page.locator('#final-label').input_value()=='杯子、勺子、桌子'
  await page.locator('#file').set_input_files(str(ROOT/'artifacts/objects/examples/rocket.png'))
  await page.locator('#recognize').click()
  await page.wait_for_function("document.querySelector('#detection-summary').textContent.startsWith('0 个')")
  await page.screenshot(path=str(OUT/'empty-detection.png'),full_page=True)
  await page.locator('#unknown').click()
  await page.wait_for_function("document.querySelector('#saved').textContent.includes('无法判断')")
  await page.locator('#scope').click();await page.locator('#search').fill('cup')
  assert await page.locator('#category-list').inner_text()
  await page.locator('#close').click()
  await page.locator('#file').set_input_files(str(ROOT/'artifacts/objects/examples/chelsea.png'))
  await page.locator('#recognize').click()
  await page.locator('#detections').get_by_text('猫',exact=False).wait_for()
  box=await page.locator('#canvas').bounding_box()
  await page.mouse.move(box['x']+box['width']*.1,box['y']+box['height']*.1);await page.mouse.down()
  await page.mouse.move(box['x']+box['width']*.9,box['y']+box['height']*.9,steps=10);await page.mouse.up()
  await page.locator('#recognize').click()
  await page.wait_for_function("!document.querySelector('#recognize').disabled")
  assert '猫' in await page.locator('#detections').inner_text()
  await page.screenshot(path=str(OUT/'crop-detection.png'),full_page=True)
  async with page.expect_download() as event:await page.locator('#export').click()
  download=await event.value;await download.save_as(str(OUT/'browser-export.csv'))
  await page.set_viewport_size({'width':390,'height':844})
  assert await page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
  await page.screenshot(path=str(OUT/'mobile.png'),full_page=True)
  assert not errors,errors
  (OUT/'browser-check.json').write_text(json.dumps(dict(passed=True,page_errors=errors,checks=['real upload and detection','save and reload','empty detection and unknown review','category search','manual crop and re-inference','CSV download','390px no horizontal overflow']),ensure_ascii=False,indent=2)+'\n')
  await browser.close()
 print('Browser checks passed')

if __name__=='__main__':asyncio.run(main())
