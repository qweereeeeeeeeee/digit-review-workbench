"""Real browser checks. User photographs and full screenshots never enter Git."""
import asyncio,json,os
from pathlib import Path
from playwright.async_api import async_playwright
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'var/round3/browser'
async def main():
 OUT.mkdir(parents=True,exist_ok=True);errors=[];rows=[]
 async with async_playwright() as p:
  browser=await p.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless=True)
  page=await browser.new_page(viewport={'width':1440,'height':1000},locale='zh-CN');page.set_default_timeout(180000)
  page.on('pageerror',lambda e:errors.append(str(e)))
  await page.goto('http://127.0.0.1:8768',wait_until='networkidle');await page.wait_for_function("document.querySelector('#health').textContent.includes('OCR')")
  # All previous private photos are exercised through the actual upload API.
  private=[x for x in json.loads((ROOT/'var/round3/baseline.json').read_text()) if x['name'].startswith('微信')]
  inputs=private+[{'name':n+'.png','path':'artifacts/objects/examples/'+n+'.png'} for n in ['rocket','astronaut','coffee']]
  for idx,item in enumerate(inputs):
   await page.locator('#file').set_input_files(str(ROOT/item['path']))
   async with page.expect_response(lambda r:'/api/recognize' in r.url and r.request.method=='POST') as response:
    await page.locator('#recognize').click()
   r=await response.value;assert r.status==200,await r.text();record=await r.json()
   await page.wait_for_function("!document.querySelector('#recognize').disabled")
   assert record['result']['vision'] is not None
   await page.screenshot(path=str(OUT/f'{idx}-actual.png'),full_page=True)
   rows.append({'name':item['name'],'id':record['id'],'result':record['result']})
   (OUT/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
   print(item['name'],[t['text'] for t in record['result']['vision']['texts']],[t['label'] for t in record['result']['vision']['regions']],flush=True)
   if item['name']=='rocket.png':assert 'rocket' in [t['label'] for t in record['result']['vision']['regions']]
   if item['name']=='astronaut.png':assert len(record['result']['vision']['regions'])>=3
  await page.locator('#final-label').fill('咖啡杯、勺子');await page.locator('#note').fill('新版浏览器流程核对')
  await page.locator('#confirm').click();await page.wait_for_function("document.querySelector('#saved').textContent.includes('已确认')")
  await page.reload(wait_until='networkidle');await page.locator('.record').first.click()
  await page.wait_for_function("document.querySelector('#final-label').value==='咖啡杯、勺子'")
  assert await page.locator('#final-label').input_value()=='咖啡杯、勺子'
  assert not await page.locator('#reading').is_hidden()
  # Validate new structured results survive export, not only the screen.
  response=await page.request.get('http://127.0.0.1:8768/api/export');assert response.ok
  assert 'florence_revision' in await response.text()
  await page.set_viewport_size({'width':390,'height':844})
  assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth')
  assert not errors,errors
  report={'passed':True,'real_uploads':len(inputs),'page_errors':errors,'checks':['private originals','rocket region','multiple regions','save and reload','CSV contains provenance','390px layout']}
  (ROOT/'artifacts/vision/browser-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
  await browser.close()
 print('General-vision browser checks passed')
if __name__=='__main__':asyncio.run(main())
