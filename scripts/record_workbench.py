"""Record the running service. Public dataset examples, no mocked API responses."""
import asyncio
import io
import json
import os
import time
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image
from playwright.async_api import async_playwright
from sklearn.datasets import load_digits

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/review/demo'

async def main():
    OUT.mkdir(parents=True,exist_ok=True)
    # Render an actual public-data sample; this is not a newly collected handwriting image.
    pixels=np.uint8(load_digits().images[0]/16*255)
    sample=OUT/'uci-sample-0.png'
    Image.fromarray(pixels).resize((160,160),Image.Resampling.NEAREST).save(sample)
    events=[];errors=[];running=True
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path=os.environ.get('CHROME_PATH','/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'),headless=True)
        page=await browser.new_page(viewport={'width':1440,'height':1080},device_scale_factor=1)
        page.on('pageerror',lambda error:errors.append(str(error)))
        await page.goto('http://127.0.0.1:8766',wait_until='networkidle')
        await page.wait_for_function("document.getElementById('connection').textContent.includes('已就绪')")
        writer=imageio.get_writer(str(OUT/'运行演示.mp4'),fps=5,codec='libx264',quality=8,macro_block_size=2)
        start=time.monotonic();frames=0
        async def record():
            nonlocal frames
            while running:
                tick=time.monotonic()
                shot=await page.screenshot()
                writer.append_data(np.asarray(Image.open(io.BytesIO(shot)).convert('RGB')))
                frames+=1
                await asyncio.sleep(max(0,.2-(time.monotonic()-tick)))
        task=asyncio.create_task(record())
        async def stage(label,seconds=3):
            events.append({'wall_seconds':round(time.monotonic()-start,2),'video_seconds':round(frames/5,2),'action':label})
            await asyncio.sleep(seconds)
        try:
            await stage('本机工作区；以下由自动化浏览器操作')
            await page.locator('#load-demo').click()
            await page.wait_for_function("document.querySelectorAll('.card').length===36")
            await stage('载入公开测试集 270 张，按置信度从低到高排序',4)
            await page.locator('.card').filter(has_text='test-555.png').click()
            await page.locator('#reviewer').fill('自动化录制（演示）')
            await page.locator('#digits button').filter(has_text='9').click()
            await page.locator('#note').fill('依据公开数据集 sample 555 的原始标签 9 演示纠错；不是新增人工标注。')
            await stage('原预测为 3；根据已有真值演示纠正为 9',4)
            await page.locator('#save').click()
            await page.wait_for_function("document.getElementById('saved-info').textContent.includes('已保存标签 9')")
            await page.locator('summary').filter(has_text='修改记录').click()
            await stage('查看署名、版本和修改记录；原预测仍保留',4)
            await page.screenshot(path=str(OUT/'复核记录.png'))
            await page.reload(wait_until='networkidle')
            await page.locator('.card').filter(has_text='test-555.png').click()
            await page.wait_for_function("document.getElementById('saved-info').textContent.includes('已保存标签 9')")
            await stage('刷新后仍能读取已保存标签',3)
            async with page.expect_download() as download:
                await page.locator('#export').click()
            await (await download.value).save_as(OUT/'演示导出.csv')
            await stage('导出本批次 CSV，未复核项的最终标签留空')
            await page.locator('#new-batch').click()
            await page.locator('#batch-name').fill('公开样本 0 · 重复文件导入演示')
            payload={'name':'uci-sample-0.png','mimeType':'image/png','buffer':sample.read_bytes()}
            await page.locator('#files').set_input_files([payload,dict(payload,name='same-sample.png')])
            await page.locator('#polarity').select_option('light')
            await stage('导入同一公开样本两次，验证文件去重')
            await page.locator('#submit-import').click()
            await page.wait_for_function("document.getElementById('notice').textContent.includes('已导入 1 张')")
            await page.locator('.card').first.click()
            await page.locator('#original').wait_for(state='visible')
            await stage('一条记录；保留上传原图和模型实际输入',4)
            await page.screenshot(path=str(OUT/'原图与输入.png'))
            await page.locator('#experiment-tab').click()
            await page.wait_for_function("document.querySelectorAll('#experiment-table tr').length===3")
            await page.locator('#experiment-table').scroll_into_view_if_needed()
            await stage('回顾性队列实验及随机对照；不冒充人工提效测量',5)
            await page.screenshot(path=str(OUT/'队列实验.png'),full_page=True)
        finally:
            running=False
            await task
            writer.close()
        await page.set_viewport_size({'width':390,'height':844})
        await page.goto('http://127.0.0.1:8766',wait_until='networkidle')
        overflow=await page.evaluate('document.documentElement.scrollWidth > innerWidth')
        await page.screenshot(path=str(OUT/'移动端.png'),full_page=True)
        await browser.close()
    checks={'page_errors':errors,'mobile_horizontal_overflow':overflow,'events':events,'frames':frames,
        'video_seconds':frames/5,'recording':'Actual Chrome screenshots at nominal 5 fps; automated interactions, no audio. Encoding time may differ from wall time.',
        'sample_source':'sklearn load_digits sample 0 rendered with nearest-neighbor scaling; review uses existing truth for sample 555; no new human labeling study.'}
    (OUT/'browser_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
    if errors or overflow:raise RuntimeError('Browser checks failed')
    print(json.dumps(checks,ensure_ascii=False,indent=2))

if __name__=='__main__':asyncio.run(main())
