"""Record actual local browser frames; does not reconstruct or mock inference UI."""
import asyncio
import io
import json
import time
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image
from playwright.async_api import async_playwright

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/demo'


async def main():
    OUT.mkdir(parents=True,exist_ok=True)
    events=[]; errors=[]; running=True
    start=time.monotonic()
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless=True)
        page=await browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
        page.on('pageerror',lambda error:errors.append(str(error)))
        await page.goto('http://127.0.0.1:8765',wait_until='networkidle')
        await page.wait_for_function("document.getElementById('health').textContent.includes('已连接')")
        writer=imageio.get_writer(str(OUT/'运行演示.mp4'),fps=5,codec='libx264',quality=8,macro_block_size=2)
        async def record():
            next_tick=time.monotonic()
            while running:
                shot=await page.screenshot()
                writer.append_data(np.asarray(Image.open(io.BytesIO(shot)).convert('RGB')))
                next_tick+=.2
                await asyncio.sleep(max(0,next_tick-time.monotonic()))
        task=asyncio.create_task(record())
        async def stage(label, seconds=4):
            events.append({'seconds':round(time.monotonic()-start,2),'action':label})
            await asyncio.sleep(seconds)
        await stage('首页：展示真实实验指标',5)
        await page.locator('#lab').scroll_into_view_if_needed()
        await stage('识别实验室：加载真实测试样本',2)
        for i in range(3):
            await page.locator('#sample').click()
            await page.locator('#predict').click()
            await page.wait_for_function("!document.getElementById('predict').disabled && !document.getElementById('result').hidden")
            await stage(f'真实样本 {i+1}：模型预测、置信度及遮挡分析',4)
        await page.screenshot(path=str(OUT/'实验室截图.png'))
        await page.locator('#clear').click()
        await page.locator('#predict').click()
        await page.wait_for_function("document.getElementById('error').textContent.length > 0")
        await stage('空白输入：实际 422 错误反馈',3)
        # A genuinely new hand-drawn input. Outcome is not preselected.
        box=await page.locator('#canvas').bounding_box()
        await page.mouse.move(box['x']+80,box['y']+65)
        await page.mouse.down()
        await page.mouse.move(box['x']+205,box['y']+65,steps=16)
        await page.mouse.move(box['x']+125,box['y']+225,steps=22)
        await page.mouse.up()
        await page.locator('#predict').click()
        await page.wait_for_function("!document.getElementById('predict').disabled && !document.getElementById('result').hidden")
        await stage('新手绘数字：真实推理，不预设正确结果',5)
        await page.locator('#evidence').scroll_into_view_if_needed()
        await stage('实际训练曲线、可靠性及压力测试',6)
        await page.locator('.note').scroll_into_view_if_needed()
        await stage('混淆矩阵、数据分布与适用边界',5)
        await page.screenshot(path=str(OUT/'完整页面.png'),full_page=True)
        running=False
        await task
        writer.close()
        # Separate mobile inspection, not part of desktop recording.
        await page.set_viewport_size({'width':390,'height':844})
        await page.goto('http://127.0.0.1:8765',wait_until='networkidle')
        overflow=await page.evaluate('document.documentElement.scrollWidth > innerWidth')
        await page.screenshot(path=str(OUT/'移动端.png'),full_page=True)
        await browser.close()
    (OUT/'browser_checks.json').write_text(json.dumps({'page_errors':errors,'mobile_horizontal_overflow':overflow,'events':events,'recording':'Actual headless Chrome screenshots sampled at 5fps; no audio; wall time may differ slightly from encoded time'},ensure_ascii=False,indent=2))
    if errors or overflow: raise RuntimeError('Browser check failed; inspect browser_checks.json')
    print('Recorded actual browser run:',OUT/'运行演示.mp4')

if __name__=='__main__':asyncio.run(main())
