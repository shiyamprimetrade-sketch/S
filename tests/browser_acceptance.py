import json, subprocess, sys, tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from playwright.sync_api import sync_playwright

def main():
    with tempfile.TemporaryDirectory() as root:
        from project_s.secure import SecureEngine
        e=SecureEngine(root)
        failed,_=e.create('Create calculator')
        e.transition(failed,'AWAITING_AUTHORIZATION','FAILED',{'error':'injected browser fixture'})
        p=subprocess.Popen([sys.executable,'-m','project_s.local_ui','--root',root,'--port','0'],stdout=subprocess.PIPE,text=True)
        try:
            url=json.loads(p.stdout.readline())['url']
            with sync_playwright() as pw:
                browser=pw.chromium.launch()
                page=browser.new_page(viewport={'width':390,'height':844})
                page.goto(url)
                def create(requirement):
                    page.locator('#req').fill(requirement)
                    page.get_by_role('button',name='Create plan').click()
                    page.get_by_role('button',name='Approve generation').click()
                    page.get_by_text('Export generated file').wait_for()
                create('Create calculator')
                page.get_by_placeholder('(2+3)*4').fill('(2+3)*4')
                page.get_by_role('button',name='Calculate',exact=True).click()
                page.locator('#actions pre').get_by_text('20',exact=True).wait_for()
                print('PASS calculator browser operation',flush=True)
                create('Create website for Cedar Studio')
                frame=page.frame_locator('iframe')
                frame.get_by_role('heading',name='Cedar Studio').wait_for()
                frame.get_by_role('link',name='Contact').click()
                frame.get_by_role('button',name='Show information').click()
                frame.get_by_text('Contact details pending configuration.').wait_for()
                assert frame.locator('html').evaluate('(e)=>e.scrollWidth <= e.clientWidth')
                print('PASS website browser navigation and mobile width',flush=True)
                create('Create inventory')
                page.get_by_placeholder('Product name').fill('Apple')
                page.locator('#actions input[type=number]').fill('4')
                page.get_by_role('button',name='register',exact=True).click()
                page.wait_for_function("document.querySelector('#actions pre').textContent.includes('Apple')")
                page.locator('#actions input[type=number]').fill('3')
                page.get_by_role('button',name='adjust',exact=True).click()
                page.wait_for_function("JSON.parse(document.querySelector('#actions pre').textContent).products[0][1]===7")
                print('PASS inventory browser registration and adjustment',flush=True)
                page.get_by_placeholder('Product name').fill('=1+1')
                page.locator('#actions input[type=number]').fill('2')
                page.get_by_role('button',name='register',exact=True).click()
                page.wait_for_function("document.querySelector('#actions pre').textContent.includes('=1+1')")
                with page.expect_download() as download:
                    page.get_by_role('button',name='Export inventory CSV').click()
                import csv
                with open(download.value.path(),encoding='utf-8',newline='') as f:
                    rows=list(csv.reader(f))
                assert rows==[['name','stock'],["'=1+1",'2'],['Apple','7']],rows
                print('PASS inventory CSV download and formula neutralization',flush=True)
                page.locator(f'#history button[data-task-id="{failed}"]').click()
                assert json.loads(page.locator('#out').text_content())['state']=='FAILED'
                page.get_by_role('button',name='Propose repair').click()
                page.get_by_role('button',name='Approve repair').wait_for()
                proposal=json.loads(page.locator('#out').text_content())
                forged=dict(proposal,attempt=2)
                rejected=page.evaluate("""async p=>{try{await api('/repair/execute',{proposal:p,approve:true});return false}catch(e){return true}}""",forged)
                assert rejected
                page.get_by_role('button',name='Cancel repair').click()
                page.get_by_role('button',name='Propose repair').click()
                page.get_by_role('button',name='Approve repair').click()
                page.get_by_text('Export generated file').wait_for()
                page.get_by_placeholder('(2+3)*4').fill('6*7')
                page.get_by_role('button',name='Calculate',exact=True).click()
                page.locator('#actions pre').get_by_text('42',exact=True).wait_for()
                assert e.get(failed)['state']=='FAILED'
                print('PASS graphical repair authorization and successor operation',flush=True)
                browser.close()
        finally:
            p.terminate()
            p.wait(timeout=10)

if __name__=='__main__':main()
