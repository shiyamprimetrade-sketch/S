"""Optional manual UI smoke test. Requires playwright plus Chromium installed.

The hardened browser environment used for validation blocked all HTTP navigation;
this script loads actual production DOM/CSS/JS, using an in-process proxy to
exercise the real S local HTTP handlers. No fake S replies are generated.
Run: python tests/manual_browser_smoke.py (after pip install playwright).
"""
import tempfile, threading, json
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from playwright.sync_api import sync_playwright
from continuum.server import AppServer

root=Path(__file__).resolve().parent.parent
with tempfile.TemporaryDirectory() as temp:
    server=AppServer('127.0.0.1',0,Path(temp)/'s.sqlite3')
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        with sync_playwright() as p:
            import shutil
            executable=shutil.which('chromium') or shutil.which('google-chrome')
            launch_opts={'headless':True}
            if executable:launch_opts['executable_path']=executable
            browser=p.chromium.launch(**launch_opts)
            page=browser.new_page(viewport={'width':1280,'height':840})
            errors=[]
            page.on('pageerror',lambda e:errors.append(str(e)))
            # Browser sandbox forbids navigation even to localhost. Load the actual
            # production HTML/JS and proxy local fetches through the Python test harness.
            def backend(url, method, raw):
                method=method.upper()
                req=Request('http://127.0.0.1:'+str(server.server_port)+url,
                            method=method,
                            data=raw.encode('utf-8') if raw is not None else None,
                            headers={'Content-Type':'application/json'})
                try:
                    with urlopen(req,timeout=15) as r:
                        return {'status':r.status,'body':json.loads(r.read())}
                except HTTPError as e:
                    return {'status':e.code,'body':json.loads(e.read())}
            page.expose_function('localBackend',backend)
            html=(root/'web/index.html').read_text()
            html=html.replace('<link rel="stylesheet" href="/style.css">','')
            html=html.replace('<script src="/app.js" defer></script>','')
            page.set_content(html,wait_until='load')
            page.add_style_tag(content=(root/'web/style.css').read_text())
            page.evaluate('''(()=>{window.fetch = async (path, opts={})=>{
              const resp=await window.localBackend(path,opts.method||'GET',opts.body===undefined?null:opts.body);
              return new Response(JSON.stringify(resp.body),{status:resp.status,headers:{'Content-Type':'application/json'}})
            }})()''')
            page.evaluate("Object.defineProperty(window,'localStorage',{value:{getItem:()=>null,setItem:()=>{}},configurable:true})")
            page.add_script_tag(content=(root/'web/app.js').read_text())
            page.wait_for_timeout(1000)
            assert 'Local · Ready' in page.locator('#status-label').inner_text()
            page.click('#knowledge-open')
            page.wait_for_selector('#memory-teach',state='visible')
            page.click('#memory-teach')
            page.fill('#field-title','Name S')
            page.fill('#field-principle','The single symbol S can transform an existing word in context.')
            page.fill('#field-cues','Why choose the name S?\nHow can a symbol transform a word?')
            page.click('#dialog-save')
            page.locator('.memory-card').first.wait_for(timeout=5000)
            assert 'Name S' in page.locator('.memory-card').first.inner_text()
            print('Teacher UI saved: PASS')
            page.fill('#memory-query','Why choose the name S?')
            page.click('#memory-recall')
            page.locator('.memory-match').first.wait_for(timeout=5000)
            print('Recall UI: PASS')
            page.click('#memory-concept')
            page.fill('#field-label','reasoning')
            page.fill('#field-variants','deep thinking\ncareful investigation')
            page.click('#dialog-save')
            page.get_by_text('Meaning equivalence saved',exact=False).wait_for(timeout=5000)
            print('Phrase concept UI: PASS')
            page.click('#apprentice-add')
            page.fill('#field-question','Apply careful investigation to a new design')
            # This test is a no-match control; adding requires no selected lesson.
            page.click('#dialog-save')
            page.locator('.apprentice-row').first.wait_for(timeout=5000)
            page.click('#apprentice-run')
            page.locator('.apprentice-score').first.wait_for(timeout=5000)
            assert 'Last run:' in page.locator('.apprentice-score').first.inner_text()
            print('Held-out teacher evaluation UI: PASS')
            page.locator('#knowledge-panel .back-chat').click()
            page.fill('#chat-input','Why choose the name S?')
            page.click('#send-btn')
            page.wait_for_timeout(1500)
            assert page.locator('.message-row').count() == 2
            page.locator('.answer').first.wait_for(timeout=5000)
            assert 'Name S' in page.locator('.answer').first.inner_text()
            print('Chat consumes teacher memory: PASS')
            print('Browser JS errors:',errors)
            assert not errors
            browser.close()
    finally:
        server.shutdown();server.server_close();thread.join(timeout=2)
