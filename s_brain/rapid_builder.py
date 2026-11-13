"""S-Brain rapid artifact prototyping: offline, standard-library-only generation.

This module writes source files, never installs or executes generated apps.
Android output is a source scaffold, not a signed APK.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

CALCULATOR_HTML = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>S Calculator</title>
<style>body{font:20px system-ui;background:#151c2c;color:white;display:grid;place-items:center;min-height:90vh}
main{width:min(95vw,360px)}input{box-sizing:border-box;width:100%;font-size:32px;padding:15px;background:#26354d;color:white;border:0}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:12px}
button{padding:20px 0;font-size:22px;border:0;border-radius:9px;background:#394d6b;color:white}</style>
<main><h1>S Calculator</h1><input id="screen" aria-label="Expression" readonly>
<div class="grid" id="keys"></div><p id="status" role="status"></p></main>
<script>
const screen=document.getElementById('screen'), keys=document.getElementById('keys');
const labels=['7','8','9','/','4','5','6','*','1','2','3','-','C','0','.','+','='];
function calculate(expr){
 const tokens=expr.match(/(?:\\d+(?:\\.\\d*)?|\\.\\d+)|[+*/-]/g);
 if(!tokens||tokens.join('')!==expr)throw Error('Invalid expression');
 let nums=[],ops=[],expectNumber=true;
 for(const token of tokens){
  if(expectNumber){if(!/^(?:\\d+(?:\\.\\d*)?|\\.\\d+)$/.test(token))throw Error('Number expected');nums.push(Number(token));}
  else {if(!['+','-','*','/'].includes(token))throw Error('Operator expected');ops.push(token);}
  expectNumber=!expectNumber;
 }
 if(!nums.length||expectNumber)throw Error('Incomplete expression');
 for(let i=0;i<ops.length;){if(ops[i]==='*'||ops[i]==='/'){
  const n=ops[i]==='*'?nums[i]*nums[i+1]:nums[i]/nums[i+1];
  if(!Number.isFinite(n))throw Error('Division by zero');
  nums.splice(i,2,n);ops.splice(i,1);
 }else i++;}
 let result=nums[0];ops.forEach((op,i)=>result=op==='+'?result+nums[i+1]:result-nums[i+1]);
 if(!Number.isFinite(result))throw Error('Invalid result');
 return String(Number(result.toPrecision(12)));
}
for(const label of labels){const b=document.createElement('button');b.textContent=label;b.onclick=()=>{
 const status=document.getElementById('status');status.textContent='';
 if(label==='C'){screen.value='';return;}
 if(label==='='){try{screen.value=calculate(screen.value)}catch(e){status.textContent=e.message;}return;}
 if(screen.value.length<80)screen.value+=label;
};keys.appendChild(b);}
</script>"""

CHESS_HTML = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>S Chess Board</title><style>body{font:18px system-ui;text-align:center}#board{display:grid;grid-template-columns:repeat(8,1fr);width:min(95vw,560px);margin:auto}
button{aspect-ratio:1;border:0;font-size:clamp(22px,5vw,45px)}button:nth-child(16n+1),button:nth-child(16n+3),button:nth-child(16n+5),button:nth-child(16n+7),button:nth-child(16n+10),button:nth-child(16n+12),button:nth-child(16n+14),button:nth-child(16n+16){background:#b7c6a6}button{background:#eee8d5}</style>
<h1>S Chess Board</h1><p>Interactive board prototype — not a complete chess rules engine.</p><p id="status"></p><div id="board"></div>
<script>
const pieces=['♜♞♝♛♚♝♞♜','♟♟♟♟♟♟♟♟','','','','','♙♙♙♙♙♙♙♙','♖♘♗♕♔♗♘♖'];
const board=Array.from({length:64},(_,i)=>pieces[Math.floor(i/8)]?.[i%8]||'');
let selected=null;const root=document.getElementById('board');
function draw(){root.replaceChildren();board.forEach((piece,i)=>{const b=document.createElement('button');b.textContent=piece;b.setAttribute('aria-label','Square '+i);b.onclick=()=>{
if(selected===null){if(piece){selected=i;document.getElementById('status').textContent='Selected square '+i;}return;}
board[i]=board[selected];board[selected]='';selected=null;draw();
};root.appendChild(b);});}draw();
</script>"""

WEBSITE_HTML = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>S Website</title><style>body{font:18px system-ui;margin:0;color:#152337}header{background:#172e56;color:white;padding:4rem 2rem}main{max-width:800px;margin:3rem auto;padding:1rem}a{color:#2167c5}</style>
<header><h1>Welcome to S Website</h1><p>Built locally with S-Brain.</p></header>
<main><h2>About</h2><p>Replace this starter content with your business details.</p><a href="mailto:hello@example.invalid">Contact placeholder</a></main></html>"""

ANDROID_MANIFEST = """<manifest xmlns:android="http://schemas.android.com/apk/res/android"><application android:label="S Calculator" android:theme="@android:style/Theme.Material.Light.NoActionBar" android:usesCleartextTraffic="false"><activity android:name=".MainActivity" android:exported="true"><intent-filter><action android:name="android.intent.action.MAIN"/><category android:name="android.intent.category.LAUNCHER"/></intent-filter></activity></application></manifest>"""
ANDROID_ACTIVITY = """package com.sbrain.calculator;
import android.app.Activity;
import android.os.Bundle;
import android.webkit.WebView;
import android.webkit.WebViewClient;
public class MainActivity extends Activity {
  @Override public void onCreate(Bundle state) {
    super.onCreate(state);
    WebView view = new WebView(this);
    view.getSettings().setJavaScriptEnabled(true);
    view.setWebViewClient(new WebViewClient());
    setContentView(view);
    view.loadUrl("file:///android_asset/index.html");
  }
}
"""
ANDROID_GRADLE = """plugins { id 'com.android.application' version '8.7.3' }
android { namespace 'com.sbrain.calculator'; compileSdk 35
 defaultConfig { applicationId 'com.sbrain.calculator'; minSdk 23; targetSdk 35 } }
"""

def classify_request(request: str) -> str:
    if not isinstance(request, str) or not 1 <= len(request.strip()) <= 4000:
        raise ValueError("Request must contain 1-4000 characters")
    words = set(re.findall(r"[a-z]+", request.casefold()))
    if "calculator" in words and words.intersection({"android", "phone", "mobile", "apk", "install"}):
        return "android-calculator"
    if "calculator" in words:
        return "calculator"
    if "chess" in words:
        return "chess"
    if words.intersection({"website", "webpage", "site"}):
        return "website"
    raise ValueError("Supported requests: calculator, Android calculator, chess board, website")

def generate(request: str, destination: Path) -> dict:
    kind = classify_request(request)
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Destination must be empty to protect existing files")
    files = {"index.html": CALCULATOR_HTML if "calculator" in kind else
             CHESS_HTML if kind == "chess" else WEBSITE_HTML}
    if kind == "android-calculator":
        files = {
            "app/src/main/assets/index.html": CALCULATOR_HTML,
            "app/src/main/AndroidManifest.xml": ANDROID_MANIFEST,
            "app/src/main/java/com/sbrain/calculator/MainActivity.java": ANDROID_ACTIVITY,
            "app/build.gradle": ANDROID_GRADLE,
            "settings.gradle": "pluginManagement { repositories { google(); mavenCentral(); gradlePluginPortal() } }\n"
                               "dependencyResolutionManagement { repositories { google(); mavenCentral() } }\n"
                               "rootProject.name = 'SCalculator'\ninclude ':app'\n",
        }
    for name, content in files.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return {"kind": kind, "files": sorted(files), "directory": str(destination),
            "installed": False, "build_verified": False,
            "note": "Source generated; Android requires an SDK build and device installation."}

def main(argv=None):
    parser = argparse.ArgumentParser(description="S-Brain fast source generator")
    parser.add_argument("request")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.request, args.output), indent=2))

if __name__ == "__main__":
    main()
