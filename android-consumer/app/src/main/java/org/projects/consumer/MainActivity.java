package org.projects.consumer;
import android.app.Activity;
import android.os.Bundle;
import android.webkit.*;
import android.widget.TextView;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
public class MainActivity extends Activity {
 private static String endpoint;
 @Override public void onCreate(Bundle state) {
  super.onCreate(state);
  TextView status=new TextView(this); status.setText("Starting Project S…"); setContentView(status);
  new Thread(()->{try {
   synchronized(MainActivity.class) {
    if(!Python.isStarted()) Python.start(new AndroidPlatform(getApplicationContext()));
    if(endpoint==null) endpoint=Python.getInstance().getModule("mobile_entry").callAttr("start",getFilesDir().getAbsolutePath()+"/workspace").toString();
   }
   runOnUiThread(()->{
    WebView web=new WebView(this); web.getSettings().setJavaScriptEnabled(true);
    web.getSettings().setAllowFileAccess(false); web.getSettings().setAllowContentAccess(false);
    web.setWebViewClient(new WebViewClient(){
     @Override public boolean shouldOverrideUrlLoading(WebView v, WebResourceRequest r){return !r.getUrl().toString().startsWith("http://127.0.0.1:8768/");}
    });
    setContentView(web); web.loadUrl(endpoint);
   });
  }catch(Exception e){runOnUiThread(()->status.setText("Engine startup failed: "+e.getMessage()));}}).start();
 }
}
