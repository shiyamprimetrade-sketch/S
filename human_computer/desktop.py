"""Windows desktop interface; entirely local and stdlib Tkinter."""
from __future__ import annotations
import json

def run(pc):
    import tkinter as tk
    from tkinter import ttk
    root=tk.Tk()
    root.title('Axiom Human Computer — Research Desktop')
    root.geometry('1040x690')
    tabs=ttk.Notebook(root)
    tabs.pack(fill='both',expand=True,padx=12,pady=12)
    interact=ttk.Frame(tabs); tabs.add(interact,text='Think · Act · Learn')
    machine=ttk.Frame(tabs); tabs.add(machine,text='Computer · Evidence')
    history=tk.Text(interact,wrap='word',font=('Consolas',11))
    history.pack(fill='both',expand=True,padx=10,pady=10)
    bar=ttk.Frame(interact); bar.pack(fill='x',padx=10,pady=10)
    prompt=ttk.Entry(bar,font=('Consolas',11));prompt.pack(side='left',fill='x',expand=True)
    def submit(event=None):
        msg=prompt.get().strip()
        if not msg:return
        prompt.delete(0,'end')
        result=pc.respond(msg)
        history.insert('end',f'YOU  {msg}\nAXIOM [{result.status}]  {result.text}\n\n')
        history.see('end')
    def generate():
        msg=prompt.get().strip()
        if not msg:return
        prompt.delete(0,'end')
        history.insert('end',f'YOU [GENERATE] {msg}\nAXIOM [RUNNING LOCAL MODEL] …\n')
        history.see('end')
        import threading
        def work():
            result=pc.generate_local(msg)
            def show():
                history.insert('end',f'AXIOM [{result.status}]  {result.text}\n\n')
                history.see('end')
            root.after(0,show)
        threading.Thread(target=work,daemon=True).start()
    ttk.Button(bar,text='Generate (original AI)',command=generate).pack(side='right',padx=5)
    ttk.Button(bar,text='Run goal/skill',command=submit).pack(side='right',padx=5)
    prompt.bind('<Return>',submit)
    ttk.Label(machine,text='Hardware and operating-system observations (read-only)').pack(pady=8)
    statusbox=tk.Text(machine,wrap='word',font=('Consolas',10))
    statusbox.pack(fill='both',expand=True,padx=12,pady=8)
    def refresh():
        statusbox.delete('1.0','end')
        statusbox.insert('end','SYSTEM STATUS\n'+json.dumps(pc.status(),indent=2))
        statusbox.insert('end','\n\nRECENT EVIDENCE\n'+json.dumps(pc.events(12),indent=2))
    ttk.Button(machine,text='Refresh verified state',command=refresh).pack(pady=10)
    history.insert('end','AXIOM HUMAN COMPUTER — independent local foundation\n')
    history.insert('end','Try: status | goal invent a learning language | teach mirror CAT => CATTAC\n\n')
    refresh();prompt.focus_set();root.mainloop()