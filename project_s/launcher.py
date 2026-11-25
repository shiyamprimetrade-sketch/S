"""Launch the local consumer application using the installed distribution."""
import argparse
from pathlib import Path
from .local_ui import serve

def main():
    parser=argparse.ArgumentParser(description="Project S local consumer workspace")
    parser.add_argument("--root",default=str(Path.home()/".project-s"))
    parser.add_argument("--port",type=int,default=8768)
    args=parser.parse_args()
    serve(args.root,args.port)

if __name__=="__main__":
    main()
