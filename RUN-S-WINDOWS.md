# Install and run S on Windows 11

## First use

1. Extract the entire S ZIP to a folder such as `C:\S`. **Do not run the program from inside the ZIP.**
2. Install Python 3.11 or newer from https://www.python.org/downloads/windows/ if `py -3 --version` does not work. Include the Python launcher. No Python packages are required for the default S Research mode.
3. Double-click `START-S-WINDOWS.cmd`. Keep the opened terminal running.
4. If your browser does not open, visit http://127.0.0.1:8765 yourself.
5. Choose S Research to begin; its answers are deterministic research scaffolds, **not a full language model**.

## Add a capable local language model (optional, after installing S)

1. Install Ollama for Windows using the official installer at https://ollama.com/download/windows.
2. Open PowerShell and run `ollama pull qwen3:1.7b` for a smaller model (~1.4 GB download), or `ollama pull qwen3:4b` for a larger model (~2.5 GB download). These are third-party open-weight models, not S-original inventions.
3. Restart S or click Refresh models under Settings. Select the model you downloaded. Inference is local; initial installation and model download use the internet.
4. S explicitly blocks Ollama cloud model labels. Do not enable remote API proxies if preserving local-only data is essential.

## Move our teacher–student history into S

1. In ChatGPT open Settings → Data controls → Export data, or use the Privacy Portal. Save the export ZIP privately on your PC.
2. Open PowerShell inside your S folder. Run:

   `py -3 -m continuum.cli import-chatgpt "C:\Users\YOUR_NAME\Downloads\your-export.zip"`

3. Import creates a separate local project, storing eligible text from the export in S's private database. It does not train a model immediately, and it cannot recover messages omitted from the export.
4. Search lessons in S. Add corrections and evidence using Knowledge.
5. For experimental **original-model** training on the private archive, install PyTorch and follow `docs/S-CONTINUITY.md`. The small transformer is a research experiment; use an installed open-weight model for stronger day-to-day assistance.

## Back up before the subscription ends

Run in PowerShell from your S folder:

`py -3 -m continuum.cli backup --output "$HOME\Documents\S-backup.sqlite3"`

Keep your original ChatGPT export ZIP, S source ZIP, model weights and S database backup on two drives. The local database is `%USERPROFILE%\.s-computer\s.sqlite3`.

## Project limitations

S is a local application running on Windows, not yet a separately bootable S operating system. The voice button explains the missing offline speech model; it doesn't fake listening. External web search opens a browser only after confirmation and does not automatically import search results. Basic text attachments are supported; binary/image/PDF parsing isn't yet integrated. The old Axiom/Continuum foundation and the original byte-token model are bundled for continued development.
