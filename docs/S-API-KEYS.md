# S-issued API keys

S can issue its own random keys for locally running programs. This is not an OpenAI or another third-party API key.

## Generate in S

Open **Settings → S API keys → Generate API key**. Give it a name and choose permissions (read, chat, or both). Copy the key immediately: only its SHA-256 digest remains stored in SQLite.

To revoke a key, press **Revoke** in the same panel.

## Generate in PowerShell

```powershell
cd C:\S
py -3 -m continuum.cli api-key create --name "My Windows Tool" --scopes read,chat
py -3 -m continuum.cli api-key list
py -3 -m continuum.cli api-key revoke YOUR_KEY_ID
```

## Call the local S API

```powershell
$SKey = Read-Host "Paste your S API key"
Invoke-RestMethod -Uri http://127.0.0.1:8765/v1/models -Headers @{Authorization="Bearer $SKey"}
```

`POST /v1/chat` accepts `session_id`, `text`, and optional `model`. Create a session through the local UI first. The `chat` permission is required.

## Security boundary

S listens on loopback only. Bearer keys protect the **/v1/** routes. Existing **/api/** routes are intentionally available to the local browser UI and are not protected by API keys. Keys are therefore suitable for local integrations, not for publishing S as a multi-user internet service. Never forward port 8765 to the internet. Backup files containing chat history must stay private.
